// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Augmented Lagrangian method
//!
//! The augmented Lagrangian method minimizes a cost `f(x)` under equality constraints
//! `c(x) = 0`. Each outer iteration minimizes the augmented Lagrangian
//!
//! `L_A(x; λ, μ) = f(x) + λᵀc(x) + (μ/2)·‖c(x)‖²`
//!
//! with an unconstrained inner solver, then updates the Lagrange multipliers `λ ← λ + μ·c(x)`.
//! The multipliers move the minimizer of `L_A` onto the constraints, so the method meets them
//! with a finite penalty weight `μ`. A quadratic penalty alone needs `μ → ∞`, and a large `μ`
//! makes a narrow valley that is hard for the inner solver.
//!
//! The inner solver can be any argmin solver, derivative-free ones included. A factory builds it
//! anew at each outer iteration, from the current parameter vector.
//!
//! See [`AugmentedLagrangian`] for details.
//!
//! ## Sign convention of the multipliers
//!
//! The Lagrangian is `L(x, λ) = f(x) + λᵀc(x)`. At a solution, `∇f(x) + J(x)ᵀλ = 0`, and
//! `-λᵢ` is the derivative of the optimal cost with respect to `bᵢ` when constraint `i` is
//! `cᵢ(x) = bᵢ`.
//!
//! ## References
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization, chapter 17.
//! Springer. ISBN 0-387-30303-0.
//!
//! Ernesto G. Birgin and José Mario Martínez (2020). Complexity and performance of an
//! Augmented Lagrangian algorithm. Optimization Methods and Software, 35(5), 885-920.
//! <https://arxiv.org/abs/1907.02401> (the penalty rule and the initial penalty).
//!
//! ## Example
//!
//! Minimize `x² + y²` under `x + y − 1 = 0` with a Nelder-Mead inner solver. The solution is
//! `(0.5, 0.5)` and the multiplier is `-1`.
//!
//! ```
//! use argmin::core::{CostFunction, Error, Executor, IterState, Operator, State};
//! use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
//! use argmin::solver::neldermead::NelderMead;
//!
//! struct Problem {}
//!
//! impl CostFunction for Problem {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!
//!     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!         Ok(p[0].powi(2) + p[1].powi(2))
//!     }
//! }
//!
//! // One value per constraint.
//! impl Operator for Problem {
//!     type Param = Vec<f64>;
//!     type Output = Vec<f64>;
//!
//!     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!         Ok(vec![p[0] + p[1] - 1.0])
//!     }
//! }
//!
//! # fn main() -> Result<(), Error> {
//! // Builds a Nelder-Mead solver around the current point, and its initial state.
//! let factory = |x: &Vec<f64>| {
//!     let simplex = vec![x.clone(), vec![x[0] + 0.1, x[1]], vec![x[0], x[1] + 0.1]];
//!     let solver = NelderMead::new(simplex).with_sd_tolerance(1e-15)?;
//!     Ok((solver, IterState::new().max_iters(1000)))
//! };
//! let solver = AugmentedLagrangian::new(factory).with_tolerance(1e-6)?;
//!
//! let res = Executor::new(Problem {}, solver)
//!     .configure(|state| state.param(vec![0.0, 0.0]).max_iters(50))
//!     .run()?;
//!
//! let x = res.state().get_best_param().unwrap();
//! let multipliers = res.state().get_multipliers().unwrap();
//! assert!((x[0] - 0.5).abs() < 1e-5 && (x[1] - 0.5).abs() < 1e-5);
//! assert!((multipliers[0] + 1.0).abs() < 1e-4);
//! # Ok(())
//! # }
//! ```

mod state;

pub use state::AugmentedLagrangianState;

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Operator, OptimizationResult, Problem, SendAlias,
    Solver, State, SyncAlias, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{ArgminDot, ArgminL2Norm, ArgminScaledAdd, ArgminZeroLike};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;

/// # Augmented Lagrangian method
///
/// Minimizes a cost `f(x)` under equality constraints `c(x) = 0`. Each outer iteration
///
/// 1) builds an inner solver and its initial state with the factory, from the current
///    parameter vector `x`,
/// 2) minimizes `f(x) + λᵀc(x) + (μ/2)·‖c(x)‖²` with it, from that state,
/// 3) updates the multipliers `λ ← λ + μ·c(x)` at the best parameter vector of the inner run,
/// 4) from the second outer iteration on, multiplies `μ` by a factor `γ` (default `10`,
///    configurable via [`with_penalty_factor`](`AugmentedLagrangian::with_penalty_factor`))
///    when `‖c(x)‖₂` did not fall below a fraction `τ` (default `0.5`, configurable via
///    [`with_decrease_ratio`](`AugmentedLagrangian::with_decrease_ratio`)) of its value at the
///    previous outer iterate.
///
/// The penalty rule is the one of Algencan \[1\]: `μ` never grows in the first outer
/// iteration, and it is never compared with `c(x₀)`, so a start that is nearly feasible, such
/// as the solution of a neighbouring problem, does not raise it. Algencan compares the
/// ∞-norm `‖c(x)‖∞`; this solver compares the 2-norm `‖c(x)‖₂`, which is the same for one
/// constraint.
///
/// The solver stops when `‖c(x)‖` is at most the tolerance (default `sqrt(EPSILON)`,
/// configurable via [`with_tolerance`](`AugmentedLagrangian::with_tolerance`)) and the last
/// inner run converged, that is, it ended with
/// [`SolverConverged`](`TerminationReason::SolverConverged`) or
/// [`TargetCostReached`](`TerminationReason::TargetCostReached`). Set the maximum number of
/// outer iterations with the `max_iters` of the state.
///
/// The initial multipliers are zero. The initial `μ` is the one of Algencan \[1\],
///
/// `μ₀ = 10·max{1, |f(x₀)| / max{1, ½‖c(x₀)‖₂²}}`,
///
/// so that it follows the scale of the cost; set it with
/// [`with_penalty`](`AugmentedLagrangian::with_penalty`). `μ₀` should match the scale of `f`
/// and `c`: a small `μ` needs more outer iterations, a large one makes the augmented cost hard
/// for the inner solver and the multipliers less accurate.
///
/// A sequence of nearby problems, such as the points of a profile, starts each one from the
/// solution and the multipliers of the previous one with
/// [`with_multipliers`](`AugmentedLagrangian::with_multipliers`). Passing the previous `μ` with
/// [`with_penalty`](`AugmentedLagrangian::with_penalty`) is optional.
///
/// ## The inner solver
///
/// The factory is a function `Fn(&P) -> Result<(S, I), Error>`: it takes the current parameter
/// vector and returns the inner solver `S` and its initial state `I`. A new inner solver at
/// each outer iteration lets solvers that are built with their start, such as
/// [`NelderMead`](`crate::solver::neldermead::NelderMead`) with its simplex, restart from
/// the new point. The state sets the limits of the inner run, such as its `max_iters`. The inner
/// solver minimizes an [`AugmentedLagrangianCost`].
///
/// The next point is the best parameter vector of the inner state, or what it borrows as: a
/// [`Particle`](`crate::solver::particleswarm::Particle`) of
/// [`ParticleSwarm`](`crate::solver::particleswarm::ParticleSwarm`) borrows as its position.
/// A particle swarm run converges only when it has a stop rule, such as
/// [`with_radius_stop`](`crate::solver::particleswarm::ParticleSwarm::with_radius_stop`) or
/// [`with_improvement_stop`](`crate::solver::particleswarm::ParticleSwarm::with_improvement_stop`);
/// without one it ends at its `max_iters`, and the solver then stops only at its own
/// `max_iters`.
///
/// An error of the factory or of an inner run stops the solver. The error keeps the original
/// one as its cause and names the outer iteration that failed.
///
/// ## State
///
/// The solver works on an [`AugmentedLagrangianState`]. Beside the parameter vector and its
/// cost `f(x)`, it holds the multipliers, `μ`, `c(x)`, `‖c(x)‖` and the iterations of the inner
/// runs. Each outer iteration also gives observers `constraint_norm`, `penalty`, `inner_iters`
/// and `inner_converged`.
///
/// An evaluation of the augmented cost counts as one cost and one operator evaluation.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] (the cost `f(x)`) and
/// [`Operator`] (the constraint values `c(x)`, one value per constraint).
///
/// ## References
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization, chapter 17.
/// Springer. ISBN 0-387-30303-0.
///
/// \[1\] Ernesto G. Birgin and José Mario Martínez (2020). Complexity and performance of an
/// Augmented Lagrangian algorithm. Optimization Methods and Software, 35(5), 885-920.
/// <https://arxiv.org/abs/1907.02401> (Algorithm 2.1 and its implementation section).
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct AugmentedLagrangian<B, M, F> {
    /// Builds the inner solver and its initial state from the current parameter vector
    factory: B,
    /// Initial Lagrange multipliers
    multipliers: Option<M>,
    /// Initial penalty weight (None = the scaled default of Algencan)
    penalty: Option<F>,
    /// Factor by which the penalty weight grows
    penalty_factor: F,
    /// Fraction by which the constraint norm must fall in an outer iteration
    decrease_ratio: F,
    /// Tolerance on the constraint norm
    tolerance: F,
}

impl<B, M, F> AugmentedLagrangian<B, M, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`AugmentedLagrangian`]
    ///
    /// Takes the factory of the inner solver, a function `Fn(&P) -> Result<(S, I), Error>` that
    /// builds the inner solver and its initial state from the current parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> = AugmentedLagrangian::new(factory);
    /// ```
    pub fn new(factory: B) -> Self {
        AugmentedLagrangian {
            factory,
            multipliers: None,
            penalty: None,
            penalty_factor: float!(10.0),
            decrease_ratio: float!(0.5),
            tolerance: F::epsilon().sqrt(),
        }
    }

    /// Set the tolerance on the constraint norm `‖c(x)‖`
    ///
    /// Must be non-negative and defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> =
    ///     AugmentedLagrangian::new(factory).with_tolerance(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tolerance: F) -> Result<Self, Error> {
        if tolerance < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`AugmentedLagrangian`: tolerance must be >= 0."
            ));
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Set the initial Lagrange multipliers `λ`
    ///
    /// One value per constraint, with the convention `L(x, λ) = f(x) + λᵀc(x)`. Defaults to
    /// zero. Pass the multipliers of a solved neighbouring problem to warm-start a sequence of
    /// problems.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> =
    ///     AugmentedLagrangian::new(factory).with_multipliers(vec![-1.0]);
    /// ```
    #[must_use]
    pub fn with_multipliers(mut self, multipliers: M) -> Self {
        self.multipliers = Some(multipliers);
        self
    }

    /// Set the initial penalty weight `μ`
    ///
    /// Must be larger than 0. Defaults to the initial penalty of Algencan,
    /// `10·max{1, |f(x₀)| / max{1, ½‖c(x₀)‖₂²}}`, computed at the initial parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> =
    ///     AugmentedLagrangian::new(factory).with_penalty(100.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_penalty(mut self, penalty: F) -> Result<Self, Error> {
        if penalty <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`AugmentedLagrangian`: penalty must be > 0."
            ));
        }
        self.penalty = Some(penalty);
        Ok(self)
    }

    /// Set the factor by which the penalty weight `μ` grows
    ///
    /// Must be larger than 1 and defaults to 10.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> =
    ///     AugmentedLagrangian::new(factory).with_penalty_factor(5.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_penalty_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor <= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`AugmentedLagrangian`: penalty factor must be > 1."
            ));
        }
        self.penalty_factor = factor;
        Ok(self)
    }

    /// Set the fraction `τ` of its value at the previous outer iterate that `‖c(x)‖₂` must fall
    /// below in an outer iteration, or `μ` grows
    ///
    /// Must be in `(0, 1)` and defaults to 0.5, as in Algencan. The first outer iteration never
    /// grows `μ`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangian;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let factory = ();
    /// let al: AugmentedLagrangian<_, Vec<f64>, f64> =
    ///     AugmentedLagrangian::new(factory).with_decrease_ratio(0.5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_decrease_ratio(mut self, ratio: F) -> Result<Self, Error> {
        if ratio <= float!(0.0) || ratio >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`AugmentedLagrangian`: decrease ratio must be in (0, 1)."
            ));
        }
        self.decrease_ratio = ratio;
        Ok(self)
    }
}

/// The augmented Lagrangian `f(x) + λᵀc(x) + (μ/2)·‖c(x)‖²` of a problem, which the inner
/// solver of [`AugmentedLagrangian`] minimizes.
///
/// It owns the problem during an inner run and gives it back afterwards.
#[derive(Clone, Debug)]
pub struct AugmentedLagrangianCost<O, M, F> {
    /// The problem, which gives `f(x)` and `c(x)`
    problem: O,
    /// Lagrange multipliers `λ`
    multipliers: M,
    /// Penalty weight `μ`
    penalty: F,
}

impl<O, P, M, F> CostFunction for AugmentedLagrangianCost<O, M, F>
where
    O: CostFunction<Param = P, Output = F> + Operator<Param = P, Output = M>,
    M: ArgminDot<M, F>,
    F: ArgminFloat,
{
    type Param = P;
    type Output = F;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let cost = self.problem.cost(param)?;
        let constraints = self.problem.apply(param)?;
        Ok(cost
            + self.multipliers.dot(&constraints)
            + float!(0.5) * self.penalty * constraints.dot(&constraints))
    }

    /// The costs of `params`, in parallel when [`parallelize`](`CostFunction::parallelize`) is
    /// `true` and the `rayon` feature is enabled. When several points fail, the error is that of
    /// the first one in the order of `params`, whatever the order in which the threads score them.
    fn bulk_cost<Q>(&self, params: &[Q]) -> Result<Vec<Self::Output>, Error>
    where
        Q: Borrow<Self::Param> + SyncAlias,
        Self::Output: SendAlias,
        Self: SyncAlias,
    {
        #[cfg(feature = "rayon")]
        {
            if self.parallelize() {
                let costs: Vec<Result<F, Error>> = params
                    .par_iter()
                    .map(|param| self.cost(param.borrow()))
                    .collect();
                return costs.into_iter().collect();
            }
        }
        params
            .iter()
            .map(|param| self.cost(param.borrow()))
            .collect()
    }

    /// Whether the problem scores its points in parallel, as its [`CostFunction`] says.
    fn parallelize(&self) -> bool {
        CostFunction::parallelize(&self.problem)
    }
}

impl<O, B, S, I, P, M, F> Solver<O, AugmentedLagrangianState<P, M, F>>
    for AugmentedLagrangian<B, M, F>
where
    O: CostFunction<Param = P, Output = F> + Operator<Param = P, Output = M>,
    B: Fn(&P) -> Result<(S, I), Error>,
    S: Solver<AugmentedLagrangianCost<O, M, F>, I>,
    I: State<Float = F>,
    I::Param: Borrow<P>,
    P: Clone,
    M: Clone + ArgminDot<M, F> + ArgminL2Norm<F> + ArgminScaledAdd<M, F, M> + ArgminZeroLike,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "Augmented Lagrangian"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: AugmentedLagrangianState<P, M, F>,
    ) -> Result<(AugmentedLagrangianState<P, M, F>, Option<KV>), Error> {
        let param = state
            .get_param()
            .ok_or_else(argmin_error_closure!(
                NotInitialized,
                concat!(
                    "`AugmentedLagrangian` requires an initial parameter vector. ",
                    "Please provide an initial guess via `Executor`s `configure` method."
                )
            ))?
            .clone();
        let cost = problem.cost(&param)?;
        let constraints = problem.apply(&param)?;
        let multipliers = self
            .multipliers
            .clone()
            .unwrap_or_else(|| constraints.zero_like());
        let penalty = self.penalty.unwrap_or_else(|| {
            // Algencan: μ₀ = 10·max{1, |f(x₀)| / max{1, ½‖c(x₀)‖₂²}}
            let half_squared_norm = float!(0.5) * constraints.dot(&constraints);
            float!(10.0)
                * F::max(
                    float!(1.0),
                    cost.abs() / F::max(float!(1.0), half_squared_norm),
                )
        });
        let mut state = state.param(param).cost(cost);
        state.constraint_norm = constraints.l2_norm();
        state.constraints = Some(constraints);
        state.multipliers = Some(multipliers);
        state.penalty = penalty;
        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: AugmentedLagrangianState<P, M, F>,
    ) -> Result<(AugmentedLagrangianState<P, M, F>, Option<KV>), Error> {
        let param = state
            .get_param()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                NotInitialized,
                "`AugmentedLagrangian`: no parameter vector in the state."
            ))?;
        let multipliers = state.take_multipliers().ok_or_else(argmin_error_closure!(
            NotInitialized,
            "`AugmentedLagrangian`: no multipliers in the state."
        ))?;
        let penalty = state.penalty;
        let prev_constraint_norm = state.constraint_norm;

        let outer_iter = state.get_iter();
        let (inner_solver, inner_state) = (self.factory)(&param).map_err(|err| {
            err.context(format!(
                "`AugmentedLagrangian`: the factory failed to build the inner solver in outer iteration {outer_iter}"
            ))
        })?;
        let augmented = AugmentedLagrangianCost {
            problem: problem.take_problem().ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`AugmentedLagrangian`: Failed to take `problem` for the inner solver"
            ))?,
            multipliers,
            penalty,
        };
        let OptimizationResult {
            problem: mut inner_problem,
            state: inner_state,
            ..
        } = Executor::new(augmented, inner_solver)
            .configure(|_| inner_state)
            .ctrlc(false)
            .run()
            .map_err(|err| {
                err.context(format!(
                    "`AugmentedLagrangian`: the inner solver failed in outer iteration {outer_iter}"
                ))
            })?;

        // Give the problem back and count each augmented cost as a cost and an operator call.
        let AugmentedLagrangianCost {
            problem: user_problem,
            multipliers,
            ..
        } = inner_problem
            .take_problem()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`AugmentedLagrangian`: No `problem` returned by the inner solver"
            ))?;
        problem.problem = Some(user_problem);
        let inner_costs = inner_problem.counts.get("cost_count").copied().unwrap_or(0);
        *problem.counts.entry("operator_count").or_insert(0) += inner_costs;
        problem.consume_func_counts(inner_problem);

        let inner_iters = inner_state.get_iter();
        let inner_converged = matches!(
            inner_state.get_termination_reason(),
            Some(TerminationReason::SolverConverged | TerminationReason::TargetCostReached)
        );
        let param = inner_state
            .get_best_param()
            .map(|best| best.borrow().clone())
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`AugmentedLagrangian`: No `param` returned by the inner solver"
            ))?;

        let cost = problem.cost(&param)?;
        let constraints = problem.apply(&param)?;
        let constraint_norm = constraints.l2_norm();
        let multipliers = multipliers.scaled_add(&penalty, &constraints);
        // Algencan: μ never grows in the first outer iteration, and from then on only when
        // ‖c‖ did not fall below a fraction of its value at the previous outer iterate.
        let penalty =
            if outer_iter > 0 && constraint_norm > self.decrease_ratio * prev_constraint_norm {
                penalty * self.penalty_factor
            } else {
                penalty
            };

        let mut state = state.param(param).cost(cost);
        state.constraints = Some(constraints);
        state.constraint_norm = constraint_norm;
        state.multipliers = Some(multipliers);
        state.penalty = penalty;
        state.inner_iters = inner_iters;
        state.inner_iters_total += inner_iters;
        state.inner_converged = inner_converged;

        Ok((
            state,
            Some(kv!(
                "constraint_norm" => constraint_norm;
                "penalty" => penalty;
                "inner_iters" => inner_iters;
                "inner_converged" => inner_converged;
            )),
        ))
    }

    fn terminate(&mut self, state: &AugmentedLagrangianState<P, M, F>) -> TerminationStatus {
        if state.inner_converged && state.constraint_norm <= self.tolerance {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        ArgminError, CostFunction, Error, Executor, IterState, KvValue, Operator, State,
    };
    use crate::solver::neldermead::NelderMead;
    use approx::assert_relative_eq;

    /// `x² + y²` under `x + y − b = 0`.
    struct Circle {
        b: f64,
    }

    impl CostFunction for Circle {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(p[0].powi(2) + p[1].powi(2))
        }
    }

    impl Operator for Circle {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(vec![p[0] + p[1] - self.b])
        }
    }

    /// Hock-Schittkowski problem 6: `(1 − x₁)²` under `10·(x₂ − x₁²) = 0`.
    struct HockSchittkowski6 {}

    impl CostFunction for HockSchittkowski6 {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok((1.0 - p[0]).powi(2))
        }
    }

    impl Operator for HockSchittkowski6 {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(vec![10.0 * (p[1] - p[0].powi(2))])
        }
    }

    type NelderMeadRun = (
        NelderMead<Vec<f64>, f64>,
        IterState<Vec<f64>, (), (), (), (), f64>,
    );

    /// Builds a Nelder-Mead run with a simplex of size 0.1 around `x`.
    ///
    /// A factory takes `&P`, here `&Vec<f64>`.
    #[allow(clippy::ptr_arg)]
    fn nelder_mead(x: &Vec<f64>) -> Result<NelderMeadRun, Error> {
        let mut simplex = vec![x.clone()];
        for i in 0..x.len() {
            let mut vertex = x.clone();
            vertex[i] += 0.1;
            simplex.push(vertex);
        }
        Ok((
            NelderMead::new(simplex).with_sd_tolerance(1e-15)?,
            IterState::new().max_iters(2000),
        ))
    }

    #[test]
    fn test_circle_meets_the_constraint_with_a_finite_penalty() {
        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-6)
            .unwrap();

        let res = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(50))
            .run()
            .unwrap();

        let state = res.state();
        let x = state.get_best_param().unwrap();
        assert_relative_eq!(x[0], 0.5, epsilon = 1e-5);
        assert_relative_eq!(x[1], 0.5, epsilon = 1e-5);
        assert!(state.get_constraint_norm() <= 1e-6);
        assert!(state.get_penalty().is_finite());
        assert!(state.get_penalty() <= 1e4);
        assert_eq!(
            state.get_termination_reason(),
            Some(&crate::core::TerminationReason::SolverConverged)
        );
    }

    #[test]
    fn test_multiplier_is_minus_the_derivative_of_the_optimal_cost() {
        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-6)
            .unwrap();

        let res = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(50))
            .run()
            .unwrap();

        let multipliers = res.state().get_multipliers().unwrap();
        assert_eq!(multipliers.len(), 1);
        assert_relative_eq!(multipliers[0], -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_feasible_start_with_the_right_multiplier_keeps_the_penalty() {
        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-8)
            .unwrap()
            .with_multipliers(vec![-1.0]);

        let res = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.5, 0.5]).max_iters(50))
            .run()
            .unwrap();

        let state = res.state();
        assert_eq!(
            state.get_termination_reason(),
            Some(&crate::core::TerminationReason::SolverConverged)
        );
        // μ₀ = 10·max{1, 0.5 / max{1, 0}} = 10 at (0.5, 0.5).
        assert_eq!(state.get_penalty().to_ne_bytes(), 10.0f64.to_ne_bytes());
        assert_relative_eq!(state.get_multipliers().unwrap()[0], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_warm_step_to_a_neighbour_converges_to_its_multiplier() {
        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-8)
            .unwrap();
        let neighbour = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(50))
            .run()
            .unwrap();
        let neighbour = neighbour.state();

        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-8)
            .unwrap()
            .with_multipliers(neighbour.get_multipliers().unwrap().clone());
        let res = Executor::new(Circle { b: 1.05 }, solver)
            .configure(|state| {
                state
                    .param(neighbour.get_best_param().unwrap().clone())
                    .max_iters(50)
            })
            .run()
            .unwrap();

        assert_eq!(
            res.state().get_termination_reason(),
            Some(&crate::core::TerminationReason::SolverConverged)
        );
        assert_relative_eq!(
            res.state().get_multipliers().unwrap()[0],
            -1.05,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_profile_with_secant_starts_keeps_the_penalty_bounded() {
        // 17 levels: 0, then 8 steps of 0.1·1.3ᵏ on each side.
        let solve = |b: f64, start: Vec<f64>, warm: Option<(Vec<f64>, f64)>| {
            let mut solver = AugmentedLagrangian::new(nelder_mead)
                .with_tolerance(1e-8)
                .unwrap();
            if let Some((multipliers, penalty)) = warm {
                solver = solver
                    .with_multipliers(multipliers)
                    .with_penalty(penalty)
                    .unwrap();
            }
            let res = Executor::new(Circle { b }, solver)
                .configure(|state| state.param(start).max_iters(100))
                .run()
                .unwrap();
            let state = res.state().clone();
            assert_eq!(
                state.get_termination_reason(),
                Some(&crate::core::TerminationReason::SolverConverged)
            );
            assert!(state.get_penalty() <= 1e3, "μ = {}", state.get_penalty());
            let multiplier = state.get_multipliers().unwrap()[0];
            assert!(
                (multiplier + b).abs() <= 1e-6,
                "λ = {multiplier} at c = {b}"
            );
            state
        };

        let first = solve(0.0, vec![0.0, 0.0], None);
        let mut points = 1;
        for sign in [1.0, -1.0] {
            let mut prev = first.clone();
            let mut prev_prev: Option<(Vec<f64>, f64)> = None;
            let (mut b_prev, mut b, mut step) = (0.0, 0.0, 0.1);
            for _ in 0..8 {
                b += sign * step;
                step *= 1.3;
                let x_prev = prev.get_best_param().unwrap().clone();
                let start = match &prev_prev {
                    Some((x_pp, b_pp)) => {
                        let t = (b - b_prev) / (b_prev - b_pp);
                        x_prev
                            .iter()
                            .zip(x_pp)
                            .map(|(a, c)| a + t * (a - c))
                            .collect()
                    }
                    None => x_prev.clone(),
                };
                let warm = Some((prev.get_multipliers().unwrap().clone(), prev.get_penalty()));
                let next = solve(b, start, warm);
                prev_prev = Some((x_prev, b_prev));
                b_prev = b;
                prev = next;
                points += 1;
            }
        }
        assert_eq!(points, 17);
    }

    #[test]
    fn test_initial_penalty_follows_the_algencan_formula() {
        // μ₀ = 10·max{1, |f(x₀)| / max{1, ½‖c(x₀)‖²}}
        for (start, expected) in [
            // |f| small: f = 0.02, ½c² = 0.32, so μ₀ = 10.
            (vec![0.1, 0.1], 10.0),
            // |f| large, c = 0: f = 13, so μ₀ = 130.
            (vec![3.0, -2.0], 130.0),
            // |f| large, ½c² large: f = 200, ½c² = 180.5, so μ₀ = 2000 / 180.5.
            (vec![10.0, 10.0], 2000.0 / 180.5),
        ] {
            let mut al = new_solver();
            let (state, _) = al
                .init(
                    &mut Problem::new(Circle { b: 1.0 }),
                    AugmentedLagrangianState::new().param(start),
                )
                .unwrap();
            assert_relative_eq!(state.get_penalty(), expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hock_schittkowski_6() {
        let solver = AugmentedLagrangian::new(nelder_mead)
            .with_tolerance(1e-6)
            .unwrap();

        let res = Executor::new(HockSchittkowski6 {}, solver)
            .configure(|state| state.param(vec![-1.2, 1.0]).max_iters(50))
            .run()
            .unwrap();

        let state = res.state();
        let x = state.get_best_param().unwrap();
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-4);
        assert!(state.get_constraint_norm() <= 1e-6);
        assert_eq!(
            state.get_termination_reason(),
            Some(&crate::core::TerminationReason::SolverConverged)
        );
    }
    /// A solver that moves to `(0.4, 0.4)` when it starts at the origin and fails elsewhere.
    #[derive(Clone)]
    struct FailsAwayFromTheOrigin {
        start: Vec<f64>,
    }

    impl<O> Solver<O, IterState<Vec<f64>, (), (), (), (), f64>> for FailsAwayFromTheOrigin
    where
        O: CostFunction<Param = Vec<f64>, Output = f64>,
    {
        fn name(&self) -> &str {
            "Fails away from the origin"
        }

        fn next_iter(
            &mut self,
            problem: &mut Problem<O>,
            state: IterState<Vec<f64>, (), (), (), (), f64>,
        ) -> Result<(IterState<Vec<f64>, (), (), (), (), f64>, Option<KV>), Error> {
            if self.start != vec![0.0, 0.0] {
                return Err(Error::msg("inner failure"));
            }
            let param = vec![0.4, 0.4];
            let cost = problem.cost(&param)?;
            Ok((state.param(param).cost(cost), None))
        }

        fn terminate(
            &mut self,
            _state: &IterState<Vec<f64>, (), (), (), (), f64>,
        ) -> TerminationStatus {
            TerminationStatus::NotTerminated
        }
    }

    #[test]
    fn test_inner_error_stops_the_solver_with_the_outer_iteration() {
        let factory = |x: &Vec<f64>| {
            Ok((
                FailsAwayFromTheOrigin { start: x.clone() },
                IterState::new().max_iters(1),
            ))
        };
        let solver = AugmentedLagrangian::new(factory);

        let err = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(50))
            .run()
            .err()
            .unwrap();

        assert_eq!(
            err.to_string(),
            "`AugmentedLagrangian`: the inner solver failed in outer iteration 1"
        );
        assert_eq!(err.root_cause().to_string(), "inner failure");
    }

    #[test]
    fn test_factory_error_stops_the_solver_with_the_outer_iteration() {
        let factory =
            |_: &Vec<f64>| -> Result<NelderMeadRun, Error> { Err(Error::msg("no inner solver")) };
        let solver = AugmentedLagrangian::new(factory);

        let err = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]))
            .run()
            .err()
            .unwrap();

        assert_eq!(
            err.to_string(),
            "`AugmentedLagrangian`: the factory failed to build the inner solver in outer iteration 0"
        );
        assert_eq!(err.root_cause().to_string(), "no inner solver");
    }

    #[test]
    fn test_state_reports_the_outer_and_inner_iterations_the_norm_and_the_penalty() {
        let mut solver = AugmentedLagrangian::new(nelder_mead);
        let mut problem = Problem::new(Circle { b: 1.0 });
        let state = AugmentedLagrangianState::new().param(vec![0.0, 0.0]);

        let (state, _) = solver.init(&mut problem, state).unwrap();
        assert_relative_eq!(state.get_constraint_norm(), 1.0);
        assert_relative_eq!(state.get_penalty(), 10.0);
        assert_eq!(state.get_multipliers(), Some(&vec![0.0]));

        let (state, kv) = solver.next_iter(&mut problem, state).unwrap();
        let kv = kv.unwrap();
        assert!(state.get_inner_iters() > 0);
        assert_eq!(state.get_inner_iters_total(), state.get_inner_iters());
        assert!(state.get_inner_converged());
        assert_eq!(
            kv.get("inner_iters"),
            Some(&KvValue::Uint(state.get_inner_iters()))
        );
        assert_eq!(
            kv.get("constraint_norm"),
            Some(&KvValue::Float(state.get_constraint_norm()))
        );
        assert_eq!(
            kv.get("penalty"),
            Some(&KvValue::Float(state.get_penalty()))
        );
        // The inner minimizer of x² + y² + (10/2)·(x + y − 1)² is x = y = 10/22.
        assert_relative_eq!(state.get_constraint_norm(), 1.0 / 11.0, epsilon = 1e-6);
        assert_relative_eq!(
            state.get_multipliers().unwrap()[0],
            -10.0 / 11.0,
            epsilon = 1e-5
        );
        let first_inner_iters = state.get_inner_iters();

        let mut state = state;
        state.increment_iter();
        let (state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(
            state.get_inner_iters_total(),
            first_inner_iters + state.get_inner_iters()
        );
        assert_eq!(state.get_iter(), 1);
        assert_eq!(
            problem.counts.get("operator_count"),
            problem.counts.get("cost_count")
        );
    }
    type NelderMeadFactory = fn(&Vec<f64>) -> Result<NelderMeadRun, Error>;

    test_trait_impl!(
        augmented_lagrangian,
        AugmentedLagrangian<NelderMeadFactory, Vec<f64>, f64>
    );

    fn new_solver() -> AugmentedLagrangian<NelderMeadFactory, Vec<f64>, f64> {
        AugmentedLagrangian::new(nelder_mead)
    }

    #[test]
    fn test_new() {
        let al = new_solver();
        assert!(al.multipliers.is_none());
        assert!(al.penalty.is_none());
        assert_eq!(al.penalty_factor.to_ne_bytes(), 10.0f64.to_ne_bytes());
        assert_eq!(al.decrease_ratio.to_ne_bytes(), 0.5f64.to_ne_bytes());
        assert_eq!(
            al.tolerance.to_ne_bytes(),
            f64::EPSILON.sqrt().to_ne_bytes()
        );
    }

    #[test]
    fn test_with_tolerance() {
        for tol in [0.0, 1e-8, 1.0] {
            let al = new_solver().with_tolerance(tol).unwrap();
            assert_eq!(al.tolerance.to_ne_bytes(), tol.to_ne_bytes());
        }
        for tol in [-f64::EPSILON, -1.0] {
            assert_error!(
                new_solver().with_tolerance(tol),
                ArgminError,
                "Invalid parameter: \"`AugmentedLagrangian`: tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_with_multipliers() {
        let al = new_solver().with_multipliers(vec![-1.0, 2.0]);
        assert_eq!(al.multipliers, Some(vec![-1.0, 2.0]));
    }

    #[test]
    fn test_with_penalty() {
        for penalty in [f64::EPSILON, 1.0, 1e6] {
            let al = new_solver().with_penalty(penalty).unwrap();
            assert_eq!(al.penalty, Some(penalty));
        }
        for penalty in [0.0, -1.0] {
            assert_error!(
                new_solver().with_penalty(penalty),
                ArgminError,
                "Invalid parameter: \"`AugmentedLagrangian`: penalty must be > 0.\""
            );
        }
    }

    #[test]
    fn test_with_penalty_factor() {
        for factor in [1.0 + f64::EPSILON, 2.0, 100.0] {
            let al = new_solver().with_penalty_factor(factor).unwrap();
            assert_eq!(al.penalty_factor.to_ne_bytes(), factor.to_ne_bytes());
        }
        for factor in [1.0, 0.5, -1.0] {
            assert_error!(
                new_solver().with_penalty_factor(factor),
                ArgminError,
                "Invalid parameter: \"`AugmentedLagrangian`: penalty factor must be > 1.\""
            );
        }
    }

    #[test]
    fn test_with_decrease_ratio() {
        for ratio in [f64::EPSILON, 0.5, 1.0 - f64::EPSILON] {
            let al = new_solver().with_decrease_ratio(ratio).unwrap();
            assert_eq!(al.decrease_ratio.to_ne_bytes(), ratio.to_ne_bytes());
        }
        for ratio in [0.0, 1.0, -1.0, 2.0] {
            assert_error!(
                new_solver().with_decrease_ratio(ratio),
                ArgminError,
                "Invalid parameter: \"`AugmentedLagrangian`: decrease ratio must be in (0, 1).\""
            );
        }
    }

    #[test]
    fn test_init_param_not_initialized() {
        let mut al = new_solver();
        let res = al.init(
            &mut Problem::new(Circle { b: 1.0 }),
            AugmentedLagrangianState::new(),
        );
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`AugmentedLagrangian` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_init_takes_the_initial_multipliers_and_penalty() {
        let mut al = new_solver()
            .with_multipliers(vec![-1.0])
            .with_penalty(100.0)
            .unwrap();
        let (state, kv) = al
            .init(
                &mut Problem::new(Circle { b: 1.0 }),
                AugmentedLagrangianState::new().param(vec![0.5, 0.0]),
            )
            .unwrap();
        assert!(kv.is_none());
        assert_eq!(state.get_multipliers(), Some(&vec![-1.0]));
        assert_eq!(state.get_constraints(), Some(&vec![-0.5]));
        assert_relative_eq!(state.get_constraint_norm(), 0.5);
        assert_relative_eq!(state.get_penalty(), 100.0);
        assert_relative_eq!(state.get_cost(), 0.25);
    }

    #[test]
    fn test_augmented_cost() {
        let augmented = AugmentedLagrangianCost {
            problem: Circle { b: 1.0 },
            multipliers: vec![-1.0],
            penalty: 4.0,
        };
        // f = 1 + 4 = 5, c = 2, λc = -2, (μ/2)c² = 8.
        assert_relative_eq!(augmented.cost(&vec![1.0, 2.0]).unwrap(), 11.0);
    }

    /// `x²` under `x = 0`, which fails at each point `x ≥ 40` with a message that names the
    /// point. The point 40 fails 50 ms after the call, so the other failing points fail first.
    struct FailsFromForty {
        parallel: bool,
    }

    impl CostFunction for FailsFromForty {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            if (p[0] - 40.0).abs() < 0.5 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if p[0] >= 40.0 {
                return Err(Error::msg(format!("fails at {}", p[0])));
            }
            Ok(p[0].powi(2))
        }

        fn parallelize(&self) -> bool {
            self.parallel
        }
    }

    impl Operator for FailsFromForty {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(vec![p[0]])
        }
    }

    #[test]
    fn test_augmented_cost_parallelizes_as_its_problem() {
        for parallel in [true, false] {
            let augmented = AugmentedLagrangianCost {
                problem: FailsFromForty { parallel },
                multipliers: vec![0.0],
                penalty: 1.0,
            };
            assert_eq!(augmented.parallelize(), parallel);
        }
    }

    #[test]
    fn test_augmented_bulk_cost_returns_the_error_of_the_first_failing_point() {
        let points: Vec<Vec<f64>> = (0..64).map(|i| vec![f64::from(i)]).collect();
        for parallel in [true, false] {
            let augmented = AugmentedLagrangianCost {
                problem: FailsFromForty { parallel },
                multipliers: vec![0.0],
                penalty: 1.0,
            };
            let error = augmented.bulk_cost(&points).unwrap_err();
            assert_eq!(error.to_string(), "fails at 40");
            // f = x², c = x, (μ/2)c² = x²/2.
            let costs = augmented.bulk_cost(&points[..40]).unwrap();
            assert_relative_eq!(costs[3], 13.5);
        }
    }
    #[cfg(feature = "rand")]
    #[test]
    fn test_particle_swarm_as_inner_solver() {
        use crate::core::PopulationState;
        use crate::solver::particleswarm::{Particle, ParticleSwarm};
        use rand::SeedableRng;

        let factory = |_: &Vec<f64>| {
            let solver = ParticleSwarm::new((vec![-2.0, -2.0], vec![2.0, 2.0]), 40)
                .with_rng_generator(rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42))
                .with_radius_stop(1e-8)?;
            let state: PopulationState<Particle<Vec<f64>, f64>, f64> =
                PopulationState::new().max_iters(1000);
            Ok((solver, state))
        };
        let solver = AugmentedLagrangian::new(factory)
            .with_tolerance(1e-4)
            .unwrap();

        let res = Executor::new(Circle { b: 1.0 }, solver)
            .configure(|state| state.param(vec![0.0, 0.0]).max_iters(8))
            .run()
            .unwrap();

        let state = res.state();
        let x = state.get_best_param().unwrap();
        assert_relative_eq!(x[0], 0.5, epsilon = 1e-3);
        assert_relative_eq!(x[1], 0.5, epsilon = 1e-3);
        assert_relative_eq!(state.get_multipliers().unwrap()[0], -1.0, epsilon = 1e-2);
        // The radius stop rule ends each swarm run with `SolverConverged`, so the solver stops by
        // its tolerance.
        assert!(state.get_inner_converged());
        assert!(state.get_constraint_norm() <= 1e-4);
        assert!(state.get_iter() < 8);
        assert_eq!(
            state.get_termination_reason(),
            Some(&crate::core::TerminationReason::SolverConverged)
        );
    }
}
