// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Safeguarded Newton root finding
//!
//! [`NewtonRoot`] finds a root of a function of one variable from the function and its
//! derivative, inside a bracket where the function changes sign.
//!
//! ## The method
//!
//! The solver starts at the middle of the bracket. At each step it evaluates `f(x)` and narrows
//! the bracket to the side where the sign changes. Then it takes the Newton step `x − f(x)/f'(x)`
//! when that point falls strictly inside the new bracket, and else it bisects the bracket. A
//! derivative of 0, or one that is not finite, gives a bisection too. The bracket always holds
//! the root, so the solver never leaves it and always converges; near a simple root it takes
//! Newton steps and converges quadratically.
//!
//! The solver stops when `f(x)` is exactly 0, when a step is at most `tol · |x|` (the tolerance
//! is relative and defaults to `4 ε`), when the next point is an end of the bracket (the bracket
//! holds no other floating-point number), or after the maximum number of iterations.
//!
//! The solver evaluates `f` at the ends of the bracket to check the change of sign. A caller
//! that already evaluated them gives the values with
//! [`NewtonRoot::with_bracket_values`] and saves these two evaluations.
//!
//! ## When to prefer it to `BrentRoot`
//!
//! Use [`NewtonRoot`] when the derivative is available at a small cost, for example the density
//! when you invert a distribution function. Near the root, each step gains about twice the
//! correct digits of the step before, so it needs fewer evaluations than
//! [`BrentRoot`](`crate::solver::brent::BrentRoot`). Use `BrentRoot` when there is no derivative,
//! or when the derivative costs as much as the function or more.
//!
//! ## Reference
//!
//! William H. Press, Saul A. Teukolsky, William T. Vetterling and Brian P. Flannery (2007).
//! _Numerical Recipes: The Art of Scientific Computing_, 3rd edition, section 9.4 (Newton-Raphson
//! method using derivative, safe Newton with bisection). Cambridge University Press.

use crate::core::{
    ArgminFloat, CostFunction, Error, Gradient, IterState, Problem, Solver, State,
    TerminationReason, KV,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Safeguarded Newton root finding
///
/// Finds a root of `f` in a bracket `[min, max]` where `f(min)` and `f(max)` have different
/// signs, with Newton steps kept inside a bracket that bisection narrows. See the
/// [module documentation](`crate::solver::newtonroot`) for the method.
///
/// The solver starts at the middle of the bracket and ignores an initial parameter given to the
/// [`Executor`](`crate::core::Executor`). Set the maximum number of iterations with the
/// `Executor`s `configure` method.
///
/// The result is the last point, `state.param`, and the cost of a point is `|f(x)|`. When the
/// solver stops on the step tolerance, it does not evaluate the last point: its cost is infinite
/// (not computed), and `state.best_param` is the point before, at most `tol · |x|` away.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] (the function `f`) and
/// [`Gradient`] (its derivative `f'`), both of a scalar `F`.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NewtonRoot<F> {
    /// Relative tolerance on the step
    tol: F,
    /// Lower end of the bracket
    low: F,
    /// Upper end of the bracket
    high: F,
    /// `f(low)`
    f_low: F,
    /// `f(high)`
    f_high: F,
    /// Current point
    x: F,
    /// `f(x)`
    fx: F,
    /// Whether the caller gave `f_low` and `f_high`, so that `init` does not evaluate them
    ends_known: bool,
}

impl<F: ArgminFloat> NewtonRoot<F> {
    /// Construct a new instance of [`NewtonRoot`] on the bracket `[min, max]`.
    ///
    /// `min` and `max` must be finite and `min` smaller than `max`. The solver checks at
    /// initialization that `f(min)` and `f(max)` have different signs.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newtonroot::NewtonRoot;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let solver = NewtonRoot::new(0.0f64, 2.0f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(min: F, max: F) -> Result<Self, Error> {
        if !(min.is_finite() && max.is_finite() && min < max) {
            return Err(argmin_error!(
                InvalidParameter,
                "`NewtonRoot`: `min` and `max` must be finite and `min` smaller than `max`."
            ));
        }
        Ok(NewtonRoot {
            tol: float!(4.0) * F::epsilon(),
            low: min,
            high: max,
            f_low: F::nan(),
            f_high: F::nan(),
            x: F::nan(),
            fx: F::nan(),
            ends_known: false,
        })
    }

    /// Set the relative tolerance on the step: the solver stops when a step is at most
    /// `tol · |x|`.
    ///
    /// Must be non-negative and defaults to `4 ε`, four times the machine epsilon of `F`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newtonroot::NewtonRoot;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let solver = NewtonRoot::new(0.0f64, 2.0f64)?.with_tolerance(1e-10)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol.is_nan() || tol < F::zero() {
            return Err(argmin_error!(
                InvalidParameter,
                "`NewtonRoot`: Tolerance must be non-negative."
            ));
        }
        self.tol = tol;
        Ok(self)
    }

    /// Give the values of `f` at the ends of the bracket, `f_min = f(min)` and
    /// `f_max = f(max)`, so that the solver does not evaluate them.
    ///
    /// Use it when the caller already evaluated `f` at the ends, for example while it searched
    /// for the bracket: the solver then makes two fewer evaluations of `f` and gives the same
    /// result. Both values must be finite. At initialization, the solver checks them as it checks
    /// the values it evaluates: a value of 0 makes that end the root, and two values of the same
    /// sign are an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newtonroot::NewtonRoot;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// // f(x) = x³ − 2: f(0) = −2 and f(2) = 6.
    /// let solver = NewtonRoot::new(0.0f64, 2.0f64)?.with_bracket_values(-2.0, 6.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_bracket_values(mut self, f_min: F, f_max: F) -> Result<Self, Error> {
        if !(f_min.is_finite() && f_max.is_finite()) {
            return Err(argmin_error!(
                InvalidParameter,
                "`NewtonRoot`: f(min) and f(max) must be finite."
            ));
        }
        self.f_low = f_min;
        self.f_high = f_max;
        self.ends_known = true;
        Ok(self)
    }
}

impl<O, F> Solver<O, IterState<F, (), (), (), (), F>> for NewtonRoot<F>
where
    O: CostFunction<Param = F, Output = F> + Gradient<Param = F, Gradient = F>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "NewtonRoot"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        // NewtonRoot starts at the middle of the bracket
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        if !self.ends_known {
            self.f_low = problem.cost(&self.low)?;
        }
        if self.f_low == F::zero() {
            return Ok((root(state, self.low), None));
        }
        if !self.ends_known {
            self.f_high = problem.cost(&self.high)?;
        }
        if self.f_high == F::zero() {
            return Ok((root(state, self.high), None));
        }
        let zero = F::zero();
        let change_of_sign =
            (self.f_low < zero && self.f_high > zero) || (self.f_low > zero && self.f_high < zero);
        if !change_of_sign {
            return Err(argmin_error!(
                InvalidParameter,
                "`NewtonRoot`: f(min) and f(max) must have different signs."
            ));
        }
        self.x = float!(0.5) * (self.low + self.high);
        self.fx = problem.cost(&self.x)?;
        if self.fx == F::zero() {
            return Ok((root(state, self.x), None));
        }
        Ok((state.param(self.x).cost(self.fx.abs()), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        let (x, fx) = (self.x, self.fx);
        // The root lies between x and the end where f has the other sign.
        if (fx > F::zero()) == (self.f_high > F::zero()) {
            self.high = x;
            self.f_high = fx;
        } else {
            self.low = x;
            self.f_low = fx;
        }
        let newton = x - fx / problem.gradient(&x)?;
        // A derivative of 0 or not finite gives a step that is not finite: the comparisons fail.
        let next = if newton > self.low && newton < self.high {
            newton
        } else {
            float!(0.5) * (self.low + self.high)
        };
        self.x = next;
        if next == self.low || next == self.high {
            // The bracket holds no other floating-point number, and f is known at its ends.
            self.fx = if next == self.low {
                self.f_low
            } else {
                self.f_high
            };
            let state = state.param(next).cost(self.fx.abs());
            return Ok((
                state.terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }
        if (next - x).abs() <= self.tol * x.abs() {
            // The last point is not evaluated: its cost is not computed, which `IterState`
            // writes as infinity.
            self.fx = F::nan();
            let state = state.param(next).cost(F::infinity());
            return Ok((
                state.terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }
        self.fx = problem.cost(&next)?;
        let state = state.param(next).cost(self.fx.abs());
        if self.fx == F::zero() {
            return Ok((
                state.terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }
        Ok((state, None))
    }
}

/// The state at a root found where `f` is exactly 0.
fn root<F: ArgminFloat>(
    state: IterState<F, (), (), (), (), F>,
    x: F,
) -> IterState<F, (), (), (), (), F> {
    state
        .param(x)
        .cost(F::zero())
        .terminate_with(TerminationReason::SolverConverged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        ArgminError, CostFunction, Error, Executor, Gradient, IterState, Problem, Solver, State,
        TerminationReason,
    };
    use crate::solver::brent::BrentRoot;
    use approx::assert_relative_eq;

    /// `x³ − c`, increasing, with its root at the cube root of `c`.
    #[derive(Clone)]
    struct Cubic {
        c: f64,
    }

    impl CostFunction for Cubic {
        type Param = f64;
        type Output = f64;

        fn cost(&self, x: &f64) -> Result<f64, Error> {
            Ok(x * x * x - self.c)
        }
    }

    impl Gradient for Cubic {
        type Param = f64;
        type Gradient = f64;

        fn gradient(&self, x: &f64) -> Result<f64, Error> {
            Ok(3.0 * x * x)
        }
    }

    /// `c − x³`, decreasing, with its root at the cube root of `c`.
    #[derive(Clone)]
    struct FallingCubic {
        c: f64,
    }

    impl CostFunction for FallingCubic {
        type Param = f64;
        type Output = f64;

        fn cost(&self, x: &f64) -> Result<f64, Error> {
            Ok(self.c - x * x * x)
        }
    }

    impl Gradient for FallingCubic {
        type Param = f64;
        type Gradient = f64;

        fn gradient(&self, x: &f64) -> Result<f64, Error> {
            Ok(-3.0 * x * x)
        }
    }

    /// `atan(x − 1)`: far from the root, the Newton step overshoots.
    #[derive(Clone)]
    struct Arctangent {}

    impl CostFunction for Arctangent {
        type Param = f64;
        type Output = f64;

        fn cost(&self, x: &f64) -> Result<f64, Error> {
            Ok((x - 1.0).atan())
        }
    }

    impl Gradient for Arctangent {
        type Param = f64;
        type Gradient = f64;

        fn gradient(&self, x: &f64) -> Result<f64, Error> {
            Ok(1.0 / (1.0 + (x - 1.0).powi(2)))
        }
    }

    /// The logistic distribution function minus a level `p`, as commol
    /// inverts a distribution function.
    #[derive(Clone)]
    struct Logistic {
        p: f64,
    }

    impl CostFunction for Logistic {
        type Param = f64;
        type Output = f64;

        fn cost(&self, x: &f64) -> Result<f64, Error> {
            Ok(1.0 / (1.0 + (-x).exp()) - self.p)
        }
    }

    impl Gradient for Logistic {
        type Param = f64;
        type Gradient = f64;

        fn gradient(&self, x: &f64) -> Result<f64, Error> {
            let s = 1.0 / (1.0 + (-x).exp());
            Ok(s * (1.0 - s))
        }
    }

    fn count(state: &IterState<f64, (), (), (), (), f64>, name: &str) -> u64 {
        state.get_func_counts().get(name).copied().unwrap_or(0)
    }

    test_trait_impl!(newton_root, NewtonRoot<f64>);

    #[test]
    fn test_new() {
        let NewtonRoot {
            tol,
            low,
            high,
            f_low,
            f_high,
            x,
            fx,
            ends_known,
        } = NewtonRoot::new(0.0f64, 2.0f64).unwrap();

        assert_eq!(tol.to_ne_bytes(), (4.0 * f64::EPSILON).to_ne_bytes());
        assert_eq!(low.to_ne_bytes(), 0.0f64.to_ne_bytes());
        assert_eq!(high.to_ne_bytes(), 2.0f64.to_ne_bytes());
        assert!(f_low.is_nan());
        assert!(f_high.is_nan());
        assert!(x.is_nan());
        assert!(fx.is_nan());
        assert!(!ends_known);
    }

    #[test]
    fn test_new_errors() {
        let message = concat!(
            "Invalid parameter: \"`NewtonRoot`: ",
            "`min` and `max` must be finite and `min` smaller than `max`.\""
        );
        assert_error!(NewtonRoot::new(2.0f64, 0.0f64), ArgminError, message);
        assert_error!(NewtonRoot::new(1.0f64, 1.0f64), ArgminError, message);
        assert_error!(NewtonRoot::new(0.0f64, f64::INFINITY), ArgminError, message);
        assert_error!(NewtonRoot::new(f64::NAN, 1.0f64), ArgminError, message);
    }

    #[test]
    fn test_tolerance() {
        let NewtonRoot { tol, .. } = NewtonRoot::new(0.0f64, 2.0f64)
            .unwrap()
            .with_tolerance(1e-8)
            .unwrap();
        assert_eq!(tol.to_ne_bytes(), 1e-8f64.to_ne_bytes());

        let NewtonRoot { tol, .. } = NewtonRoot::new(0.0f64, 2.0f64)
            .unwrap()
            .with_tolerance(0.0)
            .unwrap();
        assert_eq!(tol.to_ne_bytes(), 0.0f64.to_ne_bytes());
    }

    #[test]
    fn test_tolerance_errors() {
        let message = "Invalid parameter: \"`NewtonRoot`: Tolerance must be non-negative.\"";
        let res = NewtonRoot::new(0.0f64, 2.0f64)
            .unwrap()
            .with_tolerance(-1e-8);
        assert_error!(res, ArgminError, message);
        let res = NewtonRoot::new(0.0f64, 2.0f64)
            .unwrap()
            .with_tolerance(f64::NAN);
        assert_error!(res, ArgminError, message);
    }

    /// R1: the cube root of 2 to machine precision, in fewer evaluations
    /// than `BrentRoot`.
    #[test]
    fn test_cube_root_of_two() {
        let root = 2.0f64.cbrt();

        let solver = NewtonRoot::new(0.0f64, 2.0f64).unwrap();
        let res = Executor::new(Cubic { c: 2.0 }, solver)
            .configure(|state| state.counting(true).max_iters(100))
            .run()
            .unwrap();
        let newton = res.state();
        assert_eq!(
            newton.get_termination_reason(),
            Some(&TerminationReason::SolverConverged)
        );
        assert_relative_eq!(
            *newton.get_param().unwrap(),
            root,
            max_relative = f64::EPSILON
        );

        let solver = BrentRoot::new(0.0f64, 2.0f64, 0.0f64);
        let res = Executor::new(Cubic { c: 2.0 }, solver)
            .configure(|state| state.counting(true).max_iters(100))
            .run()
            .unwrap();
        let brent = res.state();
        assert_relative_eq!(
            *brent.get_param().unwrap(),
            root,
            max_relative = f64::EPSILON
        );

        // The evaluations of f: two at the ends of the bracket, one at its
        // middle and one at each of the five Newton points, the last of
        // which gives f(x) = 0. NewtonRoot also evaluates f' once per
        // iteration.
        assert_eq!(count(newton, "cost_count"), 8);
        assert_eq!(count(newton, "gradient_count"), 5);
        assert_eq!(count(brent, "cost_count"), 9);
        assert!(count(newton, "cost_count") < count(brent, "cost_count"));
    }

    /// A stop on the step tolerance does not evaluate the last point: its
    /// cost is infinite, as `IterState` writes a cost not computed.
    #[test]
    fn test_last_point_not_evaluated() {
        let solver = NewtonRoot::new(-40.0f64, 40.0f64).unwrap();
        let res = Executor::new(Logistic { p: 0.3 }, solver)
            .configure(|state| state.counting(true).max_iters(100))
            .run()
            .unwrap();
        let state = res.state();
        let root = (0.3f64 / 0.7).ln();
        assert_eq!(
            state.get_termination_reason(),
            Some(&TerminationReason::SolverConverged)
        );
        assert!(state.get_cost().is_infinite());
        assert_relative_eq!(
            *state.get_param().unwrap(),
            root,
            max_relative = 4.0 * f64::EPSILON
        );
        // The best point is the last point evaluated.
        assert_relative_eq!(
            *state.get_best_param().unwrap(),
            root,
            max_relative = 4.0 * f64::EPSILON
        );
        // Two ends, the middle and every Newton point but the last.
        let iters = state.get_iter();
        assert_eq!(count(state, "cost_count"), 2 + iters);
        assert_eq!(count(state, "gradient_count"), iters);
    }

    /// A decreasing function: the bracket narrows by the sign of f(max).
    #[test]
    fn test_decreasing_function() {
        let solver = NewtonRoot::new(0.0f64, 2.0f64).unwrap();
        let res = Executor::new(FallingCubic { c: 2.0 }, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();
        assert_relative_eq!(
            *res.state().get_param().unwrap(),
            2.0f64.cbrt(),
            max_relative = f64::EPSILON
        );
    }

    /// R1: a Newton step that leaves the bracket gives a bisection, and the
    /// solver still converges.
    #[test]
    fn test_bisects_when_newton_leaves_bracket() {
        let mut solver = NewtonRoot::new(-10.0f64, 20.0f64).unwrap();
        let mut problem = Problem::new(Arctangent {});
        let (state, _) = solver.init(&mut problem, IterState::new()).unwrap();
        assert_eq!(
            state.get_param().unwrap().to_ne_bytes(),
            5.0f64.to_ne_bytes()
        );

        // f(5) > 0 sets the bracket to [-10, 5]; the Newton step from 5 is
        // 5 − atan(4)·17 ≈ −17.5, outside it, so the next point is −2.5.
        let (state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(
            state.get_param().unwrap().to_ne_bytes(),
            (-2.5f64).to_ne_bytes()
        );

        let solver = NewtonRoot::new(-10.0f64, 20.0f64).unwrap();
        let res = Executor::new(Arctangent {}, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();
        assert_eq!(
            res.state().get_termination_reason(),
            Some(&TerminationReason::SolverConverged)
        );
        assert_relative_eq!(
            *res.state().get_param().unwrap(),
            1.0,
            max_relative = 4.0 * f64::EPSILON
        );
    }

    /// R1: a derivative of 0 gives a bisection, and the solver still
    /// converges.
    #[test]
    fn test_bisects_when_derivative_is_zero() {
        let mut solver = NewtonRoot::new(-2.0f64, 2.0f64).unwrap();
        let mut problem = Problem::new(Cubic { c: 2.0 });
        let (state, _) = solver.init(&mut problem, IterState::new()).unwrap();
        assert_eq!(
            state.get_param().unwrap().to_ne_bytes(),
            0.0f64.to_ne_bytes()
        );

        // f'(0) = 0: the bracket becomes [0, 2] and the next point is 1.
        let (state, _) = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(
            state.get_param().unwrap().to_ne_bytes(),
            1.0f64.to_ne_bytes()
        );

        let solver = NewtonRoot::new(-2.0f64, 2.0f64).unwrap();
        let res = Executor::new(Cubic { c: 2.0 }, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();
        assert_eq!(
            res.state().get_termination_reason(),
            Some(&TerminationReason::SolverConverged)
        );
        assert_relative_eq!(
            *res.state().get_param().unwrap(),
            2.0f64.cbrt(),
            max_relative = f64::EPSILON
        );
    }

    /// R1: a bracket with no change of sign is an error that says so.
    #[test]
    fn test_no_change_of_sign() {
        let message = concat!(
            "Invalid parameter: \"`NewtonRoot`: ",
            "f(min) and f(max) must have different signs.\""
        );
        let mut solver = NewtonRoot::new(2.0f64, 3.0f64).unwrap();
        let res = solver.init(&mut Problem::new(Cubic { c: 2.0 }), IterState::new());
        assert_error!(res, ArgminError, message);

        let solver = NewtonRoot::new(-3.0f64, -2.0f64).unwrap();
        let res = Executor::new(Cubic { c: 2.0 }, solver).run();
        assert_error!(res, ArgminError, message);
    }

    /// A root at an end of the bracket is the result, with no iteration.
    #[test]
    fn test_root_at_an_end() {
        for (min, max) in [(1.0f64, 3.0f64), (-1.0f64, 1.0f64)] {
            let solver = NewtonRoot::new(min, max).unwrap();
            let res = Executor::new(Cubic { c: 1.0 }, solver)
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap();
            assert_eq!(
                res.state().get_param().unwrap().to_ne_bytes(),
                1.0f64.to_ne_bytes()
            );
            assert_eq!(res.state().get_iter(), 0);
        }
    }

    /// A larger tolerance stops earlier, further from the root.
    #[test]
    fn test_tolerance_stops_earlier() {
        let run = |tol: f64| {
            let solver = NewtonRoot::new(0.0f64, 2.0f64)
                .unwrap()
                .with_tolerance(tol)
                .unwrap();
            Executor::new(Cubic { c: 2.0 }, solver)
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap()
        };
        let loose = run(1e-3);
        let tight = run(4.0 * f64::EPSILON);
        assert!(loose.state().get_iter() < tight.state().get_iter());
        assert_relative_eq!(
            *loose.state().get_param().unwrap(),
            2.0f64.cbrt(),
            max_relative = 1e-3
        );
    }

    /// The maximum number of iterations stops the solver.
    #[test]
    fn test_max_iters() {
        let solver = NewtonRoot::new(-10.0f64, 20.0f64).unwrap();
        let res = Executor::new(Arctangent {}, solver)
            .configure(|state| state.max_iters(2))
            .run()
            .unwrap();
        assert_eq!(res.state().get_iter(), 2);
        assert_eq!(
            res.state().get_termination_reason(),
            Some(&TerminationReason::MaxItersReached)
        );
    }

    /// The loop that commol inverts a distribution function with: the same
    /// steps for an increasing function, with no evaluation at the ends.
    #[allow(clippy::float_cmp)]
    fn reference(problem: &Logistic, mut low: f64, mut high: f64, steps: u64) -> f64 {
        let mut x = 0.5 * (low + high);
        for _ in 0..steps {
            let error = problem.cost(&x).unwrap();
            if error == 0.0 {
                return x;
            }
            if error > 0.0 {
                high = x;
            } else {
                low = x;
            }
            let density = problem.gradient(&x).unwrap();
            let newton = x - error / density;
            let next = if density > 0.0 && newton > low && newton < high {
                newton
            } else {
                0.5 * (low + high)
            };
            if (next - x).abs() <= 4.0 * f64::EPSILON * x.abs() || next == low || next == high {
                return next;
            }
            x = next;
        }
        x
    }

    /// The solver gives, bit for bit, the result of the loop it replaces.
    #[test]
    fn test_same_result_as_reference_loop() {
        for steps in [1, 2, 3, 100] {
            for p in [1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6] {
                let problem = Logistic { p };
                let expected = reference(&problem, -40.0, 40.0, steps);
                let solver = NewtonRoot::new(-40.0f64, 40.0f64).unwrap();
                let res = Executor::new(problem, solver)
                    .configure(|state| state.max_iters(steps))
                    .run()
                    .unwrap();
                assert_eq!(
                    res.state().get_param().unwrap().to_ne_bytes(),
                    expected.to_ne_bytes(),
                    "p = {p}, steps = {steps}"
                );
            }
        }
    }

    /// Runs the solver on `problem` in `[min, max]`, with the values of f at
    /// the ends when `ends` is given.
    fn solve<O>(
        problem: O,
        min: f64,
        max: f64,
        ends: Option<(f64, f64)>,
        steps: u64,
    ) -> IterState<f64, (), (), (), (), f64>
    where
        O: CostFunction<Param = f64, Output = f64> + Gradient<Param = f64, Gradient = f64>,
    {
        let mut solver = NewtonRoot::new(min, max).unwrap();
        if let Some((f_min, f_max)) = ends {
            solver = solver.with_bracket_values(f_min, f_max).unwrap();
        }
        Executor::new(problem, solver)
            .configure(|state| state.counting(true).max_iters(steps))
            .run()
            .unwrap()
            .state()
            .clone()
    }

    #[test]
    fn test_bracket_values() {
        let NewtonRoot {
            f_low,
            f_high,
            ends_known,
            ..
        } = NewtonRoot::new(0.0f64, 2.0f64)
            .unwrap()
            .with_bracket_values(-2.0, 6.0)
            .unwrap();
        assert_eq!(f_low.to_ne_bytes(), (-2.0f64).to_ne_bytes());
        assert_eq!(f_high.to_ne_bytes(), 6.0f64.to_ne_bytes());
        assert!(ends_known);
    }

    /// Known values at the ends save the two evaluations of f there, and
    /// the result does not change.
    #[test]
    fn test_bracket_values_save_two_evaluations() {
        let without = solve(Cubic { c: 2.0 }, 0.0, 2.0, None, 100);
        let with = solve(Cubic { c: 2.0 }, 0.0, 2.0, Some((-2.0, 6.0)), 100);
        assert_eq!(
            count(&with, "cost_count") + 2,
            count(&without, "cost_count")
        );
        assert_eq!(
            count(&with, "gradient_count"),
            count(&without, "gradient_count")
        );
        assert_eq!(
            with.get_param().unwrap().to_ne_bytes(),
            without.get_param().unwrap().to_ne_bytes()
        );

        for steps in [1, 2, 3, 100] {
            for p in [1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6] {
                let problem = Logistic { p };
                let ends = (problem.cost(&-40.0).unwrap(), problem.cost(&40.0).unwrap());
                let expected = reference(&problem, -40.0, 40.0, steps);
                let without = solve(problem.clone(), -40.0, 40.0, None, steps);
                let with = solve(problem, -40.0, 40.0, Some(ends), steps);
                assert_eq!(
                    count(&with, "cost_count") + 2,
                    count(&without, "cost_count"),
                    "p = {p}, steps = {steps}"
                );
                assert_eq!(
                    with.get_param().unwrap().to_ne_bytes(),
                    expected.to_ne_bytes(),
                    "p = {p}, steps = {steps}"
                );
            }
        }
    }

    /// Known values with no change of sign give the error of evaluated ones.
    #[test]
    fn test_bracket_values_no_change_of_sign() {
        let message = concat!(
            "Invalid parameter: \"`NewtonRoot`: ",
            "f(min) and f(max) must have different signs.\""
        );
        let mut solver = NewtonRoot::new(2.0f64, 3.0f64)
            .unwrap()
            .with_bracket_values(6.0, 25.0)
            .unwrap();
        let mut problem = Problem::new(Cubic { c: 2.0 });
        let res = solver.init(&mut problem, IterState::new());
        assert_error!(res, ArgminError, message);
        // The error comes from the given values: f is not evaluated.
        assert!(problem.counts.is_empty());
    }

    /// A value that is not finite is refused.
    #[test]
    fn test_bracket_values_not_finite() {
        let message = "Invalid parameter: \"`NewtonRoot`: f(min) and f(max) must be finite.\"";
        for (f_min, f_max) in [
            (f64::NAN, 1.0),
            (-1.0, f64::NAN),
            (f64::NEG_INFINITY, 1.0),
            (-1.0, f64::INFINITY),
        ] {
            let res = NewtonRoot::new(0.0f64, 2.0f64)
                .unwrap()
                .with_bracket_values(f_min, f_max);
            assert_error!(res, ArgminError, message);
        }
    }

    /// A known value of 0 at an end makes that end the result.
    #[test]
    fn test_bracket_values_root_at_an_end() {
        let state = solve(Cubic { c: 1.0 }, 1.0, 3.0, Some((0.0, 26.0)), 100);
        assert_eq!(
            state.get_param().unwrap().to_ne_bytes(),
            1.0f64.to_ne_bytes()
        );
        assert_eq!(count(&state, "cost_count"), 0);
    }
}
