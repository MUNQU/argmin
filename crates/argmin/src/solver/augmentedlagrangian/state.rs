// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State, TerminationReason, TerminationStatus};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use web_time::Duration;

/// Maintains the state of [`AugmentedLagrangian`](`super::AugmentedLagrangian`) from one
/// outer iteration to the next.
///
/// Beside the parameter vector and its cost, it holds the Lagrange multipliers `λ`, the
/// penalty weight `μ`, the constraint values `c(x)` and their norm `‖c(x)‖`, and the
/// iterations of the inner solver.
///
/// The cost is the cost `f(x)` of the problem, not the augmented cost of the inner solver.
/// The best parameter vector is always the current one: a lower cost at a point that
/// violates the constraints more is not a better point, and each outer iteration starts the
/// inner solver from the previous one.
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct AugmentedLagrangianState<P, M, F> {
    /// Current parameter vector
    pub param: Option<P>,
    /// Previous parameter vector
    pub prev_param: Option<P>,
    /// Current best parameter vector (always the current one)
    pub best_param: Option<P>,
    /// Previous best parameter vector
    pub prev_best_param: Option<P>,
    /// Current cost `f(x)`
    pub cost: F,
    /// Previous cost
    pub prev_cost: F,
    /// Current best cost (always the current one)
    pub best_cost: F,
    /// Previous best cost
    pub prev_best_cost: F,
    /// Target cost
    pub target_cost: F,
    /// Lagrange multipliers `λ`, with the convention `L(x, λ) = f(x) + λᵀc(x)`
    pub multipliers: Option<M>,
    /// Constraint values `c(x)` at the current parameter vector
    pub constraints: Option<M>,
    /// Norm `‖c(x)‖` of the constraint values at the current parameter vector
    pub constraint_norm: F,
    /// Penalty weight `μ` of the next outer iteration
    pub penalty: F,
    /// Iterations of the last inner run
    pub inner_iters: u64,
    /// Iterations of all inner runs
    pub inner_iters_total: u64,
    /// Whether the last inner run converged
    pub inner_converged: bool,
    /// Current outer iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of outer iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Update evaluation counts?
    pub counting_enabled: bool,
    /// Time required so far
    pub time: Option<Duration>,
    /// Status of optimization execution
    pub termination_status: TerminationStatus,
}

impl<P, M, F> AugmentedLagrangianState<P, M, F>
where
    Self: State<Float = F>,
    F: ArgminFloat,
{
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// # let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    /// #     AugmentedLagrangianState::new();
    /// let state = state.param(vec![1.0, 2.0]);
    /// # assert_eq!(state.param, Some(vec![1.0, 2.0]));
    /// ```
    #[must_use]
    pub fn param(mut self, param: P) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set the current cost `f(x)`. This shifts the stored cost to the previous cost.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// # let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    /// #     AugmentedLagrangianState::new();
    /// let state = state.cost(12.0);
    /// # assert_eq!(state.cost, 12.0);
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// # let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    /// #     AugmentedLagrangianState::new();
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost, 0.0);
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set the maximum number of outer iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// # let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    /// #     AugmentedLagrangianState::new();
    /// let state = state.max_iters(100);
    /// # assert_eq!(state.max_iters, 100);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Set whether evaluation counts are stored in the state.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// # let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    /// #     AugmentedLagrangianState::new();
    /// let state = state.counting(true);
    /// # assert!(state.counting_enabled);
    /// ```
    #[must_use]
    pub fn counting(mut self, mode: bool) -> Self {
        self.counting_enabled = mode;
        self
    }

    /// Returns the current cost `f(x)`.
    pub fn get_cost(&self) -> F {
        self.cost
    }

    /// Returns the previous cost.
    pub fn get_prev_cost(&self) -> F {
        self.prev_cost
    }

    /// Returns the Lagrange multipliers `λ`.
    ///
    /// They follow the convention `L(x, λ) = f(x) + λᵀc(x)`: at a solution, `-λᵢ` is the
    /// derivative of the optimal cost with respect to a shift `c(x) = bᵢ` of constraint `i`.
    pub fn get_multipliers(&self) -> Option<&M> {
        self.multipliers.as_ref()
    }

    /// Moves the Lagrange multipliers `λ` out of the state.
    pub fn take_multipliers(&mut self) -> Option<M> {
        self.multipliers.take()
    }

    /// Returns the constraint values `c(x)` at the current parameter vector.
    pub fn get_constraints(&self) -> Option<&M> {
        self.constraints.as_ref()
    }

    /// Returns the norm `‖c(x)‖` of the constraint values at the current parameter vector.
    pub fn get_constraint_norm(&self) -> F {
        self.constraint_norm
    }

    /// Returns the penalty weight `μ` that the next outer iteration uses.
    pub fn get_penalty(&self) -> F {
        self.penalty
    }

    /// Returns the iterations of the last inner run.
    pub fn get_inner_iters(&self) -> u64 {
        self.inner_iters
    }

    /// Returns the iterations of all inner runs.
    pub fn get_inner_iters_total(&self) -> u64 {
        self.inner_iters_total
    }

    /// Returns whether the last inner run converged.
    pub fn get_inner_converged(&self) -> bool {
        self.inner_converged
    }
}

impl<P, M, F> State for AugmentedLagrangianState<P, M, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Type of parameter vector
    type Param = P;
    /// Floating point precision
    type Float = F;

    /// Create a new `AugmentedLagrangianState` instance
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// let state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    ///     AugmentedLagrangianState::new();
    /// # assert!(state.multipliers.is_none());
    /// # assert_eq!(state.penalty, 0.0);
    /// ```
    fn new() -> Self {
        AugmentedLagrangianState {
            param: None,
            prev_param: None,
            best_param: None,
            prev_best_param: None,
            cost: F::infinity(),
            prev_cost: F::infinity(),
            best_cost: F::infinity(),
            prev_best_cost: F::infinity(),
            target_cost: F::neg_infinity(),
            multipliers: None,
            constraints: None,
            constraint_norm: F::infinity(),
            penalty: F::zero(),
            inner_iters: 0,
            inner_iters_total: 0,
            inner_converged: false,
            iter: 0,
            last_best_iter: 0,
            max_iters: u64::MAX,
            counts: HashMap::new(),
            counting_enabled: false,
            time: Some(Duration::ZERO),
            termination_status: TerminationStatus::NotTerminated,
        }
    }

    /// Makes the current parameter vector and its cost the best ones.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use argmin::solver::augmentedlagrangian::AugmentedLagrangianState;
    /// let mut state: AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64> =
    ///     AugmentedLagrangianState::new().param(vec![1.0]).cost(2.0);
    /// state.best_cost = 1.0;
    /// state.update();
    /// assert_eq!(state.best_param, Some(vec![1.0]));
    /// assert_eq!(state.best_cost, 2.0);
    /// ```
    fn update(&mut self) {
        if let Some(param) = self.param.as_ref().cloned() {
            std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
            self.best_param = Some(param);
        }
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = self.cost;
        self.last_best_iter = self.iter;
    }

    fn get_param(&self) -> Option<&P> {
        self.param.as_ref()
    }

    fn get_best_param(&self) -> Option<&P> {
        self.best_param.as_ref()
    }

    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_status = TerminationStatus::Terminated(reason);
        self
    }

    fn time(&mut self, time: Option<Duration>) -> &mut Self {
        self.time = time;
        self
    }

    fn get_cost(&self) -> Self::Float {
        self.cost
    }

    fn get_best_cost(&self) -> Self::Float {
        self.best_cost
    }

    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    fn get_iter(&self) -> u64 {
        self.iter
    }

    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    fn get_termination_status(&self) -> &TerminationStatus {
        &self.termination_status
    }

    fn get_termination_reason(&self) -> Option<&TerminationReason> {
        match &self.termination_status {
            TerminationStatus::Terminated(reason) => Some(reason),
            TerminationStatus::NotTerminated => None,
        }
    }

    fn get_time(&self) -> Option<Duration> {
        self.time
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        if self.counting_enabled {
            for (k, &v) in problem.counts.iter() {
                let count = self.counts.entry(k.to_string()).or_insert(0);
                *count = v
            }
        }
    }

    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}
