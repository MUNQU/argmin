// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Greedy subset selection.
//!
//! Greedy subset selection minimizes a user-supplied cost over sorted subsets
//! of item indices. The implementation combines pair bootstrap, greedy growth,
//! seeded random restarts, and first-improvement 2-opt local search.

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, Solver, State, TerminationReason,
    TerminationStatus, KV,
};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{cmp::Ordering, collections::BinaryHeap, ops::RangeInclusive};

const EPSILON: f64 = 1e-12;
const BOOTSTRAP_SCAN_LIMIT: usize = 200;

/// A subset-selection cost function over a finite set of indexed items.
///
/// Parameters passed to [`CostFunction::cost`] are sorted ascending, duplicate
/// free `Vec<usize>` subsets. Override the incremental methods when a marginal
/// add, removal, or swap can be evaluated more cheaply than a full cost call.
pub trait SubsetSelectionCost: CostFunction<Param = Vec<usize>, Output = f64> + Sync {
    /// Total number of items available for selection.
    fn n_items(&self) -> usize;

    /// Cost of `current` with `candidate` inserted.
    ///
    /// `candidate` must not already be selected. `current` is sorted ascending.
    fn cost_with_added(&self, current: &[usize], candidate: usize) -> Result<f64, Error> {
        let mut next = Vec::with_capacity(current.len() + 1);
        next.extend_from_slice(current);
        insert_sorted(&mut next, candidate);
        self.cost(&next)
    }

    /// Cost of `current` with `removed` deleted.
    ///
    /// `removed` must be selected. `current` is sorted ascending.
    fn cost_with_removed(&self, current: &[usize], removed: usize) -> Result<f64, Error> {
        let mut next = Vec::with_capacity(current.len().saturating_sub(1));
        next.extend(current.iter().copied().filter(|&item| item != removed));
        self.cost(&next)
    }

    /// Cost of `current` with `removed` replaced by `added`.
    ///
    /// `removed` must be selected and `added` must not be selected.
    fn cost_with_swapped(
        &self,
        current: &[usize],
        removed: usize,
        added: usize,
    ) -> Result<f64, Error> {
        let mut next = Vec::with_capacity(current.len());
        next.extend(current.iter().copied().filter(|&item| item != removed));
        insert_sorted(&mut next, added);
        self.cost(&next)
    }
}

/// A solved subset.
#[derive(Clone, Debug, PartialEq)]
pub struct Subset {
    /// Sorted ascending selected indices.
    pub indices: Vec<usize>,
    /// Cost reported by [`SubsetSelectionCost`].
    pub cost: f64,
}

/// A non-dominated set of subsets indexed by size.
#[derive(Clone, Debug, PartialEq)]
pub struct ParetoFront {
    /// Non-dominated solutions sorted by size ascending.
    pub solutions: Vec<Subset>,
    /// Index into [`ParetoFront::solutions`] of the recommended solution.
    pub selected_index: usize,
}

/// Strategy for selecting the recommended solution from a [`ParetoFront`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParetoSelection {
    /// Pick the smallest cost, breaking ties by smaller size.
    MinCost,
    /// Pick the smallest valid size.
    MinSize,
    /// Pick the largest valid size.
    MaxSize,
    /// Pick the point farthest from the line joining the front endpoints.
    Knee,
}

/// Constraint on the selected subset size.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SizeMode {
    /// Subset must have exactly this size.
    Fixed(usize),
    /// Subset size must lie in `[min, max]` inclusive.
    Bounded {
        /// Minimum subset size.
        min: usize,
        /// Maximum subset size.
        max: usize,
    },
}

/// Greedy subset selection with seeded random restarts and 2-opt refinement.
#[derive(Clone, Debug)]
pub struct GreedySubsetSelection {
    /// Subset size constraint.
    pub size_mode: SizeMode,
    /// Number of starts. Start 0 is greedy; later starts are random.
    pub n_starts: usize,
    /// Seed used to derive all random starts.
    pub seed: u64,
    /// Maximum number of accepted 2-opt swaps per start.
    pub max_local_iterations: usize,
}

impl GreedySubsetSelection {
    /// Construct a new greedy subset selection solver.
    #[must_use]
    pub fn new(size_mode: SizeMode) -> Self {
        Self {
            size_mode,
            n_starts: 5,
            seed: 0,
            max_local_iterations: usize::MAX,
        }
    }

    /// Set the number of starts.
    #[must_use]
    pub fn with_n_starts(mut self, n_starts: usize) -> Self {
        self.n_starts = n_starts;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the maximum number of accepted local-search swaps per start.
    #[must_use]
    pub fn with_max_local_iterations(mut self, max_local_iterations: usize) -> Self {
        self.max_local_iterations = max_local_iterations;
        self
    }
}

impl<O, F> Solver<O, IterState<Vec<usize>, (), (), (), (), F>> for GreedySubsetSelection
where
    O: SubsetSelectionCost,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "GreedySubsetSelection"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<Vec<usize>, (), (), (), (), F>,
    ) -> Result<(IterState<Vec<usize>, (), (), (), (), F>, Option<KV>), Error> {
        let best = problem.problem("cost_count", |problem| match self.size_mode {
            SizeMode::Fixed(target_size) => solve(
                problem,
                target_size,
                self.n_starts,
                self.seed,
                self.max_local_iterations,
            ),
            SizeMode::Bounded { min, max } => {
                let pareto = solve_pareto(
                    problem,
                    min..=max,
                    self.n_starts,
                    self.seed,
                    self.max_local_iterations,
                    ParetoSelection::MinCost,
                )?;
                Ok(pareto.solutions[pareto.selected_index].clone())
            }
        })?;
        let cost = F::from(best.cost).ok_or_else(|| -> Error {
            argmin_error!(
                InvalidParameter,
                "`GreedySubsetSelection`: cost is not representable in the state's float type."
            )
        })?;
        let mut next = state
            .param(best.indices)
            .cost(cost)
            .terminate_with(TerminationReason::SolverConverged);
        next.update();
        Ok((next, None))
    }

    fn next_iter(
        &mut self,
        _problem: &mut Problem<O>,
        state: IterState<Vec<usize>, (), (), (), (), F>,
    ) -> Result<(IterState<Vec<usize>, (), (), (), (), F>, Option<KV>), Error> {
        Ok((
            state.terminate_with(TerminationReason::SolverConverged),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<Vec<usize>, (), (), (), (), F>) -> TerminationStatus {
        state.get_termination_status().clone()
    }
}

/// Run greedy subset selection for one fixed target size.
pub fn solve<P>(
    problem: &P,
    target_size: usize,
    n_starts: usize,
    seed: u64,
    max_local_iterations: usize,
) -> Result<Subset, Error>
where
    P: SubsetSelectionCost,
{
    validate_target(problem.n_items(), target_size, n_starts)?;
    let max_local_iterations = local_iteration_limit(target_size, max_local_iterations);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut best = local_search(problem, greedy(problem, target_size)?, max_local_iterations)?;

    let start_seeds: Vec<_> = (1..n_starts).map(|_| rng.random::<u64>()).collect();
    #[cfg(feature = "rayon")]
    let candidates: Vec<_> = start_seeds
        .par_iter()
        .map(|&start_seed| random_start(problem, target_size, start_seed, max_local_iterations))
        .collect();
    #[cfg(not(feature = "rayon"))]
    let candidates: Vec<_> = start_seeds
        .iter()
        .map(|&start_seed| random_start(problem, target_size, start_seed, max_local_iterations))
        .collect();

    for candidate in candidates {
        let candidate = candidate?;
        if subset_cmp(&candidate, &best) == Ordering::Less {
            best = candidate;
        }
    }
    Ok(best)
}

/// Run [`solve`] at every size in `size_range` and assemble a Pareto front.
pub fn solve_pareto<P>(
    problem: &P,
    size_range: RangeInclusive<usize>,
    n_starts_per_size: usize,
    seed: u64,
    max_local_iterations: usize,
    selection: ParetoSelection,
) -> Result<ParetoFront, Error>
where
    P: SubsetSelectionCost,
{
    let (min, max) = (*size_range.start(), *size_range.end());
    if min > max {
        return Err(argmin_error!(
            InvalidParameter,
            "`GreedySubsetSelection`: size range must not be empty."
        ));
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut solutions = Vec::with_capacity(max - min + 1);
    for target_size in size_range {
        let size_seed = rng.random::<u64>();
        solutions.push(solve(
            problem,
            target_size,
            n_starts_per_size,
            size_seed,
            max_local_iterations,
        )?);
    }
    let solutions = nondominated(solutions);
    let selected_index = select_pareto(&solutions, selection);
    Ok(ParetoFront {
        solutions,
        selected_index,
    })
}

#[derive(Copy, Clone, Debug)]
struct HeapEntry {
    delta: f64,
    cost: f64,
    item: usize,
    round: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.delta.to_bits() == other.delta.to_bits()
            && self.cost.to_bits() == other.cost.to_bits()
            && self.item == other.item
            && self.round == other.round
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f64(self.delta, other.delta)
            .then_with(|| other.item.cmp(&self.item))
            .then_with(|| other.round.cmp(&self.round))
    }
}

fn greedy<P>(problem: &P, target_size: usize) -> Result<Subset, Error>
where
    P: SubsetSelectionCost,
{
    let mut selected = bootstrap_initial_pair(problem)?;
    let mut selected_mask = selected_mask(problem.n_items(), &selected);
    let mut cost_now = problem.cost(&selected)?;
    if target_size == 2 {
        return Ok(Subset {
            indices: selected,
            cost: cost_now,
        });
    }
    let mut round = 0usize;
    let mut heap = BinaryHeap::new();
    for (item, is_selected) in selected_mask.iter().copied().enumerate() {
        if !is_selected {
            let cost = problem.cost_with_added(&selected, item)?;
            let delta = cost_now - cost;
            heap.push(HeapEntry {
                delta,
                cost,
                item,
                round,
            });
        }
    }
    while selected.len() < target_size {
        let accepted = loop {
            let entry = heap.pop().ok_or_else(|| -> Error {
                argmin_error!(
                    PotentialBug,
                    "`GreedySubsetSelection`: candidate heap unexpectedly empty."
                )
            })?;
            if selected_mask[entry.item] {
                continue;
            }
            if entry.round == round {
                break (entry.item, entry.cost);
            }
            let true_cost = problem.cost_with_added(&selected, entry.item)?;
            let true_delta = cost_now - true_cost;
            let current_top = heap.peek().map_or(f64::NEG_INFINITY, |top| top.delta);
            if cmp_f64(true_delta, current_top) != Ordering::Less {
                break (entry.item, true_cost);
            }
            heap.push(HeapEntry {
                delta: true_delta,
                cost: true_cost,
                item: entry.item,
                round,
            });
        };
        insert_sorted(&mut selected, accepted.0);
        selected_mask[accepted.0] = true;
        round += 1;
        cost_now = accepted.1;
    }
    Ok(Subset {
        indices: selected,
        cost: cost_now,
    })
}

fn bootstrap_initial_pair<P>(problem: &P) -> Result<Vec<usize>, Error>
where
    P: SubsetSelectionCost,
{
    let scan = problem.n_items().min(BOOTSTRAP_SCAN_LIMIT);
    let mut best: Option<Subset> = None;
    for first in 0..scan {
        for second in (first + 1)..scan {
            let indices = vec![first, second];
            let cost = problem.cost(&indices)?;
            let candidate = Subset { indices, cost };
            if best
                .as_ref()
                .is_none_or(|current| subset_cmp(&candidate, current) == Ordering::Less)
            {
                best = Some(candidate);
            }
        }
    }
    best.map(|subset| subset.indices).ok_or_else(|| -> Error {
        argmin_error!(
            InvalidParameter,
            "`GreedySubsetSelection`: at least two items are required."
        )
    })
}

fn local_search<P>(
    problem: &P,
    initial: Subset,
    max_local_iterations: usize,
) -> Result<Subset, Error>
where
    P: SubsetSelectionCost,
{
    let mut selected = initial.indices;
    let mut selected_mask = selected_mask(problem.n_items(), &selected);
    let mut unselected: Vec<_> = (0..problem.n_items())
        .filter(|&item| !selected_mask[item])
        .collect();
    let mut cost_now = if initial.cost.is_infinite() && initial.cost.is_sign_positive() {
        problem.cost(&selected)?
    } else {
        initial.cost
    };
    let mut iterations = 0usize;
    while iterations < max_local_iterations {
        let mut improved = false;
        let out_iter = selected.clone();
        'swaps: for removed in out_iter {
            for added in unselected.iter().copied() {
                let new_cost = problem.cost_with_swapped(&selected, removed, added)?;
                if new_cost < cost_now - EPSILON {
                    remove_sorted(&mut selected, removed);
                    insert_sorted(&mut selected, added);
                    selected_mask[removed] = false;
                    selected_mask[added] = true;
                    remove_sorted(&mut unselected, added);
                    insert_sorted(&mut unselected, removed);
                    cost_now = new_cost;
                    iterations += 1;
                    improved = true;
                    break 'swaps;
                }
            }
        }
        if !improved {
            break;
        }
    }
    Ok(Subset {
        indices: selected,
        cost: cost_now,
    })
}

fn random_start<P>(
    problem: &P,
    target_size: usize,
    seed: u64,
    max_local_iterations: usize,
) -> Result<Subset, Error>
where
    P: SubsetSelectionCost,
{
    let random = random_subset(problem.n_items(), target_size, seed);
    local_search(problem, random, max_local_iterations)
}

fn random_subset(n_items: usize, target_size: usize, seed: u64) -> Subset {
    let mut items: Vec<_> = (0..n_items).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    items.partial_shuffle(&mut rng, target_size);
    let mut indices = items[..target_size].to_vec();
    indices.sort_unstable();
    Subset {
        indices,
        cost: f64::INFINITY,
    }
}

fn selected_mask(n_items: usize, selected: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; n_items];
    for &item in selected {
        mask[item] = true;
    }
    mask
}

fn nondominated(mut solutions: Vec<Subset>) -> Vec<Subset> {
    solutions.sort_by(|left, right| {
        left.indices
            .len()
            .cmp(&right.indices.len())
            .then_with(|| subset_cmp(left, right))
    });
    let mut nondom = Vec::with_capacity(solutions.len());
    let mut best_cost_so_far = f64::INFINITY;
    for solution in solutions {
        if cmp_f64(solution.cost, best_cost_so_far) == Ordering::Less {
            best_cost_so_far = solution.cost;
            nondom.push(solution);
        }
    }
    nondom
}

fn select_pareto(solutions: &[Subset], selection: ParetoSelection) -> usize {
    match selection {
        ParetoSelection::MinCost => solutions
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| subset_cmp(left, right))
            .map_or(0, |(index, _)| index),
        ParetoSelection::MinSize => 0,
        ParetoSelection::MaxSize => solutions.len().saturating_sub(1),
        ParetoSelection::Knee => select_knee(solutions),
    }
}

fn select_knee(solutions: &[Subset]) -> usize {
    if solutions.len() < 3 {
        return select_pareto(solutions, ParetoSelection::MinCost);
    }
    let first = &solutions[0];
    let last = &solutions[solutions.len() - 1];
    let x1 = first.indices.len() as f64;
    let y1 = first.cost;
    let x2 = last.indices.len() as f64;
    let y2 = last.cost;
    let denominator = ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt();
    if denominator <= EPSILON {
        return select_pareto(solutions, ParetoSelection::MinCost);
    }
    solutions
        .iter()
        .enumerate()
        .skip(1)
        .take(solutions.len() - 2)
        .max_by(|(_, left), (_, right)| {
            let left_distance =
                point_line_distance(left.indices.len() as f64, left.cost, x1, y1, x2, y2);
            let right_distance =
                point_line_distance(right.indices.len() as f64, right.cost, x1, y1, x2, y2);
            cmp_f64(left_distance, right_distance)
        })
        .map_or(0, |(index, _)| index)
}

fn point_line_distance(x: f64, y: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1).abs()
        / ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt()
}

fn validate_target(n_items: usize, target_size: usize, n_starts: usize) -> Result<(), Error> {
    if n_starts == 0 {
        return Err(argmin_error!(
            InvalidParameter,
            "`GreedySubsetSelection`: n_starts must be at least 1."
        ));
    }
    if target_size < 2 {
        return Err(argmin_error!(
            InvalidParameter,
            "`GreedySubsetSelection`: target_size must be at least 2."
        ));
    }
    if target_size > n_items {
        return Err(argmin_error!(
            InvalidParameter,
            "`GreedySubsetSelection`: target_size must not exceed n_items."
        ));
    }
    Ok(())
}

fn local_iteration_limit(target_size: usize, configured: usize) -> usize {
    if configured == usize::MAX {
        100usize.saturating_mul(target_size)
    } else {
        configured
    }
}

fn insert_sorted(values: &mut Vec<usize>, item: usize) {
    let insert_at = values.partition_point(|&current| current < item);
    values.insert(insert_at, item);
}

fn remove_sorted(values: &mut Vec<usize>, item: usize) {
    if let Ok(index) = values.binary_search(&item) {
        values.remove(index);
    }
}

fn subset_cmp(left: &Subset, right: &Subset) -> Ordering {
    cmp_f64(left.cost, right.cost)
        .then_with(|| left.indices.len().cmp(&right.indices.len()))
        .then_with(|| left.indices.cmp(&right.indices))
}

fn cmp_f64(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Executor;

    #[derive(Clone)]
    struct MaxCover {
        coverage: Vec<Vec<bool>>,
        n_elements: usize,
    }

    impl MaxCover {
        fn exact_fixture() -> Self {
            let mut coverage = vec![vec![false; 50]; 20];
            for (item, range) in [(1, 0..13), (4, 13..25), (9, 25..38), (14, 38..50)] {
                for element in range {
                    coverage[item][element] = true;
                }
            }
            for item in 0..20 {
                if ![1, 4, 9, 14].contains(&item) {
                    coverage[item][item % 50] = true;
                }
            }
            Self {
                coverage,
                n_elements: 50,
            }
        }
    }

    impl CostFunction for MaxCover {
        type Param = Vec<usize>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            let mut covered = vec![false; self.n_elements];
            for &item in param {
                for (element, covers) in self.coverage[item].iter().copied().enumerate() {
                    covered[element] |= covers;
                }
            }
            Ok(-(covered.into_iter().filter(|covered| *covered).count() as f64))
        }
    }

    impl SubsetSelectionCost for MaxCover {
        fn n_items(&self) -> usize {
            self.coverage.len()
        }
    }

    #[derive(Clone)]
    struct PairCost {
        matrix: Vec<Vec<f64>>,
    }

    impl CostFunction for PairCost {
        type Param = Vec<usize>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            let mut cost = 0.0;
            for &left in param {
                for &right in param {
                    if left != right {
                        cost += self.matrix[left][right];
                    }
                }
            }
            Ok(cost)
        }
    }

    impl SubsetSelectionCost for PairCost {
        fn n_items(&self) -> usize {
            self.matrix.len()
        }
    }

    #[test]
    fn greedy_finds_exact_cover_fixture() {
        let problem = MaxCover::exact_fixture();
        let result = solve(&problem, 4, 1, 0, usize::MAX).unwrap();
        assert_eq!(result.indices, vec![1, 4, 9, 14]);
        assert_eq!(result.cost.to_bits(), (-50.0f64).to_bits());
    }

    #[test]
    fn default_incremental_methods_match_full_cost_path() {
        let matrix = vec![
            vec![0.0, -3.0, -2.0, 1.0, 2.0, 1.0],
            vec![-3.0, 0.0, -4.0, 1.0, 2.0, 1.0],
            vec![-2.0, -4.0, 0.0, 1.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 0.0, -1.0, -1.0],
            vec![2.0, 2.0, 2.0, -1.0, 0.0, -1.0],
            vec![1.0, 1.0, 1.0, -1.0, -1.0, 0.0],
        ];
        let problem = PairCost { matrix };
        let result = solve(&problem, 3, 5, 123, usize::MAX).unwrap();
        assert_eq!(result.indices, vec![0, 1, 2]);
    }

    #[test]
    fn repeated_seed_is_deterministic() {
        let problem = MaxCover::exact_fixture();
        let first = solve(&problem, 5, 8, 999, usize::MAX).unwrap();
        for _ in 0..10 {
            let next = solve(&problem, 5, 8, 999, usize::MAX).unwrap();
            assert_eq!(next, first);
        }
    }

    #[test]
    fn pareto_selection_min_cost_is_valid() {
        let problem = MaxCover::exact_fixture();
        let front =
            solve_pareto(&problem, 2..=4, 2, 7, usize::MAX, ParetoSelection::MinCost).unwrap();
        assert!(!front.solutions.is_empty());
        assert!(front.selected_index < front.solutions.len());
        assert_eq!(
            front.solutions[front.selected_index].cost.to_bits(),
            (-50.0f64).to_bits()
        );
    }

    #[test]
    fn executor_smoke_test() {
        let problem = MaxCover::exact_fixture();
        let solver = GreedySubsetSelection::new(SizeMode::Fixed(4))
            .with_n_starts(1)
            .with_seed(0);
        let result =
            Executor::<_, _, IterState<Vec<usize>, (), (), (), (), f64>>::new(problem, solver)
                .run()
                .unwrap();
        assert_eq!(
            result.state().get_best_param().unwrap(),
            &vec![1usize, 4, 9, 14]
        );
        assert!(matches!(
            result.state().get_termination_reason(),
            Some(TerminationReason::SolverConverged)
        ));
    }
}
