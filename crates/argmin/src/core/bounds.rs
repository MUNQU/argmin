// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Box bounds through a sine transform
//!
//! Solvers such as [`NelderMead`](`crate::solver::neldermead::NelderMead`) search an unbounded
//! space. [`SineTransform`] and [`BoundedProblem`] let them minimize a cost inside box bounds
//! `lo <= x <= hi`: the solver proposes unbounded coordinates `u`, and the cost scores
//!
//! ```text
//! x = lo + (hi − lo)·(sin u + 1)/2
//! ```
//!
//! Every `u` maps to an `x` inside the bounds, so the cost never sees a point outside them.
//!
//! The transform is better than a clamp of `x` to the bounds. A clamped cost is flat beyond a
//! bound: vertices of a simplex outside the bound all get the same cost, and the simplex can
//! collapse onto the bound and stop there. Through the sine, a step in `u` always moves `x`
//! (except at the bound itself), so the simplex keeps its shape and can still move along the
//! bound or back inside.
//!
//! A bound with `hi == lo` fixes its coordinate: `x` is `lo` for every `u`.
//!
//! ## Example
//!
//! ```
//! # use argmin::core::{BoundedProblem, CostFunction, Error, Executor, SineTransform};
//! # use argmin::solver::neldermead::NelderMead;
//! # struct Cost {}
//! # impl CostFunction for Cost {
//! #     type Param = Vec<f64>;
//! #     type Output = f64;
//! #     fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok((x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2))
//! #     }
//! # }
//! # fn main() -> Result<(), Error> {
//! let transform = SineTransform::new(vec![(0.0, 1.0), (0.0, 1.0)])?;
//! // The start and one vertex per coordinate, moved by 10% of its range.
//! let simplex = transform.initial_simplex(&[0.5, 0.5], 0.1);
//! let problem = BoundedProblem::new(Cost {}, transform);
//! let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-10)?;
//! let mut result = Executor::new(problem, solver)
//!     .configure(|state| state.max_iters(200))
//!     .run()?;
//! let problem = result.problem.take_problem().unwrap();
//! // The solver works in `u`: map its best point back to `x`.
//! let best = problem.transform().to_x(result.state.best_param.as_ref().unwrap());
//! # assert!((best[0] - 1.0).abs() < 1e-6 && best[1].abs() < 1e-6);
//! let cost = problem.into_inner();
//! # Ok(())
//! # }
//! ```
//!
//! ## References
//!
//! F. James and M. Roos (1975). "Minuit - a system for function minimization and analysis of the
//! parameter errors and correlations". Computer Physics Communications 10 (6), 343–367. The
//! transform is described in the MINUIT user's guide, section "Parameter limits".
//!
//! M. Newville et al., LMFIT: Non-Linear Least-Squares Minimization and Curve-Fitting for
//! Python, section "Bounds Implementation":
//! <https://lmfit.github.io/lmfit-py/bounds.html>

use crate::core::{ArgminFloat, CostFunction, Error, Operator};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Maps unbounded coordinates `u` to values `x` inside box bounds, one `(lo, hi)` per
/// coordinate: `x = lo + (hi − lo)·(sin u + 1)/2`.
///
/// See the [module documentation](`crate::core::bounds`) for why and how to use it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SineTransform<F> {
    /// `(lo, hi)` of each coordinate
    bounds: Vec<(F, F)>,
}

impl<F> SineTransform<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of `SineTransform` from one `(lo, hi)` per coordinate.
    ///
    /// Every bound must be finite with `lo <= hi`. A bound with `lo == hi` fixes its coordinate.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, SineTransform};
    /// # fn main() -> Result<(), Error> {
    /// let transform = SineTransform::new(vec![(0.0f64, 1.0), (-2.0, 2.0)])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(bounds: Vec<(F, F)>) -> Result<Self, Error> {
        if !bounds
            .iter()
            .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && lo <= hi)
        {
            return Err(argmin_error!(
                InvalidParameter,
                "`SineTransform`: every bound needs finite `lo <= hi`."
            ));
        }
        Ok(SineTransform { bounds })
    }

    /// Returns the `(lo, hi)` of each coordinate.
    pub fn bounds(&self) -> &[(F, F)] {
        &self.bounds
    }

    /// Bounded values `x` of the unbounded coordinates `u`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, SineTransform};
    /// # fn main() -> Result<(), Error> {
    /// let transform = SineTransform::new(vec![(0.0f64, 4.0)])?;
    /// assert_eq!(transform.to_x(&[0.0]), vec![2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_x(&self, u: &[F]) -> Vec<F> {
        u.iter()
            .zip(&self.bounds)
            .map(|(&u, &(lo, hi))| lo + (hi - lo) * (u.sin() + F::one()) / float!(2.0))
            .collect()
    }

    /// Unbounded coordinates `u` of the values `x`, clamped to the bounds first.
    ///
    /// Each `u` is in `[−π/2, π/2]`; a fixed coordinate (`lo == hi`) gets `u = 0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, SineTransform};
    /// # fn main() -> Result<(), Error> {
    /// let transform = SineTransform::new(vec![(0.0f64, 4.0)])?;
    /// assert_eq!(transform.to_u(&[2.0]), vec![0.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_u(&self, x: &[F]) -> Vec<F> {
        x.iter()
            .zip(&self.bounds)
            .map(|(&x, &(lo, hi))| {
                if hi > lo {
                    (float!(2.0) * (num_traits::clamp(x, lo, hi) - lo) / (hi - lo) - F::one())
                        .asin()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    /// Initial simplex in `u` for a solver such as
    /// [`NelderMead`](`crate::solver::neldermead::NelderMead`).
    ///
    /// The first vertex is `start`, clamped to the bounds. Each next vertex moves one coordinate
    /// by `step × (hi − lo)` in `x`, towards the farther bound (towards `hi` on a tie). A fixed
    /// coordinate (`lo == hi`) moves by 1 in `u` instead, so the simplex is not degenerate while
    /// `x` keeps its value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, SineTransform};
    /// # fn main() -> Result<(), Error> {
    /// let transform = SineTransform::new(vec![(0.0f64, 10.0), (0.0, 10.0)])?;
    /// let simplex = transform.initial_simplex(&[2.0, 8.0], 0.1);
    /// // In `x`, the vertices are (2, 8), (3, 8) and (2, 7).
    /// assert_eq!(simplex.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn initial_simplex(&self, start: &[F], step: F) -> Vec<Vec<F>> {
        let x0: Vec<F> = start
            .iter()
            .zip(&self.bounds)
            .map(|(&x, &(lo, hi))| num_traits::clamp(x, lo, hi))
            .collect();
        let u0 = self.to_u(&x0);
        let mut vertices = vec![u0.clone()];
        for (i, &(lo, hi)) in self.bounds.iter().enumerate() {
            let mut vertex = u0.clone();
            if hi > lo {
                let delta = step * (hi - lo);
                let moved = if hi - x0[i] >= x0[i] - lo {
                    x0[i] + delta
                } else {
                    x0[i] - delta
                };
                let mut x = x0.clone();
                x[i] = moved;
                vertex[i] = self.to_u(&x)[i];
            } else {
                vertex[i] = vertex[i] + F::one();
            }
            vertices.push(vertex);
        }
        vertices
    }
}

/// A cost function seen through a [`SineTransform`]: the solver proposes `u`, the wrapped cost
/// function scores `x = to_x(u)`.
///
/// The solver's parameters (`best_param`, `param`) are in `u`; map them back with
/// [`SineTransform::to_x`] through [`transform`](`BoundedProblem::transform`). Take the wrapped
/// cost function back with [`into_inner`](`BoundedProblem::into_inner`).
///
/// See the [module documentation](`crate::core::bounds`) for an example.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BoundedProblem<C, F> {
    /// Wrapped cost function, which scores `x`
    cost: C,
    /// Map from `u` to `x`
    transform: SineTransform<F>,
}

impl<C, F> BoundedProblem<C, F> {
    /// Construct a new instance of `BoundedProblem` that scores `cost` at `transform.to_x(u)`.
    pub fn new(cost: C, transform: SineTransform<F>) -> Self {
        BoundedProblem { cost, transform }
    }

    /// Returns the transform from `u` to `x`.
    pub fn transform(&self) -> &SineTransform<F> {
        &self.transform
    }

    /// Returns a reference to the wrapped cost function.
    pub fn inner(&self) -> &C {
        &self.cost
    }

    /// Returns the wrapped cost function.
    pub fn into_inner(self) -> C {
        self.cost
    }
}

impl<C, F> CostFunction for BoundedProblem<C, F>
where
    C: CostFunction<Param = Vec<F>>,
    F: ArgminFloat,
{
    type Param = Vec<F>;
    type Output = C::Output;

    fn cost(&self, u: &Self::Param) -> Result<Self::Output, Error> {
        self.cost.cost(&self.transform.to_x(u))
    }

    fn parallelize(&self) -> bool {
        self.cost.parallelize()
    }
}

/// The operator applies at `x = to_x(u)`, as the cost does, so a solver such as
/// [`AugmentedLagrangian`](`crate::solver::augmentedlagrangian::AugmentedLagrangian`) sees its
/// constraints at the bounded values.
impl<C, F> Operator for BoundedProblem<C, F>
where
    C: Operator<Param = Vec<F>>,
    F: ArgminFloat,
{
    type Param = Vec<F>;
    type Output = C::Output;

    fn apply(&self, u: &Self::Param) -> Result<Self::Output, Error> {
        self.cost.apply(&self.transform.to_x(u))
    }

    fn parallelize(&self) -> bool {
        self.cost.parallelize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, Error, Executor, IterState, State};
    use crate::solver::augmentedlagrangian::AugmentedLagrangian;
    use crate::solver::neldermead::NelderMead;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;
    use std::sync::Mutex;

    test_trait_impl!(sine_transform, SineTransform<f64>);
    test_trait_impl!(bounded_problem, BoundedProblem<TestProblem, f64>);

    /// `(x0 - 3)² + (x1 + 2)²`, which records every point it scores.
    struct Recorder {
        points: Mutex<Vec<Vec<f64>>>,
    }

    impl CostFunction for Recorder {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            self.points.lock().unwrap().push(x.clone());
            Ok((x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2))
        }
    }

    #[test]
    fn test_new_rejects_invalid_bounds() {
        for bounds in [
            vec![(1.0, 0.0)],
            vec![(0.0, f64::INFINITY)],
            vec![(f64::NAN, 1.0)],
        ] {
            let error = SineTransform::new(bounds).err().unwrap();
            assert_eq!(
                error.downcast_ref::<ArgminError>().unwrap().to_string(),
                "Invalid parameter: \"`SineTransform`: every bound needs finite `lo <= hi`.\""
            );
        }
    }

    #[test]
    fn test_to_x_stays_inside_the_bounds() {
        let transform = SineTransform::new(vec![(-1.0, 3.0)]).unwrap();
        assert_relative_eq!(transform.to_x(&[0.0])[0], 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(transform.to_x(&[FRAC_PI_2])[0], 3.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            transform.to_x(&[-FRAC_PI_2])[0],
            -1.0,
            epsilon = f64::EPSILON
        );
        for u in [-1e6, -7.3, 0.4, 12.0, 1e9] {
            let x = transform.to_x(&[u])[0];
            assert!((-1.0..=3.0).contains(&x));
        }
    }

    #[test]
    fn test_round_trip_inside_the_bounds() {
        let transform = SineTransform::new(vec![(-1.0, 3.0), (0.0, 1e-3), (2.0, 2.5)]).unwrap();
        let x = vec![-0.999, 0.5e-3, 2.4999];
        let back = transform.to_x(&transform.to_u(&x));
        for (a, b) in x.iter().zip(&back) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
        let u = vec![-1.5, 0.0, 1.2];
        let back = transform.to_u(&transform.to_x(&u));
        for (a, b) in u.iter().zip(&back) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_to_u_clamps_to_the_bounds() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (0.0, 1.0)]).unwrap();
        let u = transform.to_u(&[-5.0, 7.0]);
        assert_relative_eq!(u[0], -FRAC_PI_2, epsilon = f64::EPSILON);
        assert_relative_eq!(u[1], FRAC_PI_2, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_initial_simplex_moves_towards_the_farther_bound() {
        let transform = SineTransform::new(vec![(0.0, 10.0), (0.0, 10.0)]).unwrap();
        let simplex = transform.initial_simplex(&[2.0, 8.0], 0.1);
        assert_eq!(simplex.len(), 3);
        let x: Vec<Vec<f64>> = simplex.iter().map(|u| transform.to_x(u)).collect();
        let expected = [[2.0, 8.0], [3.0, 8.0], [2.0, 7.0]];
        for (vertex, expected) in x.iter().zip(expected) {
            for (a, b) in vertex.iter().zip(expected) {
                assert_relative_eq!(*a, b, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn test_initial_simplex_starts_at_the_clamped_start() {
        let transform = SineTransform::new(vec![(0.0, 10.0)]).unwrap();
        let simplex = transform.initial_simplex(&[-4.0], 0.25);
        let x: Vec<f64> = simplex.iter().map(|u| transform.to_x(u)[0]).collect();
        assert_relative_eq!(x[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 2.5, max_relative = 1e-12);
    }

    #[test]
    fn test_fixed_bound_keeps_its_value_and_a_full_simplex() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (2.0, 2.0)]).unwrap();
        let simplex = transform.initial_simplex(&[0.5, 9.0], 0.1);
        for u in &simplex {
            assert_relative_eq!(transform.to_x(u)[1], 2.0, epsilon = f64::EPSILON);
        }
        // The edges from the first vertex span the plane: the simplex is not degenerate.
        let e1: [f64; 2] = [simplex[1][0] - simplex[0][0], simplex[1][1] - simplex[0][1]];
        let e2 = [simplex[2][0] - simplex[0][0], simplex[2][1] - simplex[0][1]];
        assert!((e1[0] * e2[1] - e1[1] * e2[0]).abs() > 1e-3);
        assert_relative_eq!(simplex[2][1] - simplex[0][1], 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_nelder_mead_stays_inside_the_bounds() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (0.0, 1.0)]).unwrap();
        let simplex = transform.initial_simplex(&[0.5, 0.5], 0.1);
        let problem = BoundedProblem::new(
            Recorder {
                points: Mutex::new(Vec::new()),
            },
            transform,
        );
        let solver: NelderMead<Vec<f64>, f64> =
            NelderMead::new(simplex).with_sd_tolerance(1e-12).unwrap();
        let mut result = Executor::new(problem, solver)
            .configure(|state| state.max_iters(500))
            .run()
            .unwrap();

        let problem = result.problem.take_problem().unwrap();
        let best = problem
            .transform()
            .to_x(result.state.get_best_param().unwrap());
        assert_relative_eq!(best[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(best[1], 0.0, epsilon = 1e-6);

        let points = problem.into_inner().points.into_inner().unwrap();
        assert!(!points.is_empty());
        for x in &points {
            assert!(x.iter().all(|x| (0.0..=1.0).contains(x)), "{x:?}");
        }
    }

    #[test]
    fn test_bounded_problem_scores_x() {
        let transform = SineTransform::new(vec![(0.0, 4.0), (-2.0, 0.0)]).unwrap();
        let problem = BoundedProblem::new(
            Recorder {
                points: Mutex::new(Vec::new()),
            },
            transform,
        );
        let cost = problem.cost(&vec![0.0, 0.0]).unwrap();
        assert_relative_eq!(cost, 1.0 + 1.0, epsilon = 1e-12);
        assert_eq!(problem.inner().points.lock().unwrap()[0], vec![2.0, -1.0]);
    }

    /// The operator that returns its parameter.
    struct Identity {}

    impl Operator for Identity {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn apply(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            Ok(x.clone())
        }
    }

    #[test]
    fn test_bounded_problem_applies_the_operator_at_x() {
        let transform = SineTransform::new(vec![(0.0, 4.0)]).unwrap();
        let problem = BoundedProblem::new(Identity {}, transform);
        assert_eq!(problem.apply(&vec![0.0]).unwrap(), vec![2.0]);
    }

    /// `(x0 - 3)² + (x1 + 2)²` under `x0 − x1 − 0.5 = 0`.
    struct Line {}

    impl CostFunction for Line {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok((p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2))
        }
    }

    impl Operator for Line {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(vec![p[0] - p[1] - 0.5])
        }
    }

    #[test]
    fn test_augmented_lagrangian_meets_the_constraint_inside_the_bounds() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (0.0, 1.0)]).unwrap();
        let factory = |u: &Vec<f64>| {
            let simplex = transform.initial_simplex(&transform.to_x(u), 0.1);
            Ok((
                NelderMead::new(simplex).with_sd_tolerance(1e-15)?,
                IterState::new().max_iters(2000),
            ))
        };
        let solver: AugmentedLagrangian<_, Vec<f64>, f64> = AugmentedLagrangian::new(factory)
            .with_tolerance(1e-8)
            .unwrap();
        let u0 = transform.to_u(&[0.5, 0.5]);
        let problem = BoundedProblem::new(Line {}, transform.clone());
        let mut result = Executor::new(problem, solver)
            .configure(|state| state.param(u0).max_iters(50))
            .run()
            .unwrap();

        let problem = result.problem.take_problem().unwrap();
        let best = problem
            .transform()
            .to_x(result.state.get_best_param().unwrap());
        assert_relative_eq!(best[0], 0.75, epsilon = 1e-4);
        assert_relative_eq!(best[1], 0.25, epsilon = 1e-4);
        assert!(best.iter().all(|x| (0.0..=1.0).contains(x)), "{best:?}");
    }
}
