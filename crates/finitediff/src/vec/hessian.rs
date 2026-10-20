// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use anyhow::{anyhow, Error};
use num::{Float, FromPrimitive};

use crate::utils::{mod_and_calc, restore_symmetry_vec, KV};

use super::{CostFn, GradientFn};

pub fn forward_hessian_vec<F>(x: &Vec<F>, grad: GradientFn<'_, F>) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_sqrt = F::epsilon().sqrt();
    let fx = (grad)(x)?;
    let mut xt = x.clone();
    let out: Vec<Vec<F>> = (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc(&mut xt, grad, i, eps_sqrt)?;
            Ok(fx1
                .iter()
                .zip(fx.iter())
                .map(|(&a, &b)| (a - b) / eps_sqrt)
                .collect::<Vec<F>>())
        })
        .collect::<Result<_, Error>>()?;

    // restore symmetry
    Ok(restore_symmetry_vec(out))
}

pub fn central_hessian_vec<F>(x: &[F], grad: GradientFn<'_, F>) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let mut xt = x.to_owned();
    let out: Vec<Vec<F>> = (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc(&mut xt, grad, i, eps_cbrt)?;
            let fx2 = mod_and_calc(&mut xt, grad, i, -eps_cbrt)?;
            Ok(fx1
                .iter()
                .zip(fx2.iter())
                .map(|(&a, &b)| (a - b) / (F::from_f64(2.0).unwrap() * eps_cbrt))
                .collect::<Vec<F>>())
        })
        .collect::<Result<_, Error>>()?;

    // restore symmetry
    Ok(restore_symmetry_vec(out))
}

pub fn forward_hessian_vec_prod_vec<F>(
    x: &Vec<F>,
    grad: GradientFn<'_, F>,
    p: &[F],
) -> Result<Vec<F>, Error>
where
    F: Float,
{
    let eps_sqrt = F::epsilon().sqrt();
    let fx = (grad)(x)?;
    let out: Vec<F> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + pi * eps_sqrt)
            .collect();
        let fx1 = (grad)(&x1)?;
        fx1.iter()
            .zip(fx.iter())
            .map(|(&a, &b)| (a - b) / eps_sqrt)
            .collect::<Vec<F>>()
    };
    Ok(out)
}

pub fn central_hessian_vec_prod_vec<F>(
    x: &[F],
    grad: GradientFn<'_, F>,
    p: &[F],
) -> Result<Vec<F>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let out: Vec<F> = {
        // TODO: Do this in single array
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + pi * eps_cbrt)
            .collect();
        let x2 = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi - pi * eps_cbrt)
            .collect();
        let fx1 = (grad)(&x1)?;
        let fx2 = (grad)(&x2)?;
        fx1.iter()
            .zip(fx2.iter())
            .map(|(&a, &b)| (a - b) / (F::from_f64(2.0).unwrap() * eps_cbrt))
            .collect::<Vec<F>>()
    };
    Ok(out)
}

pub fn forward_hessian_nograd_vec<F>(x: &Vec<F>, f: CostFn<'_, F>) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let n = x.len();
    let mut xt = x.clone();

    // Precompute f(x + sqrt(EPS) * e_i) for all i
    let fxei: Vec<F> = (0..n)
        .map(|i| mod_and_calc(&mut xt, f, i, eps_sqrt_nograd))
        .collect::<Result<_, Error>>()?;

    let mut out: Vec<Vec<F>> = vec![vec![F::from_f64(0.0).unwrap(); n]; n];

    for i in 0..n {
        for j in 0..=i {
            let t = {
                let xti = xt[i];
                let xtj = xt[j];
                xt[i] += eps_sqrt_nograd;
                xt[j] += eps_sqrt_nograd;
                let fxij = (f)(&xt)?;
                xt[i] = xti;
                xt[j] = xtj;
                (fxij - fxei[i] - fxei[j] + fx) / eps_nograd
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    Ok(out)
}

pub fn forward_hessian_nograd_sparse_vec<F>(
    x: &Vec<F>,
    f: CostFn<'_, F>,
    indices: Vec<[usize; 2]>,
) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let n = x.len();
    let mut xt = x.clone();

    let mut idxs: Vec<usize> = indices
        .iter()
        .flat_map(|i| i.iter())
        .cloned()
        .collect::<Vec<usize>>();
    idxs.sort();
    idxs.dedup();

    let mut fxei = KV::new(idxs.len());

    for idx in idxs.iter() {
        fxei.set(*idx, mod_and_calc(&mut xt, f, *idx, eps_sqrt_nograd)?);
    }

    let mut out: Vec<Vec<F>> = vec![vec![F::from_f64(0.0).unwrap(); n]; n];
    for [i, j] in indices {
        let t = {
            let xti = xt[i];
            let xtj = xt[j];
            xt[i] += eps_sqrt_nograd;
            xt[j] += eps_sqrt_nograd;
            let fxij = (f)(&xt)?;
            xt[i] = xti;
            xt[j] = xtj;

            let fxi = fxei.get(i).ok_or(anyhow!("Bug"))?;
            let fxj = fxei.get(j).ok_or(anyhow!("Bug"))?;
            (fxij - fxi - fxj + fx) / eps_nograd
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    Ok(out)
}

/// A Hessian by central differences with the steps of the caller.
///
/// The result of [`central_hessian_nograd_steps_vec`].
#[derive(Debug, Clone, PartialEq)]
pub struct CentralHessian<F> {
    /// The Hessian, symmetric by construction.
    pub hessian: Vec<Vec<F>>,
    /// The number of evaluations of the cost: `2p² + 1` for `p` coordinates.
    pub evaluations: usize,
}

/// A Hessian by central differences with the steps of the caller, and an
/// estimate of the error of each entry.
///
/// The result of [`central_hessian_nograd_error_vec`]. `truncation` and
/// `rounding` are symmetric, as `hessian` is; [`error`](Self::error) is their
/// sum.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralHessianEstimate<F> {
    /// The Hessian with the steps `h`, symmetric by construction.
    pub hessian: Vec<Vec<F>>,
    /// The estimated truncation error of each entry: `|H(h) − H(2h)| / 3`.
    pub truncation: Vec<Vec<F>>,
    /// The bound of the error that the inaccuracy of the cost causes in each
    /// entry: `δ · Σ|c| · max|f|`, over the evaluations of the entry with
    /// the steps `h` and their coefficients `c`.
    pub rounding: Vec<Vec<F>>,
    /// The number of evaluations of the cost: `4p² + 1` for `p` coordinates.
    pub evaluations: usize,
}

impl<F: Float> CentralHessianEstimate<F> {
    /// The estimated error of each entry: `truncation + rounding`.
    pub fn error(&self) -> Vec<Vec<F>> {
        self.truncation
            .iter()
            .zip(&self.rounding)
            .map(|(t, r)| t.iter().zip(r).map(|(&t, &r)| t + r).collect())
            .collect()
    }
}

/// The value of a float for an error message.
fn shown<F: Float>(value: F) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

/// Checks that there is one step per coordinate, and that each step is
/// finite, positive and large enough to move its coordinate.
fn check_steps<F: Float>(x: &[F], steps: &[F]) -> Result<(), Error> {
    if steps.len() != x.len() {
        return Err(anyhow!(
            "{} steps for {} coordinates: give one step per coordinate",
            steps.len(),
            x.len()
        ));
    }
    for (i, (&xi, &hi)) in x.iter().zip(steps).enumerate() {
        if !hi.is_finite() || hi <= F::zero() {
            return Err(anyhow!(
                "step {i} is {}: a step must be finite and positive",
                shown(hi)
            ));
        }
        if xi + hi == xi || xi - hi == xi {
            return Err(anyhow!(
                "step {i} is {}: too small to change coordinate {i} ({})",
                shown(hi),
                shown(xi)
            ));
        }
    }
    Ok(())
}

/// The central second differences of `f` at `x` with `steps`, given
/// `fx = f(x)`, and for each entry the largest `|f|` of the evaluations it
/// uses. Evaluates `f` `2p²` times.
fn central_second_differences<F>(
    x: &[F],
    fx: F,
    f: CostFn<'_, F>,
    steps: &[F],
) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>), Error>
where
    F: Float + FromPrimitive,
{
    let n = x.len();
    let two = F::from_f64(2.0).unwrap();
    let four = F::from_f64(4.0).unwrap();
    let mut xt = x.to_vec();
    let mut hessian = vec![vec![F::zero(); n]; n];
    let mut size = vec![vec![F::zero(); n]; n];
    for i in 0..n {
        let plus = mod_and_calc(&mut xt, f, i, steps[i])?;
        let minus = mod_and_calc(&mut xt, f, i, -steps[i])?;
        hessian[i][i] = (plus - two * fx + minus) / (steps[i] * steps[i]);
        size[i][i] = fx.abs().max(plus.abs()).max(minus.abs());
    }
    for i in 0..n {
        for j in 0..i {
            let mut corner = |si: F, sj: F| {
                xt[i] = x[i] + si * steps[i];
                xt[j] = x[j] + sj * steps[j];
                let value = (f)(&xt);
                xt[i] = x[i];
                xt[j] = x[j];
                value
            };
            let pp = corner(F::one(), F::one())?;
            let pm = corner(F::one(), -F::one())?;
            let mp = corner(-F::one(), F::one())?;
            let mm = corner(-F::one(), -F::one())?;
            let entry = (pp - pm - mp + mm) / (four * steps[i] * steps[j]);
            let largest = pp.abs().max(pm.abs()).max(mp.abs()).max(mm.abs());
            hessian[i][j] = entry;
            hessian[j][i] = entry;
            size[i][j] = largest;
            size[j][i] = largest;
        }
    }
    Ok((hessian, size))
}

/// The Hessian of a cost with no gradient, by central differences with the
/// step `h_i` of each coordinate given by the caller.
///
/// The stencils are
///
/// ```text
/// H_ii = (f(x + h_i e_i) − 2 f(x) + f(x − h_i e_i)) / h_i²
/// H_ij = (f(x + h_i e_i + h_j e_j) − f(x + h_i e_i − h_j e_j)
///         − f(x − h_i e_i + h_j e_j) + f(x − h_i e_i − h_j e_j)) / (4 h_i h_j)
/// ```
///
/// Each entry is computed once, so the Hessian is symmetric by construction.
/// The truncation error of each entry is of order `h²`; an error `δ|f|` in
/// each value of the cost adds up to `4δ|f| / h_i²` on the diagonal and
/// `δ|f| / (h_i h_j)` off it. [`central_hessian_nograd_error_vec`] estimates
/// both, and explains how to choose the steps.
///
/// Cost: `2p² + 1` evaluations of `f` for `p` coordinates (`f(x)`, `2p`
/// on the diagonal, `4` for each of the `p(p − 1)/2` pairs), reported in
/// [`CentralHessian::evaluations`].
///
/// # Errors
///
/// When the number of steps differs from the number of coordinates, when a
/// step is not finite or not positive, or when `x_i ± h_i` equals `x_i`; and
/// any error of `f`.
pub fn central_hessian_nograd_steps_vec<F>(
    x: &[F],
    f: CostFn<'_, F>,
    steps: &[F],
) -> Result<CentralHessian<F>, Error>
where
    F: Float + FromPrimitive,
{
    check_steps(x, steps)?;
    let fx = (f)(&x.to_vec())?;
    let (hessian, _) = central_second_differences(x, fx, f, steps)?;
    let p = x.len();
    Ok(CentralHessian {
        hessian,
        evaluations: 2 * p * p + 1,
    })
}

/// The Hessian of a cost with no gradient, by central differences with the
/// step `h_i` of each coordinate given by the caller, and an estimate of the
/// error of each entry.
///
/// The Hessian `H(h)` uses the stencils of [`central_hessian_nograd_steps_vec`]
/// with the steps `h`. The same stencils with the steps `2h` give `H(2h)`, and
/// for each entry:
///
/// - the truncation error is `|H(h) − H(2h)| / 3` (Richardson): the error
///   of a central second difference is `c h² + O(h⁴)`, so
///   `H(h) − H(2h) ≈ −3 c h²`;
/// - the rounding error is at most `δ · Σ|c| · max|f|` when each value of the
///   cost has a relative error of at most `δ` (`accuracy`): `Σ|c|` is
///   `4 / h_i²` on the diagonal and `1 / (h_i h_j)` off it, and `max|f|` is
///   the largest `|f|` of the evaluations of the entry.
///
/// The sum, [`CentralHessianEstimate::error`], estimates `|H(h) − H|`. The
/// rounding term is a bound; the truncation term is an asymptotic estimate,
/// which can fall short of the true truncation error by a relative `O(h²)`
/// (the `h⁴` term of the error). The returned Hessian is `H(h)`, not the
/// extrapolation `(4H(h) − H(2h)) / 3`, which would amplify the noise. An
/// entry is reliable when its error is small against its value; an error
/// dominated by `rounding` says the steps are too small for the accuracy of
/// the cost, one dominated by `truncation` that they are too large.
///
/// # Choosing the steps
///
/// For a cost of relative accuracy `δ` (an ODE solved to the relative
/// tolerance `δ`, or `δ = ε` for a cost exact to rounding), the step
/// `h_i = δ^(1/4) · s_i`, with `s_i` the scale of coordinate `i` (for
/// example `max(|x_i|, 1)`), balances the truncation error, of order `h²`,
/// with the rounding error, of order `δ / h²`. With `δ = 1e-6` that is
/// `h_i ≈ 0.03 s_i`; a step of `√ε` gives an error of order `δ / ε` and a
/// useless Hessian. The steps `2h_i` must also be valid points of the cost.
///
/// Cost: `4p² + 1` evaluations of `f` for `p` coordinates (`f(x)` and
/// `2p²` for each of `h` and `2h`), reported in
/// [`CentralHessianEstimate::evaluations`].
///
/// # Errors
///
/// When the number of steps differs from the number of coordinates, when a
/// step or its double is not finite, when a step is not positive, when
/// `x_i ± h_i` equals `x_i`, or when `accuracy` is negative or not finite;
/// and any error of `f`.
pub fn central_hessian_nograd_error_vec<F>(
    x: &[F],
    f: CostFn<'_, F>,
    steps: &[F],
    accuracy: F,
) -> Result<CentralHessianEstimate<F>, Error>
where
    F: Float + FromPrimitive,
{
    check_steps(x, steps)?;
    let two = F::from_f64(2.0).unwrap();
    let doubled: Vec<F> = steps.iter().map(|&h| two * h).collect();
    if let Some(i) = doubled.iter().position(|h| !h.is_finite()) {
        return Err(anyhow!(
            "step {i} is {}: twice the step must be finite",
            shown(steps[i])
        ));
    }
    if !accuracy.is_finite() || accuracy < F::zero() {
        return Err(anyhow!(
            "the accuracy is {}: it must be finite and not negative",
            shown(accuracy)
        ));
    }
    let fx = (f)(&x.to_vec())?;
    let (hessian, size) = central_second_differences(x, fx, f, steps)?;
    let (coarse, _) = central_second_differences(x, fx, f, &doubled)?;
    let n = x.len();
    let three = F::from_f64(3.0).unwrap();
    let four = F::from_f64(4.0).unwrap();
    let mut truncation = vec![vec![F::zero(); n]; n];
    let mut rounding = vec![vec![F::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            truncation[i][j] = (hessian[i][j] - coarse[i][j]).abs() / three;
            let stencil = if i == j { four } else { F::one() };
            rounding[i][j] = accuracy * stencil * size[i][j] / (steps[i] * steps[j]);
        }
    }
    Ok(CentralHessianEstimate {
        hessian,
        truncation,
        rounding,
        evaluations: 4 * n * n + 1,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Vec<f64>) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
    }

    fn g(x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
    }

    fn x() -> Vec<f64> {
        vec![1.0f64, 1.0, 1.0, 1.0]
    }

    fn p() -> Vec<f64> {
        vec![2.0, 3.0, 4.0, 5.0]
    }

    fn res1() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ]
    }

    fn res2() -> Vec<f64> {
        vec![0.0, 6.0, 10.0, 18.0]
    }

    #[test]
    fn test_forward_hessian_vec_f64() {
        let hessian = forward_hessian_vec(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_vec_f64() {
        let hessian = central_hessian_vec(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_vec_prod_vec_f64() {
        let hessian = forward_hessian_vec_prod_vec(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64() {
        let hessian = central_hessian_vec_prod_vec(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64() {
        let hessian = forward_hessian_nograd_vec(&x(), &f).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_vec_f64() {
        let indices = vec![[1, 1], [2, 3], [3, 3]];
        let hessian = forward_hessian_nograd_sparse_vec(&x(), &f, indices).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    /// A cost that counts how often it is evaluated.
    fn counted<'a>(
        f: &'a dyn Fn(&[f64]) -> f64,
        count: &'a Cell<usize>,
    ) -> impl Fn(&Vec<f64>) -> Result<f64, Error> + 'a {
        move |x| {
            count.set(count.get() + 1);
            Ok(f(x))
        }
    }

    fn quadratic_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, -1.0, 0.5],
            vec![-1.0, 3.0, 2.0],
            vec![0.5, 2.0, 5.0],
        ]
    }

    /// `½xᵀAx + bᵀx` with the matrix of [`quadratic_matrix`].
    fn quadratic(x: &[f64]) -> f64 {
        let a = quadratic_matrix();
        let b = [1.0, -2.0, 0.5];
        let mut value = 0.0;
        for i in 0..3 {
            value += b[i] * x[i];
            for j in 0..3 {
                value += 0.5 * x[i] * a[i][j] * x[j];
            }
        }
        value
    }

    fn exp_product(x: &[f64]) -> f64 {
        (x[0] * x[1]).exp()
    }

    /// The exact Hessian of [`exp_product`].
    fn exp_product_hessian(x: &[f64]) -> Vec<Vec<f64>> {
        let e = (x[0] * x[1]).exp();
        let cross = (1.0 + x[0] * x[1]) * e;
        vec![vec![x[1] * x[1] * e, cross], vec![cross, x[0] * x[0] * e]]
    }

    /// A deterministic value in `[-1, 1]` for each point, as the error of an
    /// ODE solved to a tolerance is a function of the point.
    fn noise(x: &[f64]) -> f64 {
        let mut z = x.iter().fold(0x9E37_79B9_7F4A_7C15u64, |hash, value| {
            (hash ^ value.to_bits())
                .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                .rotate_left(31)
        });
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    #[test]
    fn central_hessian_with_steps_is_exact_for_a_quadratic() {
        let x = vec![0.3, -1.2, 2.0];
        let a = quadratic_matrix();
        for steps in [
            vec![1e-3, 1e-3, 1e-3],
            vec![1e-2, 0.5, 2.0],
            vec![1.0, 1e-3, 0.1],
        ] {
            let count = Cell::new(0);
            let f = counted(&quadratic, &count);
            let out = central_hessian_nograd_steps_vec(&x, &f, &steps).unwrap();
            assert_eq!(out.evaluations, 2 * 3 * 3 + 1);
            assert_eq!(count.get(), out.evaluations);
            for i in 0..3 {
                for j in 0..3 {
                    let tolerance = 100.0 * f64::EPSILON * 20.0 / (steps[i] * steps[j]);
                    assert!(
                        (out.hessian[i][j] - a[i][j]).abs() <= tolerance,
                        "entry ({i}, {j}) with steps {steps:?}: {} against {}",
                        out.hessian[i][j],
                        a[i][j]
                    );
                    assert_eq!(out.hessian[i][j], out.hessian[j][i]);
                }
            }
        }
    }

    #[test]
    fn central_hessian_with_steps_rejects_bad_steps() {
        let x = vec![1.0, 2.0];
        let f = |x: &Vec<f64>| Ok(exp_product(x));

        let error = central_hessian_nograd_steps_vec(&x, &f, &[1e-3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("1 steps for 2 coordinates"), "{error}");

        for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY, 1e-17] {
            let error = central_hessian_nograd_steps_vec(&x, &f, &[1e-3, bad])
                .unwrap_err()
                .to_string();
            assert!(error.contains("step 1"), "{error}");
            let error = central_hessian_nograd_error_vec(&x, &f, &[1e-3, bad], 1e-6)
                .unwrap_err()
                .to_string();
            assert!(error.contains("step 1"), "{error}");
        }

        // The step is finite, twice the step is not.
        let error = central_hessian_nograd_error_vec(&x, &f, &[f64::MAX, 1e-3], 1e-6)
            .unwrap_err()
            .to_string();
        assert!(error.contains("step 0"), "{error}");

        for bad in [-1e-6, f64::NAN, f64::INFINITY] {
            let error = central_hessian_nograd_error_vec(&x, &f, &[1e-3, 1e-3], bad)
                .unwrap_err()
                .to_string();
            assert!(error.contains("accuracy"), "{error}");
        }
    }

    #[test]
    fn error_estimate_bounds_the_error_for_exp_product() {
        let x = vec![-0.7, 1.3];
        let exact = exp_product_hessian(&x);
        // ε^(1/4) ≈ 1.2e-4 is the step for a cost exact to rounding.
        for factor in [1e-4, 1e-3, 1e-2, 1e-1] {
            let steps: Vec<f64> = x.iter().map(|v| factor * v.abs().max(1.0)).collect();
            let count = Cell::new(0);
            let f = counted(&exp_product, &count);
            let out = central_hessian_nograd_error_vec(&x, &f, &steps, f64::EPSILON).unwrap();
            assert_eq!(out.evaluations, 4 * 2 * 2 + 1);
            assert_eq!(count.get(), out.evaluations);
            let error = out.error();
            for i in 0..2 {
                for j in 0..2 {
                    let actual = (out.hessian[i][j] - exact[i][j]).abs();
                    let message = format!(
                        "entry ({i}, {j}) with steps {steps:?}: error {actual:e}, estimate {:e}",
                        error[i][j]
                    );
                    if factor <= 1e-3 {
                        assert!(actual <= error[i][j], "{message}");
                    } else {
                        // Truncation dominates, and its estimate is asymptotic:
                        // close to the error, short of it by a relative O(h²) at most.
                        assert!((actual - error[i][j]).abs() <= 0.01 * actual, "{message}");
                    }
                    assert_eq!(error[i][j], error[j][i]);
                }
            }
        }
    }

    #[test]
    fn error_estimate_tells_good_steps_from_noisy_ones() {
        let delta = 1e-6;
        let noisy = |x: &Vec<f64>| Ok(exp_product(x) * (1.0 + delta * noise(x)));
        let x = vec![1.2, 0.9];
        let exact = exp_product_hessian(&x);

        // Steps that balance truncation and noise: δ^(1/4) times the scale.
        let steps: Vec<f64> = x
            .iter()
            .map(|v| delta.powf(0.25) * v.abs().max(1.0))
            .collect();
        let good = central_hessian_nograd_error_vec(&x, &noisy, &steps, delta).unwrap();
        let error = good.error();
        for i in 0..2 {
            for j in 0..2 {
                let actual = (good.hessian[i][j] - exact[i][j]).abs();
                println!(
                    "δ^(1/4) steps, entry ({i}, {j}): exact {:.6}, error {actual:.3e}, \
                     estimate {:.3e} (truncation {:.3e}, rounding {:.3e})",
                    exact[i][j], error[i][j], good.truncation[i][j], good.rounding[i][j]
                );
                assert!(actual <= error[i][j]);
                assert!(error[i][j] <= 0.01 * exact[i][j].abs());
            }
        }

        // Steps of √ε, as for a cost exact to rounding.
        let steps = vec![f64::EPSILON.sqrt(); 2];
        let bad = central_hessian_nograd_error_vec(&x, &noisy, &steps, delta).unwrap();
        let error = bad.error();
        for i in 0..2 {
            for j in 0..2 {
                let actual = (bad.hessian[i][j] - exact[i][j]).abs();
                println!(
                    "√ε steps, entry ({i}, {j}): exact {:.6}, error {actual:.3e}, \
                     estimate {:.3e} (truncation {:.3e}, rounding {:.3e})",
                    exact[i][j], error[i][j], bad.truncation[i][j], bad.rounding[i][j]
                );
                assert!(actual > exact[i][j].abs());
                assert!(error[i][j] > exact[i][j].abs());
                assert!(actual <= error[i][j]);
            }
        }
    }
}
