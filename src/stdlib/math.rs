use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::span::Span;

use super::RNG_STATE;
pub fn math_seed(seed: i64) {
    RNG_STATE.with(|rng| {
        *rng.borrow_mut() = StdRng::seed_from_u64(seed as u64);
    });
}

pub fn math_rand_float() -> f64 {
    RNG_STATE.with(|rng| rng.borrow_mut().gen_range(0.0..1.0))
}

pub fn math_rand_bool() -> bool {
    RNG_STATE.with(|rng| rng.borrow_mut().gen_bool(0.5))
}

pub fn math_rand_int(min: i64, max: i64, _span: Span) -> WalrusResult<i64> {
    if min > max {
        return Err(WalrusError::GenericError {
            message: format!("math.rand_int: invalid range [{min}, {max}]"),
        });
    }

    Ok(RNG_STATE.with(|rng| rng.borrow_mut().gen_range(min..=max)))
}

pub fn math_rand_range(min: f64, max: f64, _span: Span) -> WalrusResult<f64> {
    if !min.is_finite() || !max.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.rand_range: range bounds must be finite numbers".to_string(),
        });
    }

    if min > max {
        return Err(WalrusError::GenericError {
            message: format!("math.rand_range: invalid range [{min}, {max}]"),
        });
    }

    if (min - max).abs() < f64::EPSILON {
        return Ok(min);
    }

    Ok(RNG_STATE.with(|rng| rng.borrow_mut().gen_range(min..max)))
}
