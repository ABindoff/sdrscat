//! Thin wrapper over `rustfft` giving normalised inverse transforms and
//! frequency-domain decimation.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::Arc;

pub type Cf64 = Complex<f64>;

/// Caches forward and inverse plans keyed by transform length, so a streaming
/// caller pays the planning cost once rather than per block.
pub struct Plans {
    planner: FftPlanner<f64>,
    forward: HashMap<usize, Arc<dyn Fft<f64>>>,
    inverse: HashMap<usize, Arc<dyn Fft<f64>>>,
}

impl Plans {
    pub fn new() -> Self {
        Plans {
            planner: FftPlanner::new(),
            forward: HashMap::new(),
            inverse: HashMap::new(),
        }
    }

    /// In-place forward DFT, unnormalised (the usual `sum_t x[t] e^{-i2pi w t}`).
    pub fn forward(&mut self, buf: &mut [Cf64]) {
        let n = buf.len();
        let plan = match self.forward.get(&n) {
            Some(p) => p.clone(),
            None => {
                let p = self.planner.plan_fft_forward(n);
                self.forward.insert(n, p.clone());
                p
            }
        };
        plan.process(buf);
    }

    /// In-place inverse DFT, normalised by `1/n` so that
    /// `inverse(forward(x)) == x`.
    pub fn inverse(&mut self, buf: &mut [Cf64]) {
        let n = buf.len();
        let plan = match self.inverse.get(&n) {
            Some(p) => p.clone(),
            None => {
                let p = self.planner.plan_fft_inverse(n);
                self.inverse.insert(n, p.clone());
                p
            }
        };
        plan.process(buf);
        let scale = 1.0 / n as f64;
        for v in buf.iter_mut() {
            *v *= scale;
        }
    }
}

impl Default for Plans {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// One plan cache per thread.
    ///
    /// Planning is stateful and needs exclusive access, which would otherwise
    /// stop the per-band work from running in parallel. Keeping a cache per
    /// thread rather than per call means the planning cost is paid once per
    /// worker for the whole session, not once per block.
    static THREAD_PLANS: std::cell::RefCell<Plans> = std::cell::RefCell::new(Plans::new());
}

/// Runs `f` with this thread's plan cache.
///
/// Calls must not nest: the borrow is exclusive for the duration.
pub fn with_plans<R>(f: impl FnOnce(&mut Plans) -> R) -> R {
    THREAD_PLANS.with(|cell| f(&mut cell.borrow_mut()))
}

/// Decimates by `d` in the frequency domain: keep the `n/d` bins nearest DC and
/// discard the rest.
///
/// Truncation rather than the more common Fourier periodisation, because
/// truncation *is* an ideal brick-wall anti-alias filter, whereas periodisation
/// folds out-of-band content back onto the retained bins. For a modulus
/// envelope, which is what we ever decimate here, the discarded content is
/// genuinely small and the brick wall costs us nothing.
///
/// The `1/d` scaling makes the result equal to the time-domain signal sampled
/// every `d`th point, rather than `d` times it.
pub fn decimate_spectrum(spectrum: &[Cf64], d: usize) -> Vec<Cf64> {
    let n = spectrum.len();
    assert!(d >= 1 && n % d == 0, "decimation {d} must divide length {n}");
    if d == 1 {
        return spectrum.to_vec();
    }
    let m = n / d;
    let half = m / 2;
    let scale = 1.0 / d as f64;
    let mut out = vec![Cf64::new(0.0, 0.0); m];
    for k in 0..half {
        out[k] = spectrum[k] * scale;
    }
    // Negative frequencies live at the top of both arrays.
    for k in 1..=half {
        out[m - k] = spectrum[n - k] * scale;
    }
    out
}

/// Largest power of two that is at most `limit` and still divides `n`.
/// Returns 1 when no useful decimation is available.
pub fn power_of_two_factor(limit: f64, n: usize) -> usize {
    // Written this way rather than `limit < 1.0` so a NaN limit also lands here
    // instead of falling through to the loop.
    if !matches!(limit.partial_cmp(&1.0), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))
    {
        return 1;
    }
    let mut d = 1usize;
    while d * 2 <= limit as usize && n % (d * 2) == 0 {
        d *= 2;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_identity() {
        let mut plans = Plans::new();
        let original: Vec<Cf64> = (0..64)
            .map(|i| Cf64::new((i as f64 * 0.1).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let mut buf = original.clone();
        plans.forward(&mut buf);
        plans.inverse(&mut buf);
        for (a, b) in original.iter().zip(buf.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn decimation_matches_time_domain_subsampling() {
        // A tone well below the post-decimation Nyquist must survive
        // decimation-by-truncation exactly.
        let n = 256;
        let d = 4;
        let f = 3.0 / n as f64; // 3 cycles over the block, far below Nyquist/d
        let x: Vec<Cf64> = (0..n)
            .map(|t| Cf64::from_polar(1.0, 2.0 * std::f64::consts::PI * f * t as f64))
            .collect();

        let mut plans = Plans::new();
        let mut spectrum = x.clone();
        plans.forward(&mut spectrum);
        let mut small = decimate_spectrum(&spectrum, d);
        plans.inverse(&mut small);

        for (s, chunk) in small.iter().enumerate() {
            assert!((chunk - x[s * d]).norm() < 1e-10);
        }
    }

    #[test]
    fn factor_respects_both_limit_and_divisibility() {
        assert_eq!(power_of_two_factor(9.0, 1024), 8);
        assert_eq!(power_of_two_factor(0.5, 1024), 1);
        assert_eq!(power_of_two_factor(64.0, 96), 32);
    }
}
