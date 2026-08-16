//! Morlet wavelet filter banks and Gaussian lowpass filters, specified
//! directly in the frequency domain.
//!
//! # Frequency convention
//!
//! Everything internal to this module is in *normalised* frequency, in cycles
//! per sample, on the interval (-0.5, 0.5]. Multiply by the sample rate to get
//! Hz. The DFT convention is `X(w) = sum_t x[t] exp(-i 2 pi w t)`.
//!
//! # Symbols
//!
//! - `xi` centre frequency of a wavelet (cycles/sample). Signed: negative
//!   values are meaningful for complex baseband input, where they mean "below
//!   the local oscillator".
//! - `sigma` standard deviation of the wavelet's frequency-domain Gaussian
//!   envelope (cycles/sample).
//! - `Q` wavelets per octave in the constant-Q regime. Bandwidth is not set
//!   independently: it follows from `Q` via the overlap criterion below.
//! - `r` the amplitude at which adjacent filters cross. `sqrt(1/2)`, half
//!   power, makes the Littlewood-Paley sum flat at one.
//! - `phi` the lowpass averaging filter that sets the invariance scale.
//!
//! # Two regimes
//!
//! A purely constant-Q bank cannot reach low frequencies in a finite block:
//! time support grows as `1/xi` without bound. Following Anden and Mallat
//! (2014), the bank is constant-Q only while bandwidth stays above a floor
//! `sigma_min`, and below that elbow it becomes constant-*bandwidth*: `sigma`
//! is pinned and centre frequencies step down linearly instead of
//! geometrically. Time support is then bounded by construction, and coverage
//! continues to near DC at fixed cost per filter.
//!
//! The price is that `Q` falls with frequency below the elbow, so resolution
//! bandwidth stops shrinking. That is a real change in what the analysis
//! reports, which is why [`Wavelet::bandwidth`] is carried per filter rather
//! than inferred from `Q`.

use std::f64::consts::PI;

/// Converts a Gaussian standard deviation into the full -3 dB bandwidth of the
/// frequency response.
///
/// For `exp(-(w - xi)^2 / (2 sigma^2))` the amplitude falls to `1/sqrt(2)` at
/// `w - xi = +/- sigma sqrt(ln 2)`, so the full -3 dB width is
/// `2 sqrt(ln 2) sigma ~= 1.6651 sigma`.
fn sigma_to_bw(sigma: f64) -> f64 {
    2.0 * std::f64::consts::LN_2.sqrt() * sigma
}

/// Number of standard deviations of Gaussian tail we bother to periodise.
/// Beyond 6 sigma the contribution is below 1e-8 of the peak.
const PERIODISE_ORDER: i32 = 1;

/// Unnormalised Morlet frequency response, evaluated on the real line.
///
/// A Gabor atom minus a corrective Gaussian at DC, which forces `psi_hat(0) = 0`
/// so the wavelet has zero mean and is admissible.
fn morlet_hat(w: f64, xi: f64, sigma: f64) -> f64 {
    let gabor = (-(w - xi).powi(2) / (2.0 * sigma * sigma)).exp();
    let kappa = (-xi * xi / (2.0 * sigma * sigma)).exp();
    let corrective = (-w * w / (2.0 * sigma * sigma)).exp();
    gabor - kappa * corrective
}

/// Gaussian lowpass frequency response, unnormalised (peak 1 at DC).
fn gauss_hat(w: f64, sigma: f64) -> f64 {
    (-w * w / (2.0 * sigma * sigma)).exp()
}

/// Maps DFT bin index `k` of an `n`-point transform to signed normalised
/// frequency in (-0.5, 0.5].
pub fn bin_freq(k: usize, n: usize) -> f64 {
    let k = k as f64;
    let n = n as f64;
    if k * 2.0 <= n {
        k / n
    } else {
        (k - n) / n
    }
}

/// Samples a frequency response onto the `n`-point DFT grid, summing the
/// periodic images so that energy wrapping past +/-0.5 is accounted for rather
/// than silently truncated.
fn periodise<F: Fn(f64) -> f64>(n: usize, f: F) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let w = bin_freq(k, n);
            (-PERIODISE_ORDER..=PERIODISE_ORDER)
                .map(|m| f(w + m as f64))
                .sum()
        })
        .collect()
}

/// One Morlet band-pass filter, sampled on a fixed-length DFT grid.
#[derive(Clone, Debug)]
pub struct Wavelet {
    /// Centre frequency, cycles/sample. Signed.
    pub xi: f64,
    /// Frequency-domain Gaussian standard deviation, cycles/sample.
    pub sigma: f64,
    /// Full -3 dB bandwidth, cycles/sample.
    pub bandwidth: f64,
    /// Frequency response on the DFT grid, normalised to unit peak.
    ///
    /// Unit peak gain is deliberate: it makes `|x * psi_lambda|` equal the
    /// amplitude of a tone sitting at `lambda`, so first-order coefficients
    /// come out in the same units as the input (volts, if the input is volts).
    pub spectrum: Vec<f64>,
}

impl Wavelet {
    /// Effective time support in samples, taken as +/-3 standard deviations of
    /// the time-domain Gaussian envelope. Used to work out how much of the
    /// output is contaminated by circular-convolution wraparound.
    pub fn time_support(&self) -> f64 {
        // Fourier pair: a frequency-domain Gaussian of std `sigma` corresponds
        // to a time-domain Gaussian of std `1 / (2 pi sigma)`.
        6.0 / (2.0 * PI * self.sigma)
    }
}

/// A geometrically spaced bank of Morlet wavelets plus its lowpass companion.
#[derive(Clone, Debug)]
pub struct FilterBank {
    /// DFT length these filters are sampled on.
    pub n: usize,
    pub wavelets: Vec<Wavelet>,
    /// Lowpass averaging filter `phi`, unit gain at DC.
    pub phi: Vec<f64>,
    /// Frequency-domain standard deviation of `phi`, cycles/sample.
    pub sigma_phi: f64,
    /// Bandwidth floor imposed by the block length, cycles/sample.
    pub sigma_min: f64,
    /// Centre frequency where constant-Q gave way to constant bandwidth, if the
    /// bank reached that far down.
    pub elbow_xi: Option<f64>,
}

/// Amplitude at which adjacent filters cross.
///
/// `sqrt(1/2)` is half power, so two neighbours contribute half each and their
/// squared magnitudes sum to one: the bank tiles the frequency axis flat.
pub const DEFAULT_R: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// How a bank should cover the frequency axis.
#[derive(Clone, Copy, Debug)]
pub struct BankSpec {
    /// Wavelets per octave in the constant-Q regime.
    ///
    /// Bandwidth is derived from this, not set separately: see
    /// [`BankSpec::sigma_for`].
    pub q: f64,
    /// Amplitude at which adjacent filters cross. Use [`DEFAULT_R`].
    pub r: f64,
    /// Highest centre frequency, cycles/sample. Should leave headroom below
    /// the 0.5 Nyquist limit; 0.4 is a reasonable default.
    pub xi_max: f64,
    /// Number of octaves spanned below `xi_max`.
    pub octaves: f64,
    /// Include mirrored negative centre frequencies.
    ///
    /// Set this for complex baseband (I/Q) input, where negative baseband
    /// frequency means "below the local oscillator" and carries real signal.
    /// Leave it off for real-valued input, where the analytic (positive-only)
    /// half is sufficient and the negative half is redundant.
    pub two_sided: bool,
}

impl BankSpec {
    /// Bandwidth of a constant-Q wavelet at centre frequency `xi`.
    ///
    /// Derived rather than chosen. Adjacent filters sit at `xi` and
    /// `f * xi` with `f = 2^(-1/Q)`, and in the constant-Q regime their widths
    /// scale with their centres, so they cross where
    /// `(w - xi) / sigma = -(w - f xi) / (f sigma)`, that is at
    /// `w = 2 f xi / (1 + f)`. Requiring the amplitude there to equal `r`:
    ///
    /// ```text
    /// exp(-(xi (1 - f) / (1 + f))^2 / (2 sigma^2)) = r
    /// sigma = xi (1 - f) / ((1 + f) sqrt(2 ln(1/r)))
    /// ```
    ///
    /// At `r = sqrt(1/2)` each of the two filters contributes half power at the
    /// crossing, so their squared magnitudes sum to one and the bank tiles the
    /// frequency axis without ripple.
    pub fn sigma_for(&self, xi: f64) -> f64 {
        let f = 2f64.powf(-1.0 / self.q.max(1.0));
        xi * ((1.0 - f) / (1.0 + f)) / (2.0 * (1.0 / self.r).ln()).sqrt()
    }

    /// Spacing between equal-width filters that also cross at amplitude `r`.
    ///
    /// Two Gaussians of the same `sigma` a distance `d` apart cross at their
    /// midpoint, so `exp(-(d/2)^2 / (2 sigma^2)) = r` gives
    /// `d = 2 sigma sqrt(2 ln(1/r))`. At `r = sqrt(1/2)` that is
    /// `1.6651 sigma`, exactly one -3 dB bandwidth: below the elbow the filters
    /// sit one bandwidth apart.
    pub fn linear_step(&self, sigma: f64) -> f64 {
        2.0 * sigma * (2.0 * (1.0 / self.r).ln()).sqrt()
    }

    /// The frequency plan: `(xi, sigma)` pairs, highest first, positive half
    /// only.
    ///
    /// Constant-Q while the bandwidth stays above `sigma_min`, constant
    /// bandwidth below it. Stops at whichever comes first: the requested span
    /// of `octaves`, or the point where a filter would sit so close to DC that
    /// the Morlet's zero-mean correction dominates it.
    pub fn plan(&self, sigma_min: f64) -> Vec<(f64, f64)> {
        let f = 2f64.powf(-1.0 / self.q.max(1.0));
        let xi_floor = self.xi_max * 2f64.powf(-self.octaves);
        let mut plan = Vec::new();

        let mut xi = self.xi_max;
        let mut sigma = self.sigma_for(xi);

        // Constant-Q regime: both centre and width shrink geometrically, so
        // xi/sigma stays fixed and the wavelets are dilations of one another.
        while sigma > sigma_min && xi >= xi_floor {
            plan.push((xi, sigma));
            xi *= f;
            sigma *= f;
        }

        // Constant-bandwidth regime: width pinned, centres stepping down
        // linearly. This is what bounds time support.
        let step = self.linear_step(sigma_min);
        // Keep clear of DC. At 1.5 steps the Gabor's value at zero frequency is
        // exp(-3.1) ~ 4%, so the zero-mean correction is a small perturbation
        // rather than the dominant term.
        let dc_guard = 1.5 * step;
        while xi >= xi_floor && xi >= dc_guard {
            plan.push((xi, sigma_min));
            xi -= step;
        }

        plan
    }

    /// Centre frequency of the elbow: the lowest constant-Q filter, below which
    /// the bank switches to constant bandwidth. `None` if the block is long
    /// enough that the requested span never reaches the elbow, or so short that
    /// it starts past it.
    pub fn elbow_xi(&self, sigma_min: f64) -> Option<f64> {
        let plan = self.plan(sigma_min);
        plan.iter()
            .take_while(|(_, s)| *s > sigma_min)
            .last()
            .map(|(xi, _)| *xi)
            .filter(|_| plan.iter().any(|(_, s)| *s <= sigma_min))
    }
}

/// Gaussian lowpass on an `n`-point grid, unit gain at DC.
///
/// Exposed because sub-band processing needs the same physical filter sampled
/// on several grid lengths: a band extracted at a reduced rate spans the same
/// time, so its `sigma` scales by the length ratio.
pub fn lowpass(n: usize, sigma: f64) -> Vec<f64> {
    periodise(n, |w| gauss_hat(w, sigma))
}

/// Fraction of the block a single wavelet is allowed to occupy.
///
/// A filter as long as the block would technically "fit", but its output is
/// contaminated by circular wraparound at both ends and nothing clean is left
/// in the middle. Capping support at a quarter of the block leaves half the
/// output usable after trimming.
const MAX_SUPPORT_FRACTION: f64 = 0.25;

/// Bandwidth floor for an `n`-point block, in cycles/sample.
///
/// Time support is `6 / (2 pi sigma)` samples, so holding it at or below
/// `MAX_SUPPORT_FRACTION * n` bounds `sigma` from below. This is the elbow: no
/// filter in the bank is narrower than this, whatever its centre frequency, and
/// therefore none is longer than a quarter of the block.
pub fn sigma_min_for_length(n: usize) -> f64 {
    6.0 / (2.0 * PI * MAX_SUPPORT_FRACTION * n as f64)
}

/// Lowest centre frequency an `n`-point block can carry, in cycles/sample.
///
/// Below the elbow the filters are constant-bandwidth, so the floor is set by
/// staying clear of DC rather than by `Q`. That is the whole point of the
/// two-regime construction: **the lowest frequency reachable no longer depends
/// on `Q`**, and halving it costs twice the block rather than `2 Q` times it.
pub fn min_xi_for_length(n: usize) -> f64 {
    let sigma_min = sigma_min_for_length(n);
    // 1.5 linear steps at r = sqrt(1/2), matching the DC guard in `plan`.
    1.5 * 2.0 * sigma_min * std::f64::consts::LN_2.sqrt()
}

impl FilterBank {
    /// Builds a bank on an `n`-point grid.
    ///
    /// `invariance_samples` is `T`, the width of the averaging window in
    /// samples. It sets the time resolution of the output: coefficients are
    /// smooth on scales shorter than `T` and can be decimated accordingly.
    ///
    /// No wavelet is narrower than [`sigma_min_for_length`], so none is longer
    /// than a quarter of the block. Coverage below the elbow is bought by
    /// giving up constant `Q` rather than by giving up the frequencies, which
    /// is the trade Anden and Mallat make.
    pub fn new(spec: BankSpec, n: usize, invariance_samples: f64) -> Self {
        let sigma_min = sigma_min_for_length(n);
        let plan = spec.plan(sigma_min);
        assert!(
            !plan.is_empty(),
            "no wavelet fits in a {n}-sample block below xi_max={}; lengthen the block",
            spec.xi_max
        );

        let mut wavelets = Vec::with_capacity(plan.len() * if spec.two_sided { 2 } else { 1 });
        for (xi, sigma) in plan {
            wavelets.push(Self::make_wavelet(xi, sigma, n));
            if spec.two_sided {
                wavelets.push(Self::make_wavelet(-xi, sigma, n));
            }
        }
        // Order by signed centre frequency so plotting code gets a monotonic
        // axis without having to sort.
        wavelets.sort_by(|a, b| a.xi.partial_cmp(&b.xi).unwrap());

        // A time-domain Gaussian of standard deviation `T/2` has a
        // frequency-domain standard deviation of `1 / (2 pi (T/2))`.
        let sigma_phi = 1.0 / (PI * invariance_samples);
        let phi = periodise(n, |w| gauss_hat(w, sigma_phi));

        FilterBank {
            n,
            wavelets,
            phi,
            sigma_phi,
            sigma_min,
            elbow_xi: spec.elbow_xi(sigma_min),
        }
    }

    /// The wavelet whose centre frequency is nearest `xi`, or `None` for an
    /// empty bank.
    ///
    /// Used to answer "what is the resolution bandwidth here?" honestly: with
    /// two regimes the answer is no longer `xi / Q` everywhere, so it has to be
    /// read off the filter that actually does the measuring.
    pub fn nearest(&self, xi: f64) -> Option<&Wavelet> {
        self.wavelets.iter().min_by(|a, b| {
            (a.xi - xi)
                .abs()
                .partial_cmp(&(b.xi - xi).abs())
                .unwrap()
        })
    }

    /// Lowest centre frequency actually built, cycles/sample.
    pub fn min_xi(&self) -> f64 {
        self.wavelets
            .iter()
            .map(|w| w.xi.abs())
            .fold(f64::INFINITY, f64::min)
    }

    fn make_wavelet(xi: f64, sigma: f64, n: usize) -> Wavelet {
        let bandwidth = sigma_to_bw(sigma);
        let mut spectrum = periodise(n, |w| morlet_hat(w, xi, sigma));
        // Normalise to unit peak so first-order coefficients read in input
        // amplitude units. `fold` rather than `max` because f64 is not Ord.
        let peak = spectrum.iter().fold(0f64, |acc, v| acc.max(v.abs()));
        if peak > 0.0 {
            for v in spectrum.iter_mut() {
                *v /= peak;
            }
        }
        Wavelet { xi, sigma, bandwidth, spectrum }
    }

    /// Longest wavelet time support in the bank, in samples. Output within this
    /// distance of either edge is contaminated by circular wraparound.
    pub fn max_time_support(&self) -> f64 {
        self.wavelets
            .iter()
            .map(|w| w.time_support())
            .fold(0f64, f64::max)
    }

    /// Littlewood-Paley sum: `sum_lambda |psi_hat_lambda(w)|^2 + |phi_hat(w)|^2`
    /// evaluated on the DFT grid.
    ///
    /// A well-formed bank has this roughly flat across the band it covers.
    /// Large ripple means the analysis is unevenly sensitive to frequency, and
    /// a spectrum analyser display built on it would show phantom structure.
    pub fn littlewood_paley(&self) -> Vec<f64> {
        let mut sum: Vec<f64> = self.phi.iter().map(|p| p * p).collect();
        for w in &self.wavelets {
            for (acc, v) in sum.iter_mut().zip(w.spectrum.iter()) {
                *acc += v * v;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(q: f64, octaves: f64) -> BankSpec {
        BankSpec { q, r: DEFAULT_R, xi_max: 0.4, octaves, two_sided: true }
    }

    #[test]
    fn morlet_has_zero_mean() {
        // Admissibility: the frequency response must vanish at DC, otherwise
        // the "wavelet" leaks a DC term into every coefficient.
        for &xi in &[0.4, 0.1, 0.01] {
            assert!(morlet_hat(0.0, xi, xi / 8.0).abs() < 1e-12);
        }
    }

    #[test]
    fn stated_bandwidth_is_the_real_minus_3_db_width() {
        // The bandwidth carried on each wavelet must be the actual -3 dB width,
        // since a display quotes it as the resolution.
        let (xi, sigma) = (0.2, 0.02);
        let bw = sigma_to_bw(sigma);
        let peak = morlet_hat(xi, xi, sigma);
        let edge = morlet_hat(xi + bw / 2.0, xi, sigma);
        let ratio_db = 20.0 * (edge / peak).log10();
        assert!((ratio_db + 3.0).abs() < 0.05, "edge was {ratio_db} dB");
    }

    #[test]
    fn adjacent_filters_cross_at_the_requested_amplitude() {
        // The whole construction rests on this: sigma is chosen so neighbours
        // meet at r. If they do not, the bank either ripples or wastes filters.
        let s = spec(8.0, 4.0);
        let plan = s.plan(sigma_min_for_length(1 << 16));
        let constant_q: Vec<_> = plan.iter().take_while(|(_, sg)| *sg > s.sigma_for(0.0).max(0.0)).collect();
        assert!(plan.len() > 4);
        let _ = constant_q;

        for pair in plan.windows(2) {
            let (xi_hi, sigma_hi) = pair[0];
            let (xi_lo, sigma_lo) = pair[1];
            // Crossing point of two Gaussians of differing width.
            let cross = (xi_hi * sigma_lo + xi_lo * sigma_hi) / (sigma_hi + sigma_lo);
            let a = (-(cross - xi_hi).powi(2) / (2.0 * sigma_hi * sigma_hi)).exp();
            let b = (-(cross - xi_lo).powi(2) / (2.0 * sigma_lo * sigma_lo)).exp();
            assert!((a - b).abs() < 1e-9, "crossing point is not where both are equal");
            assert!(
                (a - DEFAULT_R).abs() < 1e-6,
                "neighbours at {xi_hi:.5} and {xi_lo:.5} cross at {a:.5}, wanted {DEFAULT_R:.5}"
            );
        }
    }

    #[test]
    fn unit_peak_gain() {
        let bank = FilterBank::new(spec(8.0, 4.0), 4096, 256.0);
        for w in &bank.wavelets {
            let peak = w.spectrum.iter().fold(0f64, |a, v| a.max(v.abs()));
            assert!((peak - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn two_sided_bank_is_symmetric() {
        let bank = FilterBank::new(spec(8.0, 3.0), 2048, 128.0);
        let positives: Vec<f64> = bank.wavelets.iter().filter(|w| w.xi > 0.0).map(|w| w.xi).collect();
        let negatives: Vec<f64> = bank.wavelets.iter().filter(|w| w.xi < 0.0).map(|w| -w.xi).collect();
        assert_eq!(positives.len(), negatives.len());
        for (p, n) in positives.iter().zip(negatives.iter().rev()) {
            assert!((p - n).abs() < 1e-12);
        }
    }

    #[test]
    fn littlewood_paley_is_flat_across_the_covered_band() {
        let s = spec(8.0, 5.0);
        let bank = FilterBank::new(s, 8192, 512.0);
        let lp = bank.littlewood_paley();

        // Only judge flatness where the bank is meant to cover: between the
        // lowest centre frequency and the highest. Outside that the response
        // rolls off by design.
        let lowest = bank.min_xi();
        let vals: Vec<f64> = (0..bank.n)
            .filter(|&k| {
                let w = bin_freq(k, bank.n).abs();
                w > lowest * 1.5 && w < s.xi_max * 0.95
            })
            .map(|k| lp[k])
            .collect();

        assert!(!vals.is_empty());
        let lo = vals.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        let hi = vals.iter().fold(0f64, |a, &v| a.max(v));
        let ripple_db = 10.0 * (hi / lo).log10();
        assert!(ripple_db < 1.0, "Littlewood-Paley ripple {ripple_db:.2} dB is too large");
    }

    /// The crossing criterion is chosen so the bank tiles flat *at one*. That is
    /// stronger than merely being flat, and it means first-order coefficients
    /// carry the signal's energy rather than an arbitrary multiple of it.
    #[test]
    fn littlewood_paley_sums_to_about_one() {
        let s = spec(8.0, 5.0);
        let bank = FilterBank::new(s, 8192, 512.0);
        let lp = bank.littlewood_paley();
        let lowest = bank.min_xi();

        let vals: Vec<f64> = (0..bank.n)
            .filter(|&k| {
                let w = bin_freq(k, bank.n).abs();
                w > lowest * 2.0 && w < s.xi_max * 0.9
            })
            .map(|k| lp[k])
            .collect();

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!((mean - 1.0).abs() < 0.1, "Littlewood-Paley sum averages {mean:.3}, wanted 1");
    }

    /// The point of the two-regime construction: no filter is longer than the
    /// block allows, however low its centre frequency goes.
    #[test]
    fn no_wavelet_outgrows_the_block() {
        for n in [1024usize, 4096, 65536] {
            for q in [1.0, 4.0, 8.0, 16.0] {
                let bank = FilterBank::new(spec(q, 12.0), n, n as f64 / 32.0);
                let longest = bank.max_time_support();
                assert!(
                    longest <= MAX_SUPPORT_FRACTION * n as f64 * 1.001,
                    "Q={q} in a {n}-sample block produced a {longest:.0}-sample filter"
                );
            }
        }
    }

    /// The reachable floor must not scale with Q any more. Under the old
    /// single-regime bank it was proportional to Q, which is exactly what made
    /// low modulation rates so expensive.
    ///
    /// It is not perfectly identical across Q, and should not be expected to
    /// be: the constant-bandwidth regime steps down by a fixed amount from
    /// wherever the constant-Q regime happened to stop, so the floor is
    /// quantised to within one step. What matters is that a 8x range of Q no
    /// longer produces an 8x range of floors.
    #[test]
    fn frequency_floor_does_not_scale_with_q() {
        let n = 1 << 16;
        let qs = [2.0, 4.0, 8.0, 16.0];
        let floors: Vec<f64> = qs
            .iter()
            .map(|&q| FilterBank::new(spec(q, 16.0), n, 256.0).min_xi())
            .collect();

        let guard = min_xi_for_length(n);
        let step = guard / 1.5; // the guard is 1.5 steps

        for (&q, &floor) in qs.iter().zip(floors.iter()) {
            assert!(
                floor >= guard * 0.999 && floor < guard + step,
                "Q={q} floor {floor:.4e} outside [{:.4e}, {:.4e})",
                guard,
                guard + step
            );
        }

        // The headline claim, stated as a ratio: Q spans 8x, the floor must not.
        let lo = floors.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        let hi = floors.iter().fold(0f64, |a, &v| a.max(v));
        assert!(hi / lo < 2.0, "floor still tracks Q: {floors:?}");
    }

    /// Above the elbow the bank is constant-Q, below it constant-bandwidth.
    /// Both halves must actually behave that way.
    #[test]
    fn the_two_regimes_do_what_they_say() {
        let n = 1 << 14;
        let s = spec(8.0, 12.0);
        let bank = FilterBank::new(s, n, 512.0);
        let elbow = bank.elbow_xi.expect("12 octaves in a short block must reach the elbow");

        let positives: Vec<&Wavelet> = bank.wavelets.iter().filter(|w| w.xi > 0.0).collect();

        for w in positives.iter().filter(|w| w.xi > elbow * 1.01) {
            // Constant-Q: bandwidth proportional to centre frequency.
            let q_here = w.xi / w.bandwidth;
            let q_top = positives.last().unwrap().xi / positives.last().unwrap().bandwidth;
            assert!(
                (q_here / q_top - 1.0).abs() < 1e-6,
                "above the elbow Q drifted: {q_here:.3} vs {q_top:.3}"
            );
        }

        for w in positives.iter().filter(|w| w.xi < elbow * 0.99) {
            // Constant bandwidth: pinned at the floor.
            assert!(
                (w.bandwidth - sigma_to_bw(bank.sigma_min)).abs() < 1e-12,
                "below the elbow bandwidth was not pinned"
            );
        }
    }
}
