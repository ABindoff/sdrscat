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
//! - `Q` quality factor, defined here as `xi / bw` where `bw` is the full
//!   -3 dB (half-power) bandwidth. Large Q means narrow filters and long time
//!   support.
//! - `phi` the lowpass averaging filter that sets the invariance scale.

use std::f64::consts::PI;

/// Converts a full -3 dB bandwidth into the Gaussian standard deviation of the
/// frequency-domain envelope.
///
/// For `exp(-(w - xi)^2 / (2 sigma^2))` the amplitude falls to `1/sqrt(2)` at
/// `w - xi = +/- sigma sqrt(ln 2)`, so the full -3 dB width is
/// `2 sqrt(ln 2) sigma ~= 1.6651 sigma`.
fn bw_to_sigma(bw: f64) -> f64 {
    bw / (2.0 * std::f64::consts::LN_2.sqrt())
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
    /// How many requested centre frequencies were too low to fit in the block
    /// and were therefore not built.
    pub dropped: usize,
}

/// How a bank should cover the frequency axis.
#[derive(Clone, Copy, Debug)]
pub struct BankSpec {
    /// Quality factor: centre frequency divided by -3 dB bandwidth.
    pub q: f64,
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
    /// Centre frequencies this spec would produce, highest first, positive half
    /// only. Geometric spacing at `q` wavelets per octave.
    pub fn centre_frequencies(&self) -> Vec<f64> {
        // Spacing the filters `q` per octave makes adjacent -3 dB skirts cross
        // at roughly half power, which is the usual constant-Q compromise
        // between coverage ripple and filter count.
        let per_octave = self.q.max(1.0).round() as usize;
        let count = (self.octaves * per_octave as f64).round() as usize;
        (0..count)
            .map(|j| self.xi_max * 2f64.powf(-(j as f64) / per_octave as f64))
            .collect()
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

/// Lowest centre frequency whose wavelet still fits inside an `n`-point block,
/// in cycles/sample.
///
/// Time support is `6 / (2 pi sigma)` and `sigma = xi / (2 sqrt(ln 2) Q)`, so
/// support works out at `1.5904 Q / xi` samples; requiring that to be at most
/// `MAX_SUPPORT_FRACTION * n` gives the bound below. In words: **halving the
/// lowest frequency you want to see doubles the block length you need**, and
/// raising `Q` costs block length in direct proportion. There is no way round
/// this; it is the uncertainty principle presenting its bill.
pub fn min_xi_for_length(q: f64, n: usize) -> f64 {
    let support_constant = 6.0 * (2.0 * std::f64::consts::LN_2.sqrt()) / (2.0 * PI);
    support_constant * q / (MAX_SUPPORT_FRACTION * n as f64)
}

impl FilterBank {
    /// Builds a bank on an `n`-point grid.
    ///
    /// `invariance_samples` is `T`, the width of the averaging window in
    /// samples. It sets the time resolution of the output: coefficients are
    /// smooth on scales shorter than `T` and can be decimated accordingly.
    ///
    /// Wavelets whose time support exceeds the block are dropped rather than
    /// built. A filter longer than the data does not measure a low frequency,
    /// it measures circular wraparound, and silently returning those
    /// coefficients would put convincing-looking rubbish at the bottom of the
    /// display. [`FilterBank::dropped`] reports how many were discarded so the
    /// caller can tell the user what coverage it actually got.
    pub fn new(spec: BankSpec, n: usize, invariance_samples: f64) -> Self {
        let min_xi = min_xi_for_length(spec.q, n);
        let mut wavelets = Vec::new();
        let mut dropped = 0;
        for xi in spec.centre_frequencies() {
            if xi < min_xi {
                dropped += 1;
                continue;
            }
            wavelets.push(Self::make_wavelet(xi, spec.q, n));
            if spec.two_sided {
                wavelets.push(Self::make_wavelet(-xi, spec.q, n));
            }
        }
        assert!(
            !wavelets.is_empty(),
            "no wavelet of Q={} fits in a {n}-sample block below xi_max={}; \
             lengthen the block or lower Q",
            spec.q,
            spec.xi_max
        );
        // Order by signed centre frequency so plotting code gets a monotonic
        // axis without having to sort.
        wavelets.sort_by(|a, b| a.xi.partial_cmp(&b.xi).unwrap());

        // A time-domain Gaussian of standard deviation `T/2` has a
        // frequency-domain standard deviation of `1 / (2 pi (T/2))`.
        let sigma_phi = 1.0 / (PI * invariance_samples);
        let phi = periodise(n, |w| gauss_hat(w, sigma_phi));

        FilterBank { n, wavelets, phi, sigma_phi, dropped }
    }

    /// Lowest centre frequency actually built, cycles/sample.
    pub fn min_xi(&self) -> f64 {
        self.wavelets
            .iter()
            .map(|w| w.xi.abs())
            .fold(f64::INFINITY, f64::min)
    }

    fn make_wavelet(xi: f64, q: f64, n: usize) -> Wavelet {
        let bandwidth = xi.abs() / q;
        let sigma = bw_to_sigma(bandwidth);
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

    #[test]
    fn morlet_has_zero_mean() {
        // Admissibility: the frequency response must vanish at DC, otherwise
        // the "wavelet" leaks a DC term into every coefficient.
        for &xi in &[0.4, 0.1, 0.01] {
            let sigma = bw_to_sigma(xi / 8.0);
            assert!(morlet_hat(0.0, xi, sigma).abs() < 1e-12);
        }
    }

    #[test]
    fn bandwidth_matches_q() {
        // Check the -3 dB points really are where the stated bandwidth says.
        let (xi, q) = (0.2, 8.0);
        let bw = xi / q;
        let sigma = bw_to_sigma(bw);
        let peak = morlet_hat(xi, xi, sigma);
        let edge = morlet_hat(xi + bw / 2.0, xi, sigma);
        let ratio_db = 20.0 * (edge / peak).log10();
        assert!((ratio_db + 3.0).abs() < 0.05, "edge was {ratio_db} dB");
    }

    #[test]
    fn unit_peak_gain() {
        let bank = FilterBank::new(
            BankSpec { q: 8.0, xi_max: 0.4, octaves: 4.0, two_sided: true },
            4096,
            256.0,
        );
        for w in &bank.wavelets {
            let peak = w.spectrum.iter().fold(0f64, |a, v| a.max(v.abs()));
            assert!((peak - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn two_sided_bank_is_symmetric() {
        let bank = FilterBank::new(
            BankSpec { q: 8.0, xi_max: 0.4, octaves: 3.0, two_sided: true },
            2048,
            128.0,
        );
        let positives: Vec<f64> = bank.wavelets.iter().filter(|w| w.xi > 0.0).map(|w| w.xi).collect();
        let negatives: Vec<f64> = bank.wavelets.iter().filter(|w| w.xi < 0.0).map(|w| -w.xi).collect();
        assert_eq!(positives.len(), negatives.len());
        for (p, n) in positives.iter().zip(negatives.iter().rev()) {
            assert!((p - n).abs() < 1e-12);
        }
    }

    #[test]
    fn littlewood_paley_is_flat_across_the_covered_band() {
        let spec = BankSpec { q: 8.0, xi_max: 0.4, octaves: 5.0, two_sided: true };
        let bank = FilterBank::new(spec, 8192, 512.0);
        let lp = bank.littlewood_paley();

        // Only judge flatness where the bank is meant to cover: between the
        // lowest centre frequency and the highest. Outside that the response
        // rolls off by design.
        let lowest = spec.centre_frequencies().last().copied().unwrap();
        let vals: Vec<f64> = (0..bank.n)
            .filter(|&k| {
                let w = bin_freq(k, bank.n).abs();
                w > lowest * 1.5 && w < spec.xi_max * 0.95
            })
            .map(|k| lp[k])
            .collect();

        assert!(!vals.is_empty());
        let lo = vals.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        let hi = vals.iter().fold(0f64, |a, &v| a.max(v));
        let ripple_db = 10.0 * (hi / lo).log10();
        assert!(ripple_db < 1.0, "Littlewood-Paley ripple {ripple_db:.2} dB is too large");
    }
}
