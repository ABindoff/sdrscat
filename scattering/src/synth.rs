//! Synthetic complex baseband signals with known answers, for validating the
//! transform without any hardware attached.
//!
//! Frequencies here are baseband offsets in Hz: 0 is the local oscillator,
//! negative is below it. All generators return exactly `n` samples.

use crate::fft::Cf64;
use std::f64::consts::TAU;

/// Unmodulated complex tone: `A exp(i 2 pi f t)`.
///
/// The reference case. First-order coefficients should read `A` at the band
/// containing `f_hz` and the second order should be empty, because a pure tone
/// has a constant envelope and therefore no modulation.
pub fn tone(n: usize, fs: f64, f_hz: f64, amplitude: f64) -> Vec<Cf64> {
    (0..n)
        .map(|t| Cf64::from_polar(amplitude, TAU * f_hz * t as f64 / fs))
        .collect()
}

/// Amplitude modulation: `A (1 + m cos(2 pi f_m t)) exp(i 2 pi f_c t)`.
///
/// With `depth` in [0, 1] the envelope stays non-negative and the modulation
/// depth recovered by [`crate::Scattergram::modulation_depth`] should equal
/// `depth`.
pub fn am(n: usize, fs: f64, carrier_hz: f64, mod_hz: f64, depth: f64, amplitude: f64) -> Vec<Cf64> {
    (0..n)
        .map(|t| {
            let secs = t as f64 / fs;
            let envelope = amplitude * (1.0 + depth * (TAU * mod_hz * secs).cos());
            Cf64::from_polar(envelope, TAU * carrier_hz * secs)
        })
        .collect()
}

/// Frequency modulation: constant envelope, carrier swinging `deviation_hz`
/// either side of `carrier_hz` at rate `mod_hz`.
///
/// A useful adversarial case: the envelope is flat, so a naive envelope
/// detector sees nothing, but each first-order band sees the carrier sweeping
/// in and out of it, which the second order picks up at `mod_hz`.
pub fn fm(
    n: usize,
    fs: f64,
    carrier_hz: f64,
    mod_hz: f64,
    deviation_hz: f64,
    amplitude: f64,
) -> Vec<Cf64> {
    (0..n)
        .map(|t| {
            let secs = t as f64 / fs;
            // Integral of the instantaneous frequency gives the phase.
            let phase = TAU * carrier_hz * secs
                + (deviation_hz / mod_hz) * (TAU * mod_hz * secs).sin();
            Cf64::from_polar(amplitude, phase)
        })
        .collect()
}

/// Two unmodulated tones, for checking resolution against the stated
/// constant-Q bandwidth.
pub fn two_tone(n: usize, fs: f64, f1_hz: f64, a1: f64, f2_hz: f64, a2: f64) -> Vec<Cf64> {
    let x1 = tone(n, fs, f1_hz, a1);
    let x2 = tone(n, fs, f2_hz, a2);
    x1.iter().zip(x2.iter()).map(|(a, b)| a + b).collect()
}

/// Linear chirp sweeping from `start_hz` to `end_hz` across the block.
pub fn chirp(n: usize, fs: f64, start_hz: f64, end_hz: f64, amplitude: f64) -> Vec<Cf64> {
    let duration = n as f64 / fs;
    let rate = (end_hz - start_hz) / duration;
    (0..n)
        .map(|t| {
            let secs = t as f64 / fs;
            let phase = TAU * (start_hz * secs + 0.5 * rate * secs * secs);
            Cf64::from_polar(amplitude, phase)
        })
        .collect()
}

/// Rectangular pulse train on a carrier: radar, TDMA bursts, or a pager.
///
/// `prf_hz` is the pulse repetition frequency and `duty` the fraction of each
/// period the carrier is on. The second order should show a line at `prf_hz`.
pub fn pulsed(
    n: usize,
    fs: f64,
    carrier_hz: f64,
    prf_hz: f64,
    duty: f64,
    amplitude: f64,
) -> Vec<Cf64> {
    let period = fs / prf_hz;
    (0..n)
        .map(|t| {
            let phase_in_period = (t as f64 % period) / period;
            let on = if phase_in_period < duty { amplitude } else { 0.0 };
            Cf64::from_polar(on, TAU * carrier_hz * t as f64 / fs)
        })
        .collect()
}

/// Deterministic pseudo-random source, so tests involving noise are
/// reproducible without pulling in an RNG dependency.
pub struct Xorshift(u64);

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        Xorshift(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform on [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Adds circular complex Gaussian noise of the given per-component standard
/// deviation, in the same units as the signal amplitude.
pub fn add_noise(x: &mut [Cf64], sigma: f64, seed: u64) {
    let mut rng = Xorshift::new(seed);
    for v in x.iter_mut() {
        *v += Cf64::new(sigma * rng.normal(), sigma * rng.normal());
    }
}

/// Quantises to a signed n-bit grid with the given full-scale amplitude,
/// mimicking the 8-bit ADC in an RTL-SDR. Values beyond full scale clip, which
/// is exactly the front-end overload behaviour worth testing against.
pub fn quantise(x: &mut [Cf64], bits: u32, full_scale: f64) {
    let levels = (1i64 << (bits - 1)) as f64;
    let step = full_scale / levels;
    let q = |v: f64| {
        let code = (v / step).round().clamp(-levels, levels - 1.0);
        code * step
    };
    for v in x.iter_mut() {
        *v = Cf64::new(q(v.re), q(v.im));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn am_envelope_matches_requested_depth() {
        let x = am(4096, 1e6, 100e3, 1e3, 0.4, 2.0);
        let mags: Vec<f64> = x.iter().map(|v| v.norm()).collect();
        let hi = mags.iter().fold(0f64, |a, &v| a.max(v));
        let lo = mags.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        // (max - min) / (max + min) is the classic modulation index.
        assert!(((hi - lo) / (hi + lo) - 0.4).abs() < 1e-3);
    }

    #[test]
    fn fm_has_constant_envelope() {
        let x = fm(4096, 1e6, 100e3, 1e3, 20e3, 1.5);
        for v in &x {
            assert!((v.norm() - 1.5).abs() < 1e-12);
        }
    }

    #[test]
    fn quantisation_clips_rather_than_wrapping() {
        let mut x = vec![Cf64::new(5.0, -5.0)];
        quantise(&mut x, 8, 1.0);
        assert!(x[0].re <= 1.0 && x[0].im >= -1.0);
        assert!(x[0].re > 0.9 && x[0].im < -0.9);
    }

    #[test]
    fn noise_has_the_requested_power() {
        let mut x = vec![Cf64::new(0.0, 0.0); 100_000];
        add_noise(&mut x, 0.5, 42);
        let var = x.iter().map(|v| v.re * v.re).sum::<f64>() / x.len() as f64;
        assert!((var.sqrt() - 0.5).abs() < 0.01, "sigma was {}", var.sqrt());
    }
}
