//! The scattering transform proper.
//!
//! # What the two orders mean
//!
//! Given input `x` (complex baseband I/Q, or real), a wavelet bank `psi` and a
//! lowpass `phi`:
//!
//! ```text
//! S0        = |x| * phi
//! S1[l1]    = | x * psi_l1 | * phi
//! S2[l1,l2] = | |x * psi_l1| * psi_l2 | * phi
//! ```
//!
//! `S1` indexed by `l1` (lambda-one, a carrier frequency in Hz) is a
//! constant-Q spectrum: the familiar picture. `S2` is the interesting one. The
//! inner modulus strips the carrier and leaves the envelope of band `l1`; the
//! second wavelet then asks at what *rate* that envelope fluctuates. So `l2` is
//! a modulation rate in Hz, not a carrier frequency, and `S2` read as an image
//! over `(l1, l2)` is a modulation spectrum.
//!
//! # Units
//!
//! Wavelets are normalised to unit peak gain, so a complex tone of amplitude
//! `A` volts sitting exactly at `l1` produces `S1[l1] = A` volts. `S2` is best
//! read relative to `S1`: see [`Scattergram::modulation_depth`].
//!
//! # Sample rates
//!
//! `S1` and `S2` share one output time axis, at `out_rate_hz`, which is far
//! below the input rate because averaging by `phi` makes the coefficients
//! smooth. Decimation happens twice: once after the first modulus (down to a
//! rate that still resolves `max_mod_hz`) and once after averaging.

use crate::fft::{decimate_spectrum, power_of_two_factor, with_plans, Cf64, Plans};
use crate::filterbank::{
    self, min_xi_for_length, sigma_min_for_length, BankSpec, FilterBank, DEFAULT_R,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Fraction of the block any one filter may occupy. Matches the rule in
/// [`crate::filterbank`] so wavelets and the averaging kernel are held to the
/// same standard.
const MAX_SUPPORT_FRACTION: f64 = 0.25;

/// Support of the averaging kernel as a multiple of `T`: the kernel has a
/// time-domain standard deviation of `T/2`, so `+/-3 sigma` spans `3T`.
const AVERAGING_SUPPORT_FACTOR: f64 = 3.0;

/// How much wider than a wavelet's `+/-3 sigma` support its extracted sub-band
/// is made. The headroom is what keeps modulus-generated harmonics from
/// folding back onto the modulation rates we report.
const SUBBAND_MARGIN: f64 = 4.0;

/// One first-order band, precomputed so the hot loop does no arithmetic on
/// filter geometry.
#[derive(Clone, Copy, Debug)]
struct Band {
    /// DFT bin of the wavelet's centre frequency, on the full-length grid.
    centre_bin: usize,
    /// Length of the extracted sub-band grid. A power of two dividing `n`.
    len: usize,
}

/// Everything needed to specify an analysis.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Input sample rate, Hz. For an RTL-SDR this is the I/Q rate, e.g. 2.4e6.
    pub fs: f64,

    /// Quality factor of the first-order (carrier) bank.
    ///
    /// This is the constant-Q tradeoff made explicit: the -3 dB resolution
    /// bandwidth at carrier offset `f` is `f / q1` Hz. High `q1` resolves close
    /// tones but responds sluggishly; low `q1` tracks fast transients.
    pub q1: f64,
    /// Octaves of carrier frequency covered below `xi1_max`.
    pub octaves1: f64,
    /// Highest first-order centre frequency as a fraction of the sample rate.
    /// Keep below 0.5; 0.4 leaves headroom for the filter skirt.
    pub xi1_max: f64,

    /// Quality factor of the second-order (modulation rate) bank.
    pub q2: f64,
    /// Octaves of modulation rate covered below `max_mod_hz`.
    pub octaves2: f64,
    /// Highest modulation rate the second order will report, Hz.
    pub max_mod_hz: f64,

    /// Averaging scale `T`, seconds. Sets output time resolution and the
    /// lowest modulation rate that is averaged away rather than resolved.
    pub invariance_s: f64,

    /// True for complex baseband input, which needs a two-sided first-order
    /// bank because negative baseband frequency means "below the local
    /// oscillator" and carries real signal. False for real input.
    pub complex_input: bool,

    /// Compute second order. Turning it off is roughly a 10x saving when only
    /// the constant-Q spectrum is wanted.
    pub order2: bool,

    /// Extract each first-order band at its own reduced sample rate instead of
    /// filtering at the full input rate.
    ///
    /// A wavelet at centre frequency `xi` passes a band only `xi/Q` wide, so
    /// its output can be shifted to baseband and represented at a far lower
    /// rate with no loss. Doing that turns the per-band cost from
    /// `O(N log N)` into something proportional to the band's own width, which
    /// for a constant-Q bank sums to a geometric series rather than a flat
    /// `J x N`. On a 2.4 MSa/s stream it is the difference between roughly ten
    /// times slower than real time and comfortably faster.
    ///
    /// The one approximation: the modulus is a nonlinear operation, so it
    /// generates content above the extracted band's edge, and computing it at
    /// the reduced rate aliases whatever lies beyond. The extraction keeps
    /// generous margin around each wavelet to push that content well down. Set
    /// this to false to filter at full rate and check the difference; the test
    /// suite asserts the two agree.
    pub subband: bool,
}

impl Config {
    /// Defaults for an RTL-SDR style complex baseband stream.
    ///
    /// Modulation rates from roughly 150 Hz to 10 kHz, reachable in a block of
    /// about a fifth of a second. Reaching further down costs block length in
    /// direct proportion: seeing mains-rate interference at 100 Hz needs
    /// something closer to a second of I/Q, which at 2.4 MSa/s is a few million
    /// complex samples. Use [`Config::block_len_for`] to size it rather than
    /// guessing, and expect the second-order display to refresh once or twice a
    /// second at that depth rather than at video rate.
    pub fn for_iq(fs: f64) -> Self {
        Config {
            fs,
            q1: 8.0,
            octaves1: 6.0,
            xi1_max: 0.4,
            q2: 4.0,
            octaves2: 6.0,
            max_mod_hz: 10_000.0,
            invariance_s: 0.005,
            complex_input: true,
            order2: true,
            subband: true,
        }
    }

    /// Defaults for a real-valued input, such as audio or a scope channel.
    pub fn for_real(fs: f64) -> Self {
        Config { complex_input: false, ..Config::for_iq(fs) }
    }

    /// Block length, in samples, needed before a modulation rate of
    /// `min_mod_hz` appears in the second order at all.
    ///
    /// Takes the larger of two constraints, the wavelet length and the
    /// averaging window, and rounds up to a power of two. The dependence on
    /// rate is inverse: seeing 10 Hz costs ten times the block that seeing
    /// 100 Hz does, and no amount of processing evades it.
    ///
    /// Since the bank turns constant-bandwidth below the elbow, this no longer
    /// scales with `q2`. What `q2` still buys is *resolution* at that rate:
    /// reaching 50 Hz is one question, resolving 50 Hz from 55 Hz is another,
    /// and only the second gets harder as `q2` rises. Check
    /// [`Scattering::rbw_hz`] for what the filter down there is actually doing.
    pub fn block_len_for(&self, min_mod_hz: f64) -> usize {
        // Wavelets are built on the grid decimated by d1, so the requirement is
        // expressed there and then scaled back up.
        let d1 = power_of_two_factor(0.25 * self.fs / self.max_mod_hz, 1 << 30);
        let fs1 = self.fs / d1 as f64;
        let target_xi = min_mod_hz / fs1;

        // min_xi_for_length is inversely proportional to n, so evaluating it at
        // n = 1 gives the constant to divide by the target frequency.
        let for_wavelet = min_xi_for_length(1) / target_xi * d1 as f64;
        let for_averaging =
            self.invariance_s * self.fs * AVERAGING_SUPPORT_FACTOR / MAX_SUPPORT_FRACTION;
        let mut n = (for_wavelet.max(for_averaging).ceil().max(1.0) as usize)
            .next_power_of_two()
            .max(d1);

        // That estimate is the *bound*, not the achieved floor. Below the elbow
        // the bank steps down by a fixed amount from wherever constant-Q
        // stopped, so where it finally lands is quantised to within one step
        // and the bound can be missed by that much. Rather than pad by a fudge
        // factor, ask the plan the bank will actually build and double until it
        // genuinely reaches. Planning is pure arithmetic, so this is cheap.
        let spec = BankSpec {
            q: self.q2,
            r: DEFAULT_R,
            xi_max: self.max_mod_hz / fs1,
            octaves: self.octaves2,
            two_sided: false,
        };
        for _ in 0..8 {
            let plan = spec.plan(sigma_min_for_length(n / d1));
            let floor = plan.last().map(|(xi, _)| *xi).unwrap_or(f64::INFINITY);
            if floor <= target_xi {
                break;
            }
            n *= 2;
        }
        n
    }
}

/// What an analyser actually ended up covering, which may be less than was
/// asked for if the block was too short.
#[derive(Clone, Copy, Debug)]
pub struct Coverage {
    /// Smallest carrier offset resolved, Hz.
    pub min_carrier_hz: f64,
    /// Largest carrier offset resolved, Hz.
    pub max_carrier_hz: f64,
    /// Slowest modulation rate resolved, Hz.
    pub min_mod_hz: f64,
    /// Fastest modulation rate resolved, Hz.
    pub max_mod_hz: f64,
    /// Carrier frequency where the first-order bank stops being constant-Q,
    /// in Hz, if the block was short enough to reach it.
    pub elbow_carrier_hz: Option<f64>,
    /// Modulation rate where the second-order bank stops being constant-Q, in
    /// Hz, if the block was short enough to reach it.
    pub elbow_mod_hz: Option<f64>,
    /// Output time-axis sample rate, Hz.
    pub out_rate_hz: f64,
    /// Block duration, seconds.
    pub block_s: f64,
}

impl fmt::Display for Coverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "carrier {:.3}..{:.3} kHz, modulation {:.1}..{:.0} Hz, \
             output {:.1} Hz over a {:.1} ms block",
            self.min_carrier_hz / 1e3,
            self.max_carrier_hz / 1e3,
            self.min_mod_hz,
            self.max_mod_hz,
            self.out_rate_hz,
            self.block_s * 1e3
        )?;
        if let Some(hz) = self.elbow_carrier_hz {
            write!(f, "; constant-Q down to {:.1} kHz carrier", hz / 1e3)?;
        }
        if let Some(hz) = self.elbow_mod_hz {
            write!(f, "; constant-Q down to {hz:.0} Hz modulation")?;
        }
        Ok(())
    }
}

/// Result of one block analysis.
pub struct Scattergram {
    /// First-order centre frequencies, Hz. Signed for complex input: negative
    /// means below the local oscillator. Monotonically increasing.
    pub lambda1_hz: Vec<f64>,
    /// Second-order centre frequencies, Hz. These are modulation *rates*.
    pub lambda2_hz: Vec<f64>,

    /// Zeroth order: the lowpassed envelope of the whole input. `[time]`
    pub s0: Vec<f64>,
    /// First order, `[lambda1][time]`, in input amplitude units.
    pub s1: Vec<Vec<f64>>,
    /// Second order, `[lambda1][lambda2][time]`. Empty if `order2` was false.
    pub s2: Vec<Vec<Vec<f64>>>,

    /// Sample rate of the output time axis, Hz.
    pub out_rate_hz: f64,
    /// Seconds at each end of the output contaminated by circular-convolution
    /// wraparound. Trim this much before believing anything near the edges.
    pub edge_margin_s: f64,
}

impl Scattergram {
    /// Number of output time steps.
    pub fn len(&self) -> usize {
        self.s0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.s0.is_empty()
    }

    /// Time-averaged first order, over the interior only. This is the
    /// constant-Q spectrum a display would show: one value per `lambda1`.
    pub fn s1_mean(&self) -> Vec<f64> {
        let (a, b) = self.interior();
        self.s1.iter().map(|row| mean(&row[a..b])).collect()
    }

    /// Time-averaged second order, `[lambda1][lambda2]`. The modulation map.
    pub fn s2_mean(&self) -> Vec<Vec<f64>> {
        let (a, b) = self.interior();
        self.s2
            .iter()
            .map(|band| band.iter().map(|row| mean(&row[a..b])).collect())
            .collect()
    }

    /// Modulation depth, `[lambda1][lambda2]`, dimensionless.
    ///
    /// Defined as `2 * S2 / S1`. The factor of two is not a fudge: a carrier
    /// amplitude-modulated to depth `m` has envelope `A(1 + m cos(2 pi f t))`,
    /// and an analytic wavelet responds to that real cosine with amplitude
    /// `A m / 2`, so `S2/S1 = m/2`. With the factor restored, a reading of 0.3
    /// at `(l1, l2)` means the carrier in band `l1` is 30% amplitude-modulated
    /// at rate `l2`.
    ///
    /// This ratio, rather than raw `S2`, is what a display should show: it is
    /// independent of how strong the carrier is and of the receiver's gain, so
    /// it needs no amplitude calibration to be meaningful.
    pub fn modulation_depth(&self) -> Vec<Vec<f64>> {
        let s1 = self.s1_mean();
        let s2 = self.s2_mean();
        // Guard against dividing by the noise floor in empty bands, which
        // would paint spurious modulation across the whole display.
        let floor = s1.iter().fold(0f64, |a, &v| a.max(v)) * 1e-3;
        s2.iter()
            .zip(s1.iter())
            .map(|(band, &c)| {
                let denom = c.max(floor);
                band.iter().map(|&v| 2.0 * v / denom).collect()
            })
            .collect()
    }

    /// Index of the strongest `lambda1` band, by time-averaged `S1`.
    pub fn peak_lambda1(&self) -> Option<usize> {
        argmax(&self.s1_mean())
    }

    /// Strongest modulation present in carrier band `i1`, as
    /// `(rate in Hz, depth)`.
    ///
    /// Reading the single largest coefficient understates the depth by up to
    /// 3 dB, because a rate falling midway between two filters excites each at
    /// only `1/sqrt(2)`. That scalloping is not a defect to be tuned away: the
    /// filters are deliberately spaced to cross there, which is what makes the
    /// bank tile flat.
    ///
    /// Flat tiling is also the cure. For a single tone the squared responses of
    /// all filters sum to one, so adding the neighbours in quadrature recovers
    /// the true amplitude wherever the tone happens to fall. The rate is then
    /// interpolated as the energy-weighted centroid in log frequency, which
    /// beats the grid spacing rather than being limited by it.
    ///
    /// This is what a marker readout should call. Returns `None` if the second
    /// order was not computed.
    pub fn modulation_peak(&self, i1: usize) -> Option<(f64, f64)> {
        let depth = self.modulation_depth();
        let row = depth.get(i1)?;
        let i2 = argmax(row)?;
        if row[i2] <= 0.0 {
            return Some((self.lambda2_hz.get(i2).copied().unwrap_or(0.0), 0.0));
        }

        let lo = i2.saturating_sub(1);
        let hi = (i2 + 1).min(row.len() - 1);

        let mut energy = 0.0;
        let mut log_rate = 0.0;
        for (v, rate) in row[lo..=hi].iter().zip(self.lambda2_hz[lo..=hi].iter()) {
            let w = v * v;
            energy += w;
            log_rate += w * rate.ln();
        }

        Some((( log_rate / energy).exp(), energy.sqrt()))
    }

    /// Output sample range excluding the contaminated edges.
    fn interior(&self) -> (usize, usize) {
        let margin = (self.edge_margin_s * self.out_rate_hz).ceil() as usize;
        let n = self.len();
        if 2 * margin >= n {
            // Block too short to have a clean interior; fall back to the whole
            // thing rather than returning nothing.
            (0, n)
        } else {
            (margin, n - margin)
        }
    }
}

/// A configured, reusable analyser. Building one precomputes the filter banks
/// and FFT plans, so the per-block cost in a live display is just transforms.
pub struct Scattering {
    config: Config,
    n: usize,
    bank1: FilterBank,
    bank2: FilterBank,
    /// Decimation applied to first-order envelopes before second order.
    d1: usize,
    /// Further decimation applied after averaging by phi.
    d2: usize,
    /// Sub-band extraction geometry for the first order, parallel to
    /// `bank1.wavelets`.
    bands1: Vec<Band>,
    /// Sub-band extraction geometry for the second order, parallel to
    /// `bank2.wavelets`, on the decimated grid.
    bands2: Vec<Band>,
    /// Averaging kernel sampled on each grid length in use, keyed by length.
    phi_cache: HashMap<usize, Vec<f64>>,
    /// Output time-axis length.
    n_out: usize,
    /// Highest modulation rate that can physically appear in each first-order
    /// band, in the same normalised units as `bank2`. Parallel to
    /// `bank1.wavelets`.
    mod_ceiling: Vec<f64>,
}

impl Scattering {
    /// Prepares an analyser for blocks of exactly `n` samples.
    ///
    /// `n` must be a power of two. Panics if the configuration cannot be
    /// satisfied at that block length, which usually means `invariance_s` is
    /// longer than the block.
    pub fn new(config: Config, n: usize) -> Self {
        assert!(n.is_power_of_two(), "block length {n} must be a power of two");
        assert!(config.max_mod_hz > 0.0 && config.max_mod_hz < config.fs / 2.0);
        // The averaging kernel has a time-domain standard deviation of T/2, so
        // its +/-3 sigma support is 3T. Held to the same quarter-block rule as
        // the wavelets, that caps T at n/12 samples.
        assert!(
            config.invariance_s * config.fs * AVERAGING_SUPPORT_FACTOR / MAX_SUPPORT_FRACTION
                <= n as f64,
            "invariance {} s needs at least {} samples, block is {n}",
            config.invariance_s,
            (config.invariance_s * config.fs * AVERAGING_SUPPORT_FACTOR / MAX_SUPPORT_FRACTION)
                .ceil()
        );

        let invariance_samples = config.invariance_s * config.fs;

        let spec1 = BankSpec {
            q: config.q1,
            r: DEFAULT_R,
            xi_max: config.xi1_max,
            octaves: config.octaves1,
            two_sided: config.complex_input,
        };
        let bank1 = FilterBank::new(spec1, n, invariance_samples);

        // Decimate first-order envelopes as far as the modulation range
        // allows. Placing max_mod_hz at a quarter of the decimated Nyquist
        // leaves the top of the lambda2 bank clear of the band edge.
        let d1 = power_of_two_factor(0.25 * config.fs / config.max_mod_hz, n);
        let n1 = n / d1;
        let fs1 = config.fs / d1 as f64;

        // The second-order bank lives on the decimated grid, and is always
        // one-sided: its input is a modulus, hence real and non-negative, so
        // negative modulation rates would be redundant.
        let spec2 = BankSpec {
            q: config.q2,
            r: DEFAULT_R,
            xi_max: config.max_mod_hz / fs1,
            octaves: config.octaves2,
            two_sided: false,
        };
        let bank2 = FilterBank::new(spec2, n1, config.invariance_s * fs1);

        // After averaging, the coefficients are bandlimited to roughly
        // sigma_phi; keeping three standard deviations inside the new Nyquist
        // is generous.
        let d2 = power_of_two_factor(1.0 / (6.0 * bank2.sigma_phi), n1);

        let n_out = n1 / d2;

        // A first-order band must be wide enough to hold its wavelet with
        // margin, and at least as wide as the common order-2 grid, since
        // everything is brought back to that grid afterwards.
        let bands1 = plan_bands(&bank1, n, n1, config.subband);
        // Second-order bands need only reach the output grid.
        let bands2 = plan_bands(&bank2, n1, n_out, config.subband);

        // A band of width `xi/Q1` cannot carry an envelope fluctuating faster
        // than that width, so pairs above this ceiling are physically empty and
        // are skipped rather than computed. Expressed in the normalised units
        // of the second-order grid.
        let mod_ceiling = bank1
            .wavelets
            .iter()
            .map(|w| w.bandwidth * d1 as f64)
            .collect();

        let mut phi_cache = HashMap::new();
        for len in bands1
            .iter()
            .map(|_| n1)
            .chain(bands2.iter().map(|b| b.len))
            .chain(std::iter::once(n1))
        {
            phi_cache.entry(len).or_insert_with(|| {
                // The same physical filter on a shorter grid spans the same
                // time at a lower rate, so its normalised width scales up.
                filterbank::lowpass(len, bank2.sigma_phi * n1 as f64 / len as f64)
            });
        }

        Scattering {
            config,
            n,
            bank1,
            bank2,
            d1,
            d2,
            bands1,
            bands2,
            phi_cache,
            n_out,
            mod_ceiling,
        }
    }

    /// Block length this analyser expects.
    pub fn block_len(&self) -> usize {
        self.n
    }

    /// Output sample rate, Hz.
    pub fn out_rate_hz(&self) -> f64 {
        self.config.fs / (self.d1 * self.d2) as f64
    }

    /// First-order centre frequencies in Hz, signed, increasing.
    pub fn lambda1_hz(&self) -> Vec<f64> {
        self.bank1.wavelets.iter().map(|w| w.xi * self.config.fs).collect()
    }

    /// Second-order centre frequencies (modulation rates) in Hz, increasing.
    pub fn lambda2_hz(&self) -> Vec<f64> {
        let fs1 = self.config.fs / self.d1 as f64;
        self.bank2.wavelets.iter().map(|w| w.xi * fs1).collect()
    }

    /// What this analyser actually covers, after any filters too long for the
    /// block were dropped. Worth logging once at startup and showing on the
    /// display, so an empty region reads as "not measured" rather than "quiet".
    pub fn coverage(&self) -> Coverage {
        let fs1 = self.config.fs / self.d1 as f64;
        let l1 = self.lambda1_hz();
        let l2 = self.lambda2_hz();
        Coverage {
            min_carrier_hz: self.bank1.min_xi() * self.config.fs,
            max_carrier_hz: l1.last().copied().unwrap_or(0.0).abs(),
            min_mod_hz: self.bank2.min_xi() * fs1,
            max_mod_hz: l2.last().copied().unwrap_or(0.0),
            elbow_carrier_hz: self.bank1.elbow_xi.map(|xi| xi * self.config.fs),
            elbow_mod_hz: self.bank2.elbow_xi.map(|xi| xi * fs1),
            out_rate_hz: self.out_rate_hz(),
            block_s: self.n as f64 / self.config.fs,
        }
    }

    /// Resolution bandwidth of the first-order analysis at a given carrier
    /// offset, in Hz: the -3 dB width of the filter that actually measures
    /// there.
    ///
    /// Read off the bank rather than computed as `offset / Q`, because that
    /// formula holds only above the elbow. Below it the filters are constant
    /// bandwidth and the true figure flattens out. A display should show this
    /// number, since it is the difference between "these two signals are
    /// genuinely one" and "my analyser cannot tell them apart".
    pub fn rbw_hz(&self, offset_hz: f64) -> f64 {
        let xi = offset_hz / self.config.fs;
        match self.bank1.nearest(xi) {
            Some(w) => w.bandwidth * self.config.fs,
            None => f64::NAN,
        }
    }

    /// Resolution bandwidth of the second-order analysis at a given modulation
    /// rate, in Hz. Same story as [`Scattering::rbw_hz`], one order up.
    pub fn mod_rbw_hz(&self, rate_hz: f64) -> f64 {
        let fs1 = self.config.fs / self.d1 as f64;
        match self.bank2.nearest(rate_hz / fs1) {
            Some(w) => w.bandwidth * fs1,
            None => f64::NAN,
        }
    }

    /// Analyses one block. `x` must be exactly `block_len()` samples.
    ///
    /// First-order bands are independent of one another, so they run in
    /// parallel across the thread pool. Takes `&self`, which means one analyser
    /// can be shared between threads.
    pub fn analyse(&self, x: &[Cf64]) -> Scattergram {
        assert_eq!(x.len(), self.n, "block length mismatch");

        let n1 = self.n / self.d1;

        let (s0, spectrum) = with_plans(|plans| {
            // Zeroth order: lowpassed envelope of the input as a whole.
            let mut env: Vec<Cf64> = x.iter().map(|v| Cf64::new(v.norm(), 0.0)).collect();
            plans.forward(&mut env);
            let s0 = average(
                plans,
                &self.phi_cache[&n1],
                self.n_out,
                &decimate_spectrum(&env, self.d1),
            );

            // Forward transform of the input, reused by every first-order band.
            let mut spectrum = x.to_vec();
            plans.forward(&mut spectrum);
            (s0, spectrum)
        });

        let per_band: Vec<(Vec<f64>, Vec<Vec<f64>>)> = (0..self.bank1.wavelets.len())
            .into_par_iter()
            .map(|i1| with_plans(|plans| self.analyse_band(i1, &spectrum, n1, plans)))
            .collect();

        let mut s1 = Vec::with_capacity(per_band.len());
        let mut s2 = Vec::with_capacity(if self.config.order2 { per_band.len() } else { 0 });
        for (first, second) in per_band {
            s1.push(first);
            if self.config.order2 {
                s2.push(second);
            }
        }

        Scattergram {
            lambda1_hz: self.lambda1_hz(),
            lambda2_hz: self.lambda2_hz(),
            s0,
            s1,
            s2,
            out_rate_hz: self.out_rate_hz(),
            edge_margin_s: self.edge_margin_s(),
        }
    }

    /// One first-order band and, if enabled, every second-order coefficient
    /// hanging off it.
    fn analyse_band(
        &self,
        i1: usize,
        spectrum: &[Cf64],
        n1: usize,
        plans: &mut Plans,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        // U1 = |x * psi_l1|. The modulus discards the carrier and keeps the
        // envelope, which is what makes the second order see modulation rather
        // than frequency.
        //
        // Rather than filtering across the whole grid, gather only the bins the
        // wavelet actually passes, re-centred on DC. That is an exact frequency
        // shift by an integer number of bins, and the modulus is blind to it,
        // so the envelope is unchanged. The saving is that the transforms below
        // run at the band's own width instead of the full input rate.
        let band = self.bands1[i1];
        let mut sub = extract_band(
            spectrum,
            &self.bank1.wavelets[i1].spectrum,
            band.centre_bin,
            band.len,
        );
        plans.inverse(&mut sub);
        let mut u1: Vec<Cf64> = sub.iter().map(|v| Cf64::new(v.norm(), 0.0)).collect();

        plans.forward(&mut u1);
        let u1_small = decimate_spectrum(&u1, band.len / n1);

        let first = average(plans, &self.phi_cache[&n1], self.n_out, &u1_small);

        if !self.config.order2 {
            return (first, Vec::new());
        }

        let ceiling = self.mod_ceiling[i1];
        let mut rows = Vec::with_capacity(self.bank2.wavelets.len());
        for (i2, w2) in self.bank2.wavelets.iter().enumerate() {
            // An envelope cannot fluctuate faster than the band that produced
            // it is wide, so anything above the ceiling is genuinely zero
            // rather than merely unmeasured, and is skipped.
            if w2.xi > ceiling {
                rows.push(vec![0.0; self.n_out]);
                continue;
            }
            let band2 = self.bands2[i2];
            let mut sub2 = extract_band(&u1_small, &w2.spectrum, band2.centre_bin, band2.len);
            plans.inverse(&mut sub2);
            let mut u2: Vec<Cf64> = sub2.iter().map(|v| Cf64::new(v.norm(), 0.0)).collect();
            plans.forward(&mut u2);
            rows.push(average(plans, &self.phi_cache[&band2.len], self.n_out, &u2));
        }
        (first, rows)
    }

    fn edge_margin_s(&self) -> f64 {
        let fs1 = self.config.fs / self.d1 as f64;
        let order1 = self.bank1.max_time_support() / self.config.fs;
        let order2 = self.bank2.max_time_support() / fs1;
        let averaging = AVERAGING_SUPPORT_FACTOR * self.config.invariance_s;
        order1.max(order2).max(averaging)
    }
}

/// Works out how wide each wavelet's extracted sub-band has to be.
///
/// Wide enough to hold the wavelet with margin, never narrower than the grid
/// the result must land on, and never wider than the grid it came from.
fn plan_bands(bank: &FilterBank, grid_len: usize, floor_len: usize, subband: bool) -> Vec<Band> {
    bank.wavelets
        .iter()
        .map(|w| {
            let needed = (SUBBAND_MARGIN * 6.0 * w.sigma * grid_len as f64).ceil() as usize;
            let len = if subband {
                needed.max(floor_len).next_power_of_two().min(grid_len)
            } else {
                grid_len
            };
            let centre_bin = (w.xi * grid_len as f64).round().rem_euclid(grid_len as f64) as usize;
            Band { centre_bin, len }
        })
        .collect()
}

/// Gathers the bins a wavelet passes, re-centred on DC, at a reduced rate.
///
/// Shifting the band to baseband is an exact integer-bin frequency shift, and
/// the modulus applied downstream is blind to it, so the envelope is unchanged.
/// The `1/d` scaling makes the result equal the filtered signal sampled every
/// `d`th point rather than `d` times it.
fn extract_band(spectrum: &[Cf64], psi: &[f64], centre_bin: usize, len: usize) -> Vec<Cf64> {
    let n = spectrum.len();
    let scale = len as f64 / n as f64;
    let mut sub = vec![Cf64::new(0.0, 0.0); len];
    for (k, slot) in sub.iter_mut().enumerate() {
        let offset = if k < len / 2 { k as isize } else { k as isize - len as isize };
        let j = (centre_bin as isize + offset).rem_euclid(n as isize) as usize;
        *slot = spectrum[j] * psi[j] * scale;
    }
    sub
}

/// Applies `phi` and decimates to the output length, returning the real
/// time-domain result.
///
/// Free-standing rather than a method so callers can hold an immutable borrow
/// of a filter bank while this takes a mutable borrow of the FFT plans.
fn average(plans: &mut Plans, phi: &[f64], out_len: usize, spectrum: &[Cf64]) -> Vec<f64> {
    let filtered: Vec<Cf64> = spectrum
        .iter()
        .zip(phi.iter())
        .map(|(v, &p)| v * p)
        .collect();
    let mut small = decimate_spectrum(&filtered, spectrum.len() / out_len);
    plans.inverse(&mut small);
    // The input to phi is a real non-negative modulus and phi is real and
    // symmetric, so the imaginary part is numerical dust.
    small.iter().map(|v| v.re).collect()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn argmax(xs: &[f64]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .fold(None, |best, (i, &v)| match best {
            Some((_, bv)) if bv >= v => best,
            _ => Some((i, v)),
        })
        .map(|(i, _)| i)
}
