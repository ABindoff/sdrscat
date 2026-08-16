//! Validation against signals whose scattering coefficients can be worked out
//! on paper. If these pass, the transform is doing what the documentation
//! claims and a display built on it can be trusted.
//!
//! Sample rate throughout is 2.4 MHz, the usual RTL-SDR I/Q rate, and
//! frequencies are baseband offsets from the local oscillator in Hz.

use scattering::{synth, Cf64, Config, Scattering, Scattergram};

const FS: f64 = 2.4e6;
const N: usize = 1 << 18; // 262144 samples, about 109 ms at 2.4 MHz

/// A configuration with modulation coverage suited to the test signals: rates
/// from roughly 30 Hz to 30 kHz.
fn config() -> Config {
    Config { max_mod_hz: 30_000.0, octaves2: 10.0, invariance_s: 0.005, ..Config::for_iq(FS) }
}

/// Index of the lambda1 band closest to a given frequency.
fn nearest(axis: &[f64], target: f64) -> usize {
    axis.iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
}

fn argmax(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

// ---------------------------------------------------------------------------
// First order: the constant-Q spectrum
// ---------------------------------------------------------------------------

/// A complex tone of amplitude `A` sitting exactly on a filter's centre
/// frequency must give `S1 = A` at that filter, because the wavelets are
/// normalised to unit peak gain. This is what makes S1 directly calibratable
/// in volts.
#[test]
fn tone_amplitude_is_recovered_in_input_units() {
    let sa = Scattering::new(config(), N);
    let axis = sa.lambda1_hz();

    // Pick an actual centre frequency rather than a round number, so the tone
    // sits on the peak of a filter rather than between two.
    let i = nearest(&axis, 300e3);
    let f = axis[i];
    let amplitude = 0.7;

    let x = synth::tone(N, FS, f, amplitude);
    let out = sa.analyse(&x);
    let s1 = out.s1_mean();

    assert_eq!(argmax(&s1), i, "peak landed in the wrong band");
    let err_pct = 100.0 * (s1[i] - amplitude).abs() / amplitude;
    assert!(err_pct < 1.0, "S1 was {:.4}, expected {amplitude} ({err_pct:.2}% out)", s1[i]);
}

/// The response must be flat in frequency: the same tone moved around the band
/// should read the same amplitude. Otherwise a spectrum display would show
/// tilt that is an artefact of the filter bank, not the signal.
#[test]
fn response_is_flat_across_the_band() {
    let sa = Scattering::new(config(), N);
    let axis = sa.lambda1_hz();
    let amplitude = 1.0;

    let mut readings = Vec::new();
    for target in [-800e3, -200e3, -40e3, 40e3, 200e3, 800e3] {
        let i = nearest(&axis, target);
        let x = synth::tone(N, FS, axis[i], amplitude);
        let out = sa.analyse(&x);
        readings.push(out.s1_mean()[i]);
    }

    let lo = readings.iter().fold(f64::INFINITY, |a, &v| a.min(v));
    let hi = readings.iter().fold(0f64, |a, &v| a.max(v));
    let spread_db = 20.0 * (hi / lo).log10();
    assert!(spread_db < 0.2, "flatness {spread_db:.3} dB across the band: {readings:?}");
}

/// Negative baseband frequency means "below the local oscillator" and is a
/// distinct signal, not a mirror. The two-sided bank must keep them apart.
#[test]
fn positive_and_negative_offsets_are_distinguished() {
    let sa = Scattering::new(config(), N);
    let axis = sa.lambda1_hz();
    let i = nearest(&axis, -450e3);
    assert!(axis[i] < 0.0);

    let x = synth::tone(N, FS, axis[i], 1.0);
    let out = sa.analyse(&x);
    let s1 = out.s1_mean();

    assert_eq!(argmax(&s1), i);

    // The mirror-image band must be quiet. Anything above about -40 dB there
    // would show up on a display as a phantom signal.
    let mirror = nearest(&axis, -axis[i]);
    let leakage_db = 20.0 * (s1[mirror] / s1[i]).log10();
    assert!(leakage_db < -60.0, "image leakage {leakage_db:.1} dB");
}

/// Two tones separated by more than the stated constant-Q bandwidth must
/// resolve into two peaks; the same pair well inside one bandwidth must not.
/// This is the documented limitation, verified rather than assumed.
#[test]
fn resolution_follows_the_stated_bandwidth() {
    let sa = Scattering::new(config(), N);

    let centre = 200e3;
    let rbw = sa.rbw_hz(centre);

    // Well above the elbow the bank is constant-Q, so bandwidth is proportional
    // to offset. Checked as a ratio rather than against `centre / q1`, because
    // bandwidth is derived from the filter-crossing criterion and not set to
    // that formula.
    let ratio = sa.rbw_hz(2.0 * centre) / rbw;
    assert!((ratio - 2.0).abs() < 0.05, "constant-Q region is not constant-Q: {ratio:.3}");

    // Well separated: three bandwidths apart.
    let x = synth::two_tone(N, FS, centre - 1.5 * rbw, 1.0, centre + 1.5 * rbw, 1.0);
    let out = sa.analyse(&x);
    assert_eq!(count_peaks(&out.s1_mean()), 2, "should resolve at 3x RBW spacing");

    // Well inside one bandwidth: a single blob is the correct, honest answer.
    let x = synth::two_tone(N, FS, centre - 0.1 * rbw, 1.0, centre + 0.1 * rbw, 1.0);
    let out = sa.analyse(&x);
    assert_eq!(count_peaks(&out.s1_mean()), 1, "should not resolve at 0.2x RBW spacing");
}

/// Counts local maxima that rise at least 3 dB above their surrounding dips,
/// which is roughly what an eye would call a separate peak on a display.
fn count_peaks(s1: &[f64]) -> usize {
    let peak = s1.iter().fold(0f64, |a, &v| a.max(v));
    let threshold = peak * 0.1;
    let mut count = 0;
    for i in 1..s1.len() - 1 {
        if s1[i] > s1[i - 1] && s1[i] >= s1[i + 1] && s1[i] > threshold {
            // Require a real dip on at least one side, so ripple on a single
            // broad peak is not double-counted.
            let dip_left = s1[..i].iter().rev().take_while(|&&v| v < s1[i]).fold(s1[i], |a, &v| a.min(v));
            if dip_left < s1[i] / 2f64.sqrt() {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Second order: the modulation spectrum
// ---------------------------------------------------------------------------

/// An unmodulated tone has a constant envelope, so every second-order
/// coefficient must be essentially zero. If this fails, the modulation display
/// would show structure in signals that have none.
#[test]
fn unmodulated_tone_has_no_second_order_energy() {
    let sa = Scattering::new(config(), N);
    let axis = sa.lambda1_hz();
    let i = nearest(&axis, 300e3);

    let x = synth::tone(N, FS, axis[i], 1.0);
    let out = sa.analyse(&x);

    let depth = out.modulation_depth();
    let worst = depth[i].iter().fold(0f64, |a, &v| a.max(v));
    assert!(worst < 0.01, "spurious modulation depth {worst:.4} on a pure tone");
}

/// The headline claim: for an AM signal, the second order peaks at the
/// modulation rate, and `2 * S2 / S1` recovers the modulation depth.
#[test]
fn am_modulation_rate_and_depth_are_recovered() {
    let cfg = config();

    // Rates spanning two and a half decades, including 100 Hz, which is the
    // one that matters for hunting mains-synchronous interference. Each block
    // is sized from the config rather than fixed, because a slow modulation
    // needs a long wavelet and therefore a long block.
    for (mod_hz, depth) in [(5_000.0, 0.25), (1_000.0, 0.4), (100.0, 0.6)] {
        let n = cfg.block_len_for(mod_hz).max(N);
        let sa = Scattering::new(cfg, n);
        let cov = sa.coverage();
        assert!(
            cov.min_mod_hz <= mod_hz,
            "block of {n} samples reaches only {:.1} Hz, cannot see {mod_hz} Hz",
            cov.min_mod_hz
        );

        let axis1 = sa.lambda1_hz();
        let i1 = nearest(&axis1, 300e3);

        let x = synth::am(n, FS, axis1[i1], mod_hz, depth, 1.0);
        let out = sa.analyse(&x);

        // The marker readout, which interpolates across neighbouring filters
        // rather than trusting a single one. Interpolation is what lets both
        // figures beat the grid spacing.
        let (rate, read_depth) = out.modulation_peak(i1).unwrap();

        let rate_err_pct = 100.0 * (rate - mod_hz).abs() / mod_hz;
        assert!(
            rate_err_pct < 5.0,
            "modulation rate read {rate:.1} Hz, expected {mod_hz} Hz ({rate_err_pct:.1}% out)"
        );

        let depth_err_pct = 100.0 * (read_depth - depth).abs() / depth;
        assert!(
            depth_err_pct < 10.0,
            "modulation depth read {read_depth:.3}, expected {depth} \
             ({depth_err_pct:.1}% out) at {mod_hz} Hz"
        );
    }
}

/// Modulation depth must not depend on carrier amplitude or receiver gain.
/// This is the property that lets the modulation display be meaningful without
/// any amplitude calibration, which matters because nothing about a $30 dongle
/// is calibrated.
#[test]
fn modulation_depth_is_independent_of_gain() {
    let sa = Scattering::new(config(), N);
    let axis1 = sa.lambda1_hz();
    let i1 = nearest(&axis1, 300e3);

    let mut readings = Vec::new();
    for gain in [0.01, 1.0, 100.0] {
        let x = synth::am(N, FS, axis1[i1], 2_000.0, 0.5, gain);
        let out = sa.analyse(&x);
        let m = out.modulation_depth();
        readings.push(m[i1][argmax(&m[i1])]);
    }

    let lo = readings.iter().fold(f64::INFINITY, |a, &v| a.min(v));
    let hi = readings.iter().fold(0f64, |a, &v| a.max(v));
    assert!((hi - lo) / hi < 0.01, "depth varied with gain: {readings:?}");
}

/// A pulse train should announce its pulse repetition frequency in the second
/// order. This is the radar and TDMA-burst case.
#[test]
fn pulse_train_reveals_its_repetition_frequency() {
    let sa = Scattering::new(config(), N);
    let axis1 = sa.lambda1_hz();
    let axis2 = sa.lambda2_hz();
    let i1 = nearest(&axis1, 300e3);

    let prf = 1_000.0;
    let x = synth::pulsed(N, FS, axis1[i1], prf, 0.25, 1.0);
    let out = sa.analyse(&x);

    let m = out.modulation_depth();
    let i2 = argmax(&m[i1]);
    let err_pct = 100.0 * (axis2[i2] - prf).abs() / prf;
    assert!(err_pct < 15.0, "PRF read {:.0} Hz, expected {prf} Hz", axis2[i2]);
}

/// FM has a perfectly constant envelope, so an envelope detector on the whole
/// signal sees nothing. But each narrow first-order band sees the carrier
/// sweeping through it, so the second order still finds the modulation rate.
/// This is a case where scattering beats naive envelope analysis outright.
#[test]
fn fm_is_detected_despite_a_constant_envelope() {
    let sa = Scattering::new(config(), N);
    let axis1 = sa.lambda1_hz();
    let axis2 = sa.lambda2_hz();

    let carrier = 300e3;
    let mod_hz = 1_000.0;
    // Deviation comparable to the band's own resolution bandwidth, so the
    // carrier really does sweep in and out of each filter.
    let deviation = sa.rbw_hz(carrier);

    let x = synth::fm(N, FS, carrier, mod_hz, deviation, 1.0);
    let out = sa.analyse(&x);

    // Confirm the envelope truly is flat, so the test is testing what it says.
    let envelope_ripple = {
        let mags: Vec<f64> = x.iter().map(|v| v.norm()).collect();
        let hi = mags.iter().fold(0f64, |a, &v| a.max(v));
        let lo = mags.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        hi - lo
    };
    assert!(envelope_ripple < 1e-9);

    // Look in a band off to the side of the carrier, where the sweep passes
    // through rather than sitting still.
    let i1 = nearest(&axis1, carrier + 0.5 * deviation);
    let m = out.modulation_depth();
    let i2 = argmax(&m[i1]);

    // FM through a filter skirt produces harmonics of the modulation rate, so
    // accept the fundamental or its second harmonic.
    let f = axis2[i2];
    let matches = (f - mod_hz).abs() / mod_hz < 0.2 || (f - 2.0 * mod_hz).abs() / (2.0 * mod_hz) < 0.2;
    assert!(matches, "FM modulation read {f:.0} Hz, expected {mod_hz} Hz or its 2nd harmonic");
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

/// Scattering coefficients are non-negative by construction (they are moduli
/// averaged by a non-negative kernel). A negative value means a bug in the
/// decimation or averaging, not a feature of the signal.
#[test]
fn all_coefficients_are_non_negative() {
    let sa = Scattering::new(config(), N);
    let mut x = synth::am(N, FS, 300e3, 1_000.0, 0.5, 1.0);
    synth::add_noise(&mut x, 0.05, 7);
    let out = sa.analyse(&x);

    assert!(out.s0.iter().all(|&v| v >= -1e-9));
    assert!(out.s1.iter().flatten().all(|&v| v >= -1e-9));
    assert!(out.s2.iter().flatten().flatten().all(|&v| v >= -1e-9));
}

/// The transform is non-expansive: it cannot make two signals further apart
/// than they already were. This is the stability property that makes
/// scattering robust to noise and small distortions, and it is cheap to check.
#[test]
fn transform_is_non_expansive() {
    let sa = Scattering::new(Config { order2: false, ..config() }, N);

    let a = synth::am(N, FS, 300e3, 1_000.0, 0.5, 1.0);
    let mut b = a.clone();
    synth::add_noise(&mut b, 0.02, 11);

    let input_distance = l2_complex(&a, &b);
    let out_a = sa.analyse(&a);
    let out_b = sa.analyse(&b);
    let output_distance = l2_scatter(&out_a, &out_b);

    assert!(
        output_distance <= input_distance,
        "output distance {output_distance:.4} exceeded input distance {input_distance:.4}"
    );
}

fn l2_complex(a: &[Cf64], b: &[Cf64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt()
}

fn l2_scatter(a: &Scattergram, b: &Scattergram) -> f64 {
    let mut acc = 0.0;
    for (ra, rb) in a.s1.iter().zip(b.s1.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            acc += (x - y).powi(2);
        }
    }
    acc.sqrt()
}

/// The whole point of the RTL-SDR question: does 8-bit quantisation and a
/// realistic noise floor destroy the modulation reading? It should not, because
/// averaging over a long block buys back a lot of the lost resolution.
#[test]
fn survives_eight_bit_quantisation_and_noise() {
    let sa = Scattering::new(config(), N);
    let axis1 = sa.lambda1_hz();
    let axis2 = sa.lambda2_hz();
    let i1 = nearest(&axis1, 300e3);

    let depth = 0.3;
    let mod_hz = 2_000.0;

    // Carrier at a quarter of ADC full scale, which is where you would set the
    // gain to leave headroom, plus noise at roughly -30 dB relative to it.
    let mut x = synth::am(N, FS, axis1[i1], mod_hz, depth, 0.25);
    synth::add_noise(&mut x, 0.008, 3);
    synth::quantise(&mut x, 8, 1.0);

    let out = sa.analyse(&x);
    let m = out.modulation_depth();
    let i2 = argmax(&m[i1]);

    assert!(
        (axis2[i2] - mod_hz).abs() / mod_hz < 0.15,
        "rate read {:.0} Hz through an 8-bit ADC",
        axis2[i2]
    );
    assert!(
        (m[i1][i2] - depth).abs() / depth < 0.2,
        "depth read {:.3}, expected {depth}, through an 8-bit ADC",
        m[i1][i2]
    );
}

/// Output rate and edge margin must be self-consistent, and the block must be
/// long enough to leave a usable interior. A configuration that leaves no clean
/// samples is a configuration error, and the caller needs to be able to see it.
#[test]
fn output_geometry_is_sane() {
    let cfg = config();
    let sa = Scattering::new(cfg, N);
    let x = synth::tone(N, FS, 300e3, 1.0);
    let out = sa.analyse(&x);

    let block_s = N as f64 / FS;
    assert!(out.out_rate_hz < FS);
    assert!((out.len() as f64 / out.out_rate_hz - block_s).abs() < block_s * 0.01);
    assert!(
        2.0 * out.edge_margin_s < block_s,
        "edge margin {:.4} s leaves nothing usable in a {block_s:.4} s block",
        out.edge_margin_s
    );

    // Nyquist for the output axis must clear the highest modulation rate we
    // claim to report, or the second-order display is aliased.
    assert!(out.lambda2_hz.last().unwrap() <= &cfg.max_mod_hz);
}

/// An over-ambitious request must be met by widening the filters, not by
/// abandoning the frequencies. The elbow is what makes that possible: below it
/// bandwidth is pinned, so time support stops growing and coverage continues.
#[test]
fn the_elbow_keeps_low_rates_at_bounded_cost() {
    // Ask for far more modulation coverage than a 109 ms block could support
    // under a purely constant-Q bank: 14 octaves below 30 kHz reaches about
    // 2 Hz, which at Q = 4 would need a wavelet several seconds long.
    let cfg = Config { max_mod_hz: 30_000.0, octaves2: 14.0, ..config() };
    let sa = Scattering::new(cfg, N);
    let cov = sa.coverage();
    let block_s = N as f64 / FS;

    let elbow = cov.elbow_mod_hz.expect("such a request must reach the elbow");
    assert!(elbow > cov.min_mod_hz, "elbow must sit above the floor");

    // Nothing below one cycle per block, which would be measuring wraparound.
    assert!(
        cov.min_mod_hz > 1.0 / block_s,
        "kept a {:.1} Hz filter in a {block_s:.3} s block",
        cov.min_mod_hz
    );

    // The edge margin still leaves a usable interior, which is the property the
    // pinned bandwidth buys.
    let out = sa.analyse(&synth::tone(N, FS, 300e3, 1.0));
    assert!(2.0 * out.edge_margin_s < block_s);

    // Above the elbow constant-Q holds: bandwidth scales with rate.
    assert!(
        sa.mod_rbw_hz(elbow * 4.0) > sa.mod_rbw_hz(elbow) * 2.0,
        "bandwidth did not scale with rate above the elbow"
    );

    // Below it the bandwidth is pinned, so two different sub-elbow rates must
    // report the same figure. Compared against each other rather than against
    // the elbow filter itself, which is the last constant-Q one and so is still
    // slightly wider than the floor.
    let at_floor = sa.mod_rbw_hz(cov.min_mod_hz);
    let midway = sa.mod_rbw_hz(0.5 * (cov.min_mod_hz + elbow));
    assert!(
        (at_floor - midway).abs() / midway < 1e-9,
        "bandwidth was not pinned below the elbow: {at_floor:.3} vs {midway:.3} Hz"
    );
}

/// The concrete payoff: 100 Hz modulation, the rate that matters for hunting
/// mains-synchronous interference, is now reachable in a block that a purely
/// constant-Q bank could not have managed.
#[test]
fn mains_rate_modulation_is_reachable_in_a_short_block() {
    let cfg = Config { max_mod_hz: 30_000.0, octaves2: 14.0, ..config() };
    let sa = Scattering::new(cfg, N);
    let cov = sa.coverage();

    assert!(
        cov.min_mod_hz <= 100.0,
        "floor is {:.1} Hz in a {:.0} ms block, cannot see mains at 100 Hz",
        cov.min_mod_hz,
        1e3 * N as f64 / FS
    );

    // And it must actually measure it, not merely have a filter there.
    let axis1 = sa.lambda1_hz();
    let axis2 = sa.lambda2_hz();
    let i1 = nearest(&axis1, 300e3);
    let x = synth::am(N, FS, axis1[i1], 100.0, 0.5, 1.0);
    let out = sa.analyse(&x);

    let m = out.modulation_depth();
    let i2 = argmax(&m[i1]);
    // Below the elbow the filters are one bandwidth apart, so accept the read
    // rate within the bandwidth of the filter that made it.
    let tolerance = sa.mod_rbw_hz(100.0);
    assert!(
        (axis2[i2] - 100.0).abs() <= tolerance,
        "read {:.1} Hz for a 100 Hz modulation (bandwidth there is {tolerance:.1} Hz)",
        axis2[i2]
    );
}

/// The sub-band shortcut must not change the answer. It filters each band at
/// its own reduced rate rather than at the full input rate, which is only
/// legitimate if the results match what full-rate filtering gives.
///
/// The modulus is nonlinear, so exact agreement is not on offer: what matters
/// is that the discrepancy sits far below anything a display would show.
#[test]
fn subband_shortcut_matches_full_rate_filtering() {
    let fast_cfg = config();
    let slow_cfg = Config { subband: false, ..fast_cfg };

    // A signal with structure at several scales, so the comparison exercises
    // narrow and wide bands alike.
    let mut x = synth::am(N, FS, 300e3, 2_000.0, 0.5, 1.0);
    for (i, v) in synth::pulsed(N, FS, -700e3, 1_000.0, 0.3, 0.6).iter().enumerate() {
        x[i] += v;
    }
    synth::add_noise(&mut x, 0.01, 19);

    let fast = Scattering::new(fast_cfg, N).analyse(&x);
    let slow = Scattering::new(slow_cfg, N).analyse(&x);

    // First order, as a fraction of the largest coefficient present.
    let s1_fast = fast.s1_mean();
    let s1_slow = slow.s1_mean();
    let scale1 = s1_slow.iter().fold(0f64, |a, &v| a.max(v));
    let worst1 = s1_fast
        .iter()
        .zip(s1_slow.iter())
        .map(|(a, b)| (a - b).abs() / scale1)
        .fold(0f64, f64::max);
    assert!(
        worst1 < 1e-3,
        "first order differs by {:.1} dB of full scale",
        20.0 * worst1.log10()
    );

    // Second order, likewise. This is the one at risk, since it is downstream
    // of the modulus where the approximation lives.
    let m_fast = fast.modulation_depth();
    let m_slow = slow.modulation_depth();
    let worst2 = m_fast
        .iter()
        .flatten()
        .zip(m_slow.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(
        worst2 < 0.01,
        "modulation depth differs by {worst2:.4}, which is visible on a display"
    );
}

/// `block_len_for` must actually predict the block length needed to reach a
/// given modulation rate, so a caller can size their buffer up front rather
/// than by trial and error.
#[test]
fn block_length_prediction_is_honest() {
    // A short averaging window, so the wavelet length is the binding constraint
    // for every target below. `block_len_for` also respects the averaging
    // window, and if that dominated the tightness check below would be testing
    // the wrong thing.
    let cfg = Config { max_mod_hz: 30_000.0, octaves2: 16.0, invariance_s: 0.0005, ..config() };

    for target_hz in [30.0, 100.0, 400.0] {
        let needed = cfg.block_len_for(target_hz);
        let cov = Scattering::new(cfg, needed).coverage();

        assert!(
            cov.min_mod_hz <= target_hz * 1.05,
            "predicted {needed} samples for {target_hz} Hz but only reached {:.1} Hz",
            cov.min_mod_hz
        );

        // Not wildly over-cautious either: rounding to a power of two can cost
        // at most a factor of two, so anything beyond that is waste.
        assert!(
            cov.min_mod_hz > target_hz / 2.5,
            "prediction of {needed} samples for {target_hz} Hz overshot to {:.1} Hz",
            cov.min_mod_hz
        );
    }
}

/// The practical payoff of the whole change: block length no longer scales with
/// `q2`. A sharper second-order bank used to cost proportionally more data to
/// reach a given rate, because a narrower filter is a longer one. With the
/// bandwidth pinned below the elbow, that proportionality is gone.
///
/// Not perfectly flat, and it should not be expected to be: the floor is
/// quantised to one linear step and block lengths are rounded to powers of two,
/// which together leave a doubling of slack. What must be gone is the *trend*.
#[test]
fn block_length_no_longer_scales_with_q2() {
    let base = Config { max_mod_hz: 30_000.0, octaves2: 16.0, invariance_s: 0.0005, ..config() };
    let qs = [2.0, 4.0, 8.0, 16.0];

    let lengths: Vec<usize> = qs
        .iter()
        .map(|&q2| Config { q2, ..base }.block_len_for(100.0))
        .collect();

    // Every one of them must genuinely reach the target, whatever the
    // quantisation did.
    for (&q2, &n) in qs.iter().zip(lengths.iter()) {
        let cov = Scattering::new(Config { q2, ..base }, n).coverage();
        assert!(
            cov.min_mod_hz <= 100.0,
            "q2={q2} predicted {n} samples but reached only {:.1} Hz",
            cov.min_mod_hz
        );
    }

    // An 8x span of q2 must not produce an 8x span of block lengths. One
    // doubling is quantisation; more than that is a surviving proportionality.
    let lo = *lengths.iter().min().unwrap() as f64;
    let hi = *lengths.iter().max().unwrap() as f64;
    assert!(
        hi / lo <= 2.0,
        "block length still tracks q2 across an 8x span: {lengths:?}"
    );
}
