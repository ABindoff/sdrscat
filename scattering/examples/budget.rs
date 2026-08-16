//! Prints what the transform covers and what it costs, at settings a real
//! RTL-SDR display would use. Run with `cargo run --release --example budget`.

use scattering::{synth, Config, Scattering};
use std::time::Instant;

fn main() {
    let fs = 2.4e6;

    println!(
        "RTL-SDR at {:.1} MSa/s complex, {} worker threads\n",
        fs / 1e6,
        rayon::current_num_threads()
    );
    println!(
        "{:<26} {:>10} {:>9} {:>8} {:>8} {:>10} {:>9}",
        "target modulation floor", "block", "duration", "MSa", "l1", "l2", "per block"
    );
    println!("{}", "-".repeat(88));

    for target_hz in [1000.0, 300.0, 100.0, 50.0] {
        let cfg = Config { octaves2: 12.0, ..Config::for_iq(fs) };
        let n = cfg.block_len_for(target_hz);
        let sa = Scattering::new(cfg, n);
        let cov = sa.coverage();

        let x = synth::am(n, fs, 300e3, 1e3, 0.4, 1.0);
        // One untimed pass so FFT plans and caches are warm, as they would be
        // in a running display.
        let _ = sa.analyse(&x);
        let start = Instant::now();
        let out = sa.analyse(&x);
        let elapsed = start.elapsed().as_secs_f64();

        let realtime = elapsed / cov.block_s;
        println!(
            "{:<26} {:>10} {:>8.2}s {:>8.1} {:>8} {:>10} {:>7.2}s  ({:.0}% of real time)",
            format!("{:.0} Hz -> got {:.0} Hz", target_hz, cov.min_mod_hz),
            n,
            cov.block_s,
            n as f64 / 1e6,
            out.lambda1_hz.len(),
            out.lambda2_hz.len(),
            elapsed,
            realtime * 100.0
        );
    }

    println!("\nCost of the sub-band shortcut, on a 0.11 s block:");
    for subband in [false, true] {
        let cfg = Config { subband, ..Config::for_iq(fs) };
        let n = 1 << 18;
        let sa = Scattering::new(cfg, n);
        let x = synth::am(n, fs, 300e3, 1e3, 0.4, 1.0);
        let _ = sa.analyse(&x);
        let start = Instant::now();
        let _ = sa.analyse(&x);
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  subband {:<5} {:>7.3} s per block  ({:>5.1}% of real time)",
            subband,
            elapsed,
            100.0 * elapsed / (n as f64 / fs)
        );
    }

    println!("\nFirst-order resolution bandwidth is constant-Q, so it widens with offset:");
    let cfg = Config::for_iq(fs);
    let sa = Scattering::new(cfg, 1 << 19);
    for offset in [10e3, 100e3, 500e3, 1e6] {
        println!(
            "  at {:>7.0} kHz offset: RBW {:>8.1} kHz  (an FFT of the same block gives {:.1} Hz)",
            offset / 1e3,
            sa.rbw_hz(offset) / 1e3,
            fs / (1u32 << 19) as f64
        );
    }

    println!("\nCoverage at defaults: {}", sa.coverage());
}
