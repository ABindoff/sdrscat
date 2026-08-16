//! Wavelet scattering for spectrum analysis.
//!
//! A scattering transform is a cascade of wavelet filter banks with a modulus
//! taken between each stage and a lowpass average at the end. The first order
//! is a constant-Q spectrum, which is familiar. The second order is the part
//! that earns its keep on radio signals: it reports, for each carrier band, the
//! rate at which that band's envelope fluctuates. Read as an image it is a
//! modulation spectrum, which fingerprints a signal (pilot tones, symbol rates,
//! pulse repetition frequencies, mains-synchronous interference) without
//! demodulating anything.
//!
//! # Scope, honestly stated
//!
//! This is not a replacement for an FFT-based power spectral density estimate.
//! Constant-Q means the resolution bandwidth at carrier offset `f` is `f / Q`,
//! so two tones 10 kHz apart at a 1 MHz offset will not separate at `Q = 8`.
//! Use Welch for resolving narrowband tones and calibrated power; use this for
//! structure, modulation and transients. A spectrum analyser built on this
//! crate should show both.
//!
//! # Block length is the binding constraint
//!
//! Resolving a modulation rate of `f` Hz at quality factor `Q` needs a wavelet
//! about `6.4 Q / f` seconds long, and a block at least four times that so
//! something survives trimming the contaminated edges. Seeing 100 Hz at
//! `Q = 4` therefore takes roughly a second of I/Q, which at 2.4 MSa/s is a few
//! million complex samples. That is a real constraint on any live display, not
//! a tuning detail: ask [`Config::block_len_for`] rather than guessing, and
//! read [`Scattering::coverage`] to find out what you actually got. Filters too
//! long for the block are dropped rather than built, so an empty region of the
//! display means "not measured", never "quiet".
//!
//! # Example
//!
//! ```
//! use scattering::{synth, Config, Scattering};
//!
//! let fs = 2.4e6;                     // RTL-SDR I/Q rate, Hz
//! let n = 1 << 19;                    // block length, about 0.22 s
//!
//! // A carrier 300 kHz above the local oscillator, 40% amplitude-modulated
//! // at 1 kHz.
//! let x = synth::am(n, fs, 300e3, 1e3, 0.4, 1.0);
//!
//! let mut sa = Scattering::new(Config::for_iq(fs), n);
//! let out = sa.analyse(&x);
//!
//! // First order finds the carrier.
//! let i1 = out.peak_lambda1().unwrap();
//! assert!((out.lambda1_hz[i1] - 300e3).abs() < 30e3);
//!
//! // Second order finds the modulation on it: rate and depth, neither of
//! // which an ordinary spectrum display reports.
//! let depth = out.modulation_depth();
//! let i2 = depth[i1]
//!     .iter()
//!     .enumerate()
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .unwrap()
//!     .0;
//! assert!((out.lambda2_hz[i2] - 1e3).abs() < 150.0);
//! assert!((depth[i1][i2] - 0.4).abs() < 0.06);
//! ```

pub mod fft;
pub mod filterbank;
pub mod synth;
pub mod transform;

pub use fft::Cf64;
pub use filterbank::{min_xi_for_length, BankSpec, FilterBank, Wavelet};
pub use transform::{Config, Coverage, Scattergram, Scattering};
