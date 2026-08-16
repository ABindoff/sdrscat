# sdrscat

A spectrum analyser for cheap SDR hardware, built around wavelet scattering.

The name pairs with `wavscat`, the R package: wavscat is
the transform, sdrscat is the instrument built on it.

## Layout

```
scattering/      pure Rust core, no I/O          <- built, validated
sdr/             SoapySDR capture, sweep, stitch  (not started)
sdrscat-app/     egui GUI                         (not started)
wavscat-rs/      extendr shim to the R package    (not started)
```

The core is deliberately free of hardware and platform dependencies so the R
package and the desktop app can share one implementation and one test suite.

## What the core does

Two orders of scattering on complex baseband (I/Q) or real input:

```
S0        = |x| * phi
S1[l1]    = | x * psi_l1 | * phi
S2[l1,l2] = | |x * psi_l1| * psi_l2 | * phi
```

`S1` over `l1` is a constant-Q spectrum. `S2` is the part worth building: the
inner modulus strips the carrier and leaves the envelope of band `l1`, and the
second wavelet asks how fast that envelope fluctuates. So `l2` is a modulation
rate in Hz, and `S2` read as an image fingerprints a signal (pilot tones, symbol
rates, pulse repetition frequencies, mains-synchronous interference) without
demodulating anything.

Report `2 * S2 / S1`, the modulation depth, rather than raw `S2`. It is
dimensionless and independent of receiver gain, so it means something without
any amplitude calibration, which matters because nothing about a $30 dongle is
calibrated.

## What it does not do

It is not a replacement for a Welch periodogram. Constant-Q means resolution
bandwidth is `f / Q`, so at `Q = 8` two carriers 10 kHz apart at a 1 MHz offset
will not separate. Measured on this machine, with a 0.11 s block:

| carrier offset | scattering RBW | FFT bin on the same block |
|---------------:|---------------:|--------------------------:|
|         10 kHz |        1.2 kHz |                    4.6 Hz |
|        100 kHz |       12.5 kHz |                    4.6 Hz |
|          1 MHz |      125.0 kHz |                    4.6 Hz |

Use an FFT for resolving narrowband tones and for calibrated power. Use this for
structure, modulation and transients. The app should show both panes.

## Block length is the binding constraint

Resolving a modulation rate of `f` Hz at quality factor `Q` needs a wavelet
about `6.4 Q / f` seconds long, and a block four times that so something
survives trimming the contaminated edges. Filters longer than the block are
dropped rather than built, so an empty region of the display means "not
measured", never "quiet". Ask `Config::block_len_for` rather than guessing, and
read `Scattering::coverage` to find out what you got.

At 2.4 MSa/s:

| modulation floor | block  | duration | 1 thread | 4 threads | 24 threads |
|-----------------:|-------:|---------:|---------:|----------:|-----------:|
|           263 Hz |  262 k |   0.11 s |     203% |       66% |        38% |
|            66 Hz | 1048 k |   0.44 s |     208% |       76% |        59% |
|            33 Hz | 2097 k |   0.87 s |     233% |       91% |        69% |

Percentages are of real time; under 100% means it keeps up with the stream. So
mains-rate interference hunting at 33 Hz costs roughly a second of I/Q per
update, and the second-order display refreshes once or twice a second rather
than at video rate. That is physics, not implementation.

Two optimisations get it there, both in `transform.rs`:

- **Sub-band extraction.** A wavelet at `xi` passes a band only `xi/Q` wide, so
  its output is shifted to baseband and processed at its own reduced rate. The
  shift is an exact integer-bin rotation and the modulus is blind to it. Worth
  4.3x, and it flattens the cost curve. `Config::subband = false` reverts to
  full-rate filtering; the test suite asserts the two agree.
- **Skipping empty pairs.** An envelope cannot fluctuate faster than the band
  that produced it is wide, so `l2 > l1/Q1` pairs are genuinely zero and are
  not computed.

The first-order bands are then independent, so they run in parallel.

## Tests

`cargo test --release` runs 29 tests. The interesting ones are in
`scattering/tests/analytic.rs`, which check the transform against signals whose
coefficients can be worked out on paper:

- a tone of amplitude `A` reads `S1 = A`, flat to 0.2 dB across the band
- positive and negative baseband offsets stay distinct (image leakage < -60 dB)
- two tones resolve at 3x RBW spacing and correctly refuse to at 0.2x
- an unmodulated tone shows no second-order energy
- AM rate and depth are recovered to within the filter spacing, at 100 Hz,
  1 kHz and 5 kHz
- modulation depth is invariant to gain over four decades
- a pulse train reveals its PRF
- FM is detected despite having a perfectly constant envelope, which a plain
  envelope detector cannot do
- all of the above still works through an 8-bit ADC with clipping and noise
- the sub-band shortcut matches full-rate filtering
- the transform is non-expansive

`cargo run --release --example budget` prints the coverage and timing tables
above for the machine it runs on.

## Next

1. `sdr/`: SoapySDR capture, retune-and-stitch across the tuner range, DC spike
   and IQ imbalance correction.
2. `sdrscat-app/`: egui, two panes (Welch PSD and the scattering pair).
3. tinySA over USB serial as a tracking generator, which gives both swept
   transmission measurement and the amplitude calibration table.
4. `wavscat-rs/`: extendr wrapper so the R package uses this same core.
