# sdrscat

A spectrum analyser for cheap SDR hardware, built around wavelet scattering.

A hobby project, not affiliated with any institution.

## Layout

```
scattering/      pure Rust core, no I/O          <- built, validated
sdr/             SoapySDR capture, sweep, stitch  (not started)
sdrscat-app/     egui GUI                         (not started)
```

The core is free of hardware and platform dependencies so it can be tested
against synthetic signals without a radio attached, and so the GUI and any
headless tooling share one implementation and one test suite.

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
bandwidth grows with offset, so two carriers 10 kHz apart at a 900 kHz offset
will not separate. Measured on this machine, with a 0.22 s block:

| carrier offset | scattering RBW | FFT bin on the same block |
|---------------:|---------------:|--------------------------:|
|         20 kHz |        1.7 kHz |                    4.6 Hz |
|        100 kHz |        8.7 kHz |                    4.6 Hz |
|        400 kHz |       35.0 kHz |                    4.6 Hz |
|        900 kHz |       76.2 kHz |                    4.6 Hz |

Use an FFT for resolving narrowband tones and for calibrated power. Use this for
structure, modulation and transients. The app should show both panes.

`rbw_hz` reads these off the filter that actually measures at that offset rather
than computing `f / Q`, because below the elbow that formula stops being true.

## Two regimes, and why

A purely constant-Q bank cannot reach low frequencies in a finite block: time
support grows as `1/xi` without bound, so the lowest rate you can see costs
block length in direct proportion, multiplied by `Q`. Following Andén and Mallat
(2014), the bank is constant-Q only while bandwidth stays above a floor set by
the block length. Below that elbow it becomes constant-*bandwidth*: width is
pinned and centre frequencies step down linearly instead of geometrically. Time
support is then bounded by construction and coverage continues to near DC.

The price is that `Q` falls with frequency below the elbow, so resolution stops
improving down there. That is the right trade for this instrument: knowing there
is 100 Hz modulation present matters more than resolving 100 Hz from 103 Hz.

Bandwidths are not chosen, they are derived. Adjacent filters are required to
cross at half power, which makes their squared magnitudes sum to one and the
bank tile the frequency axis flat. In the constant-Q regime that gives
`sigma = xi (1-f)/((1+f) sqrt(2 ln(1/r)))` with `f = 2^(-1/Q)`; below the elbow
it gives a linear spacing of exactly one -3 dB bandwidth.

Flat tiling also fixes the scalloping it causes. A rate falling midway between
two filters excites each at only `1/sqrt(2)`, so reading the single largest
coefficient understates depth by up to 3 dB. Because the squared responses sum
to one, adding the neighbours in quadrature recovers the true amplitude wherever
the tone falls. That is what `Scattergram::modulation_peak` does, and it is what
a marker readout should call.

## Block length is the binding constraint

Reaching a modulation rate of `f` Hz needs a wavelet about `6.4 / f` seconds
long and a block four times that, so something survives trimming the
contaminated edges. Since the elbow pins bandwidth, **this no longer scales with
`Q`**. What `Q` still buys is resolution at that rate, not access to it.

Ask `Config::block_len_for` rather than guessing: it verifies its answer against
the plan the bank will actually build, because the floor is quantised to one
linear step and an analytic estimate can land just short. Read
`Scattering::coverage` to find out what you got.

At 2.4 MSa/s:

| modulation floor | block  | duration | 1 thread | 4 threads | 24 threads |
|-----------------:|-------:|---------:|---------:|----------:|-----------:|
|           138 Hz |  262 k |   0.11 s |     116% |       41% |        24% |
|            69 Hz |  524 k |   0.22 s |     118% |       42% |        27% |
|            34 Hz | 1048 k |   0.44 s |     126% |       47% |        32% |

Percentages are of real time; under 100% means it keeps up with the stream. Hunting
mains-rate interference at 34 Hz costs under half a second of I/Q per update,
where a purely constant-Q bank of the same `Q` needed nearly a second to reach
only 33 Hz and took twice as long to compute it.

Two optimisations get it there, both in `transform.rs`:

- **Sub-band extraction.** A wavelet at `xi` passes a narrow band, so its output
  is shifted to baseband and processed at its own reduced rate. The shift is an
  exact integer-bin rotation and the modulus is blind to it. Worth 6x, and it
  flattens the cost curve. `Config::subband = false` reverts to full-rate
  filtering; the test suite asserts the two agree.
- **Skipping empty pairs.** An envelope cannot fluctuate faster than the band
  that produced it is wide, so pairs above that ceiling are genuinely zero and
  are not computed.

The first-order bands are then independent, so they run in parallel.

## Tests

`cargo test --release` runs 36 tests. The interesting ones are in
`scattering/tests/analytic.rs`, which check the transform against signals whose
coefficients can be worked out on paper:

- a tone of amplitude `A` reads `S1 = A`, flat to 0.2 dB across the band
- positive and negative baseband offsets stay distinct (image leakage < -60 dB)
- two tones resolve at 3x RBW spacing and correctly refuse to at 0.2x
- an unmodulated tone shows no second-order energy
- AM rate and depth are recovered to within 5% and 10%, at 100 Hz, 1 kHz and
  5 kHz
- modulation depth is invariant to gain over four decades
- a pulse train reveals its PRF
- FM is detected despite having a perfectly constant envelope, which a plain
  envelope detector cannot do
- all of the above still works through an 8-bit ADC with clipping and noise
- the sub-band shortcut matches full-rate filtering
- mains-rate modulation at 100 Hz is reachable in a 0.11 s block
- adjacent filters cross at the requested amplitude, and Littlewood-Paley sums
  to one across the covered band
- no wavelet outgrows a quarter of the block, at any `Q` and any block length
- the frequency floor and the required block length no longer scale with `Q`
- above the elbow bandwidth tracks frequency; below it, it is pinned
- the transform is non-expansive

`cargo run --release --example budget` prints the coverage and timing tables
above for the machine it runs on.

## Next

1. `sdr/`: SoapySDR capture, retune-and-stitch across the tuner range, DC spike
   and IQ imbalance correction.
2. `sdrscat-app/`: egui, two panes (Welch PSD and the scattering pair).
3. tinySA over USB serial as a tracking generator, which gives both swept
   transmission measurement and the amplitude calibration table.
4. Amplitude calibration against the tinySA's known output level, so the
   classical pane reads dBm rather than dB relative to nothing in particular.

The GUI's About panel should carry the licence, the repository link and the
support link below.

## Support

This is a hobby project given away free. If it saves you the price of a
benchtop analyser, you can [buy me a coffee](https://buymeacoffee.com/bindoffa).

## Licence

MIT. See [LICENSE](LICENSE).

The constant-bandwidth extension of the filter bank follows the construction in
Andén and Mallat (2014), "Deep Scattering Spectrum", IEEE Transactions on Signal
Processing 62(16):4114-4128, [doi:10.1109/TSP.2014.2326991](https://doi.org/10.1109/TSP.2014.2326991),
implemented from the paper.
