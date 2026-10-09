# Heterodyne

Frequency-dependent RF power, noise and linearity budgets for configurable
Q-band pulsed EPR transmitter and receiver chains. This first release models
static operating states and small-signal electronics, rather than pulse waveforms
or spin dynamics.

Install with Python 3.10 or newer:

```shell
python -m pip install -e ".[test,plot]"
python -m pytest
python examples/q_band.py
```

The example writes receiver, transmitter and leakage budget CSV files in the
current directory. All example component specifications are illustrative user
assumptions, not real device data or hardware recommendations.

```python
from heterodyne import Component, Chain, Mixer, analyze_receiver

receiver = Chain([
    Component("input_loss", -1, kind="passive"),
    Component("lna", 25, noise_figure_db=3, iip3_dbm=-5,
              input_p1db_dbm=-15, max_safe_input_dbm=10),
    Mixer("mixer", -8, noise_figure_db=9, lo_hz=33e9,
          noise_convention="SSB"),
    Component("if_amp", 30, noise_figure_db=4),
])
result = analyze_receiver(receiver, [33.5e9, 34e9, 34.5e9],
                          bandwidth_hz=20e6, input_power_dbm=-90)
print(result.aggregate["noise_figure_db"])
print(result.completeness)
result.to_csv("receiver.csv")
figure = result.plot()  # requires the plot extra
```

### Readable output

Use `print(result)` or `result.print_summary()` for a report with units, overall
results, an aligned stage table, completeness flags, assumptions and warnings.
Unknown specifications display as `unknown`. Sweep reports show the middle
frequency point by default, identify the selected point and show the sweep range.

```python
result.print_summary(frequency_index=0)  # select the first sweep point
text = result.summary(include_stages=False)  # compact report as a string
with open("receiver_report.txt", "w") as stream:
    result.print_summary(file=stream)
```

IQ reports include both branch summaries. Stage P1dB and safe-input margins
remain separate; positive margins indicate headroom. Formatting rounds values
to three decimal places while the numerical result arrays retain full precision.

## Specifications and frequency conversion

### Cable and connector losses

Use `Attenuator` with **positive** total insertion loss in dB:

```python
from heterodyne import Attenuator, FrequencyTable

cable = Attenuator("input_cable_connectors", 1.5, temperature_k=290,
                   ideal_linear=True,
                   provenance={"loss_db": "Measured cable and connector loss"})
frequency_loss = Attenuator("output_cable", FrequencyTable(
    [33e9, 35e9], [1.0, 1.4], "dB", "Datasheet insertion loss"))
```

Insert these components at their physical positions in any chain. Loss before
the first amplifier directly increases receiver noise figure at 290 K; loss
after amplification has a smaller noise contribution. Loss reduces delivered
transmitter power and changes downstream stage power and compression headroom.
Thermal noise is derived from loss and the supplied temperature, rather than a
separate noise-figure specification. Model sections at different temperatures
as separate attenuators. Tables use the local input frequency and reject
extrapolation; scalar losses may be bounded with `operating_band_hz`.

By default, missing attenuator IP3, P1dB and safe-input power remain unknown.
`ideal_linear=True` explicitly assumes no added distortion or compression and
records this assumption in reports. It keeps downstream IP3 and compression
calculations available without inventing finite cable limits. It cannot be
combined with supplied IP3 or P1dB limits. Safe-input power remains independent:
set `max_safe_input_dbm` when known, even for an ideal-linear attenuator. Reports
use `+inf` for explicitly unlimited modeled linearity and `unknown` for missing
specifications. No limiting stage is assigned to an entirely ideal-linear chain.

All frequencies and bandwidths are in Hz, temperatures in kelvin, power in dBm,
and gain, isolation and noise figure in dB. Specify a positive isolation; gain
for a loss is negative. Components support scalar specifications or a
`FrequencyTable(frequency_hz, values, unit, provenance)`. Tables interpolate
linearly in the declared units and reject extrapolation. Use units `dB` or `dBm`
as appropriate. CSV tables have a `frequency_hz` column and a chosen value
column:

```python
from heterodyne import FrequencyTable
gain = FrequencyTable.from_csv("gain.csv", "gain_db", "dB",
                               provenance="Manufacturer datasheet, revision/date")
```

Replace the example's gains, NF, IP3, P1dB, isolation and safe-input levels with
datasheet or measured values; set `operating_band_hz=(minimum, maximum)` and
`provenance={"noise_figure_db": "datasheet ..."}`. A scalar NF is explicitly
reported as constant over the evaluated band. Unknown active NF is never
invented. Without an operating band, scalar validity is a user assumption.

`Mixer` requires `noise_convention="SSB"` or `"DSB"`. Downconversion uses
`RF = LO + IF` for `sideband="upper"` and `RF = LO - IF` for `"lower"`.
Upconversion uses `conversion="up"` with the same sideband definitions.
Specifications are evaluated at each component's **input frequency**, including
IF for an upconverter; transcribe tables onto that frequency axis. A DSB noise
datum is preserved but leaves selected-sideband aggregate noise unknown because
image-band noise is not modeled. No implicit 3 dB conversion is performed.

`IQConversion(i_chain, q_chain)` reports independent branch budgets. Include
splitter/conversion losses explicitly in each branch chain. Combined results
require `combining_convention="power_sum", noise_correlation=0`; they represent
summed powers of branches with explicitly uncorrelated noise, not a coherent
complex-voltage reconstruction or automatic SNR improvement. Branches retain
their own frequency, linearity and noise results.

## Calculations and interpretation

The matched-chain noise calculation uses power gains `G` and noise factors `F`:
`F_total = F1 + (F2 - 1)/G1 + ...`. Reference temperature is 290 K.
For passive loss `L`, `F = 1 + (L - 1) T/290`. Input-referred noise density
is `k (T_source + T_equivalent)`; source temperature defaults to 290 K.
See [Keysight's cascaded noise guidance](https://www.keysight.com/jp/ja/use-cases/characterize-cascaded-noise-figure-in-rf-chains.html).

Supply exactly one of `bandwidth_hz` (equivalent noise bandwidth, local constant
gain/noise approximation) or `passband_hz=(low, high)` (trapezoidal integration
over a sorted input-frequency grid covering the full rectangular passband).
The latter produces one integrated output noise level, repeated for each signal
frequency; input-referred integrated noise uses the gain at that signal frequency.
Frequency translation preserves bandwidth; no image folding is modeled.

The reciprocal-power IP3 cascade is `1/IIP3_total = sum(G_preceding/IIP3_stage)`
with intercept powers in mW. `tone_power_dbm` is **per tone** for the equal-tone
estimate `IM3_in = 3 P_tone - 2 IIP3_total`; `input_power_dbm` is the total
power used for stage overload checks. For two equal tones set total input power
to per-tone power plus approximately 3.0103 dB. These estimates assume locally
constant characteristics across the tones and neglect distortion phase.

Input/output IP3 differ by small-signal gain. Output P1dB converts to input
P1dB as `OP1dB - gain + 1 dB`, accounting for compressed gain. Specify only
one input/output representation per metric. Estimated chain compression uses
the smallest stage input limit referred to the chain input. The limiting-stage
index refers to `result.stages`. This is a stage-limit estimate, not an exact
cascaded 1 dB compression measurement. Compression dynamic range is estimated
input P1dB minus integrated input noise; third-order SFDR is
`(2/3) (IIP3 - integrated input noise)` in dB for the chosen bandwidth.

Missing specifications produce NaNs and completeness flags, including passive
linearity/protection specifications; missing does not mean ideal or unlimited.
Power propagation continues with small-signal gain and warns when compression
is predicted. Safe-input limits are independent damage/protection data.

## Transmit states and leakage

Use `OperatingState("transmit", duty_cycle=0.01)` with `analyze_transmitter`.
`state_gain_db={"transmit": -50, "receive": -1}` selects a component's gain
by state name. Other specifications remain common to the states; use separate
component chains if they differ. Average power assumes no power between pulses.

`analyze_leakage(transmitter, coupling_path, receiver, frequency_hz,
input_power_dbm=..., state=...)` follows one explicit path. Each coupling
component's isolation adds a passive loss before that component's gain. Include
only coupling-path losses here, avoiding double counting isolation already in
gain. All component names across this path must be unique. Receiver-stage
leakage power, compression margin and safe-input margin are reported; unknown
receiver protection limits remain unknown. Parallel/coherent leakage paths are
not combined. An optional `recovery_time_s` records supplied recovery information
but does not simulate it or predict dead time.

`BudgetResult` contains aggregate arrays, per-stage arrays and provenance,
completeness flags, assumptions and warnings. CSV export includes aggregate and
stage numerical arrays; metadata remains in the Python result. IQ results export
and plot through their individual branches. For leakage plots, pass metrics such
as `("leakage_compression_margin_db", "leakage_safe_input_margin_db")`.

The model assumes matched 50 ohm power transfers. It excludes impedance
mismatch, spin/resonator dynamics, LO phase noise, detailed mixer spurs,
correlated noise, and nonlinear or recovery transients. Incomplete data cannot
establish receiver protection.
