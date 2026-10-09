"""Small-signal budgets; compression results are stage-limit estimates."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import Chain, FrequencyTable, IQConversion, Mixer, OperatingState

K_B = 1.380649e-23
T_REF = 290.0


def db(value):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(value)


@dataclass
class BudgetResult:
    frequency_hz: np.ndarray
    aggregate: dict[str, np.ndarray]
    stages: list[dict]
    completeness: dict[str, bool]
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    branches: dict[str, "BudgetResult"] = field(default_factory=dict)

    def summary(self, frequency_index=None, *, include_stages=True):
        """Return a readable report at one sweep point (middle point by default)."""
        index = len(self.frequency_hz) // 2 if frequency_index is None else frequency_index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.frequency_hz):
            raise ValueError("frequency_index must select a valid sweep point")

        def number(value):
            if np.isnan(value):
                return 'unknown'
            if np.isinf(value):
                return '+inf' if value > 0 else '-inf'
            return f'{value:.3f}'

        def metric(key):
            units = (('_dbm_hz', 'dBm/Hz'), ('_dbm', 'dBm'), ('_db', 'dB'),
                     ('_hz', 'GHz'), ('_k', 'K'))
            for suffix, unit in units:
                if key.endswith(suffix):
                    name = key[:-len(suffix)]
                    break
            else:
                name, unit = key, ''
            label = name.replace('_', ' ').capitalize()
            label = label.replace('Iip3', 'IIP3').replace('P1db', 'P1dB').replace('Sfdr3', 'SFDR3')
            label = label.replace('Snr', 'SNR').replace('Im3', 'IM3')
            return label, unit

        lines = ['RF budget summary',
                 f'Input frequency: {self.frequency_hz[index] / 1e9:.6f} GHz '
                 f'(point {index + 1} of {len(self.frequency_hz)})']
        if len(self.frequency_hz) > 1:
            lines.append(f'Sweep range: {np.min(self.frequency_hz) / 1e9:.6f} to '
                         f'{np.max(self.frequency_hz) / 1e9:.6f} GHz')
        if self.aggregate:
            lines.extend(['', 'Overall results:'])
            labels = [metric(key)[0] for key in self.aggregate]
            width = max(map(len, labels))
            for key, values in self.aggregate.items():
                label, unit = metric(key)
                value = values[index]
                if key == 'compression_limiting_stage_index':
                    value_text = self.stages[int(value)]['name'] if np.isfinite(value) else 'unknown'
                else:
                    value_text = number(value / 1e9 if unit == 'GHz' else value)
                lines.append(f'  {label:<{width}} : {value_text}' + (f' {unit}' if unit else ''))
        if include_stages and self.stages:
            lines.extend(['', 'Stage details (small-signal powers):'])
            columns = [('Stage', None), ('In GHz', 'input_frequency_hz'),
                       ('Out GHz', 'output_frequency_hz'), ('Gain dB', 'gain_db'),
                       ('NF dB', 'noise_figure_db'), ('In dBm', 'input_power_dbm'),
                       ('Out dBm', 'output_power_dbm'),
                       ('P1dB margin dB', 'compression_headroom_db'),
                       ('Safe margin dB', 'safe_input_margin_db')]
            rows = []
            for stage in self.stages:
                rows.append([stage['name']] + [number(stage[key][index] / 1e9
                             if key.endswith('_hz') else stage[key][index]) for _, key in columns[1:]])
            widths = [max(len(title), *(len(row[i]) for row in rows))
                      for i, (title, _) in enumerate(columns)]
            lines.append('  ' + ' | '.join(title.ljust(width) for (title, _), width in zip(columns, widths)))
            lines.append('  ' + '-+-'.join('-' * width for width in widths))
            lines.extend('  ' + ' | '.join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)
        if self.completeness:
            lines.extend(['', 'Specification completeness:'])
            lines.extend(f"  {key.replace('_', ' ').capitalize()}: {'complete' if valid else 'incomplete'}"
                         for key, valid in self.completeness.items())
        for title, messages in [('Assumptions', self.assumptions), ('Warnings', self.warnings)]:
            if messages:
                lines.extend(['', title + ':'])
                lines.extend('  - ' + message for message in dict.fromkeys(messages))
        for name, branch in self.branches.items():
            lines.extend(['', f'{name.upper()} branch:'])
            lines.extend('  ' + line for line in branch.summary(index, include_stages=include_stages).splitlines())
        return '\n'.join(lines)

    def print_summary(self, frequency_index=None, *, include_stages=True, file=None):
        """Print the report to stdout or a supplied text stream."""
        print(self.summary(frequency_index, include_stages=include_stages), file=file)

    def __str__(self):
        return self.summary()

    def to_csv(self, path):
        """Export aggregate and per-stage arrays as a wide table."""
        columns = {"frequency_hz": self.frequency_hz, **self.aggregate}
        for stage in self.stages:
            columns.update({f"{stage['name']}.{key}": value for key, value in stage.items()
                            if isinstance(value, np.ndarray)})
        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values()))

    def plot(self, metrics=("gain_db", "noise_figure_db", "iip3_dbm", "compression_headroom_db")):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(metrics), 1, squeeze=False, sharex=True)
        for ax, metric in zip(axes[:, 0], metrics):
            ax.plot(self.frequency_hz / 1e9, self.aggregate[metric])
            ax.set_ylabel(metric)
            ax.grid(True)
        axes[-1, 0].set_xlabel("Input frequency (GHz)")
        fig.tight_layout()
        return fig


def _grid(frequency_hz):
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("Supply finite nonnegative frequencies")
    return f


def _broadcast(value, f, name):
    a = np.broadcast_to(np.asarray(value, dtype=float), f.shape).copy()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _analyze(chain, frequency_hz, input_power_dbm, state):
    f = _grid(frequency_hz)
    power = _broadcast(input_power_dbm, f, "Input power")
    local = f.copy()
    gain_total = np.zeros_like(f)
    noise_factor = np.ones_like(f)
    inverse_ip3 = np.zeros_like(f)
    compression = np.full_like(f, np.inf)
    stages, assumptions, warnings = [], [], []
    ip3_complete = p1_complete = safe_complete = True
    for c in chain.components:
        gain = c.evaluate("gain_db", local, state.name)
        g = 10 ** (gain / 10)
        preceding = 10 ** (gain_total / 10)
        nf = c.evaluate("noise_figure_db", local)
        passive = c.kind in ("passive", "filter", "switch", "splitter") and not isinstance(c, Mixer)
        if passive and np.any(gain > 0):
            raise ValueError(f"{c.name}: passive gain cannot exceed zero dB")
        if passive and c.noise_figure_db is None:
            factor = 1 + (1 / g - 1) * c.temperature_k / T_REF
        else:
            if np.any(nf < 0):
                raise ValueError("Noise figure cannot be negative")
            factor = 10 ** (nf / 10)
        if isinstance(c, Mixer) and c.noise_convention == "DSB":
            # A DSB datum alone cannot specify selected-sideband/image noise.
            factor = np.full_like(f, np.nan)
            warnings.append(f"{c.name}: DSB NF cannot be used for an SSB chain without an image-noise model")
        if c.noise_figure_db is not None and not isinstance(c.noise_figure_db, FrequencyTable):
            assumptions.append(f"{c.name}: constant NF over evaluated operating band")
        if c.operating_band_hz is None:
            assumptions.append(f"{c.name}: operating band not supplied; user must verify validity")
        noise_factor += (factor - 1) / preceding
        ip3 = c.evaluate("iip3_dbm", local)
        if c.oip3_dbm is not None:
            ip3 = c.evaluate("oip3_dbm", local) - gain
        p1 = c.evaluate("input_p1db_dbm", local)
        if c.output_p1db_dbm is not None:
            # At P1dB actual output is input + small-signal gain - 1 dB.
            p1 = c.evaluate("output_p1db_dbm", local) - gain + 1
        safe = c.evaluate("max_safe_input_dbm", local)
        ip3_complete &= bool(np.all(np.isfinite(ip3)))
        p1_complete &= bool(np.all(np.isfinite(p1)))
        safe_complete &= bool(np.all(np.isfinite(safe)))
        inverse_ip3 += preceding / 10 ** (ip3 / 10)
        compression = np.minimum(compression, p1 - gain_total)
        out_f = c.output_frequency(local) if isinstance(c, Mixer) else local.copy()
        provenance = dict(c.provenance)
        for key in ('gain_db', 'noise_figure_db', 'iip3_dbm', 'oip3_dbm',
                    'input_p1db_dbm', 'output_p1db_dbm', 'max_safe_input_dbm', 'isolation_db'):
            spec = getattr(c, key)
            provenance.setdefault(key, spec.provenance if isinstance(spec, FrequencyTable) else 'user assumption')
        stage = dict(name=c.name, provenance=provenance,
                     input_frequency_hz=local.copy(), output_frequency_hz=out_f.copy(),
                     gain_db=gain, cumulative_gain_db=gain_total + gain,
                     noise_figure_db=db(factor), cumulative_noise_figure_db=db(noise_factor),
                     input_power_dbm=power + gain_total, output_power_dbm=power + gain_total + gain,
                     iip3_dbm=ip3, input_p1db_dbm=p1,
                     compression_headroom_db=p1 - power - gain_total,
                     safe_input_margin_db=safe - power - gain_total)
        if isinstance(c, Mixer):
            stage.update(lo_hz=c.lo_hz, sideband=c.sideband, noise_convention=c.noise_convention)
        stages.append(stage)
        local = out_f
        gain_total += gain
    aggregate = dict(gain_db=gain_total, noise_figure_db=db(noise_factor),
                     equivalent_noise_temperature_k=T_REF * (noise_factor - 1),
                     iip3_dbm=-db(inverse_ip3),
                     estimated_input_p1db_dbm=compression,
                     compression_headroom_db=compression - power,
                     output_power_dbm=power + gain_total, output_frequency_hz=local)
    complete = dict(noise=bool(np.all(np.isfinite(noise_factor))), ip3=ip3_complete,
                    compression=p1_complete, safe_input=safe_complete)
    for key, valid in complete.items():
        if not valid:
            warnings.append(f"Incomplete {key} specifications; aggregate capability is unknown")
    warnings.append("Power propagation uses small-signal gain; compression is a stage-limit estimate")
    if any(np.any(s['compression_headroom_db'] < 0) for s in stages):
        warnings.append("Predicted stage compression: propagated output power is unreliable")
    return BudgetResult(f, aggregate, stages, complete, assumptions, warnings)


def analyze_receiver(chain, frequency_hz, *, input_power_dbm=-100.0,
                     source_temperature_k=290.0, bandwidth_hz=None, passband_hz=None,
                     tone_power_dbm=None, state=None):
    """Selected-sideband budget. Tone power is per tone; input power is total."""
    if isinstance(chain, IQConversion):
        kwargs = dict(input_power_dbm=input_power_dbm, source_temperature_k=source_temperature_k,
                      bandwidth_hz=bandwidth_hz, passband_hz=passband_hz,
                      tone_power_dbm=tone_power_dbm, state=state)
        branches = {"i": analyze_receiver(chain.i_chain, frequency_hz, **kwargs),
                    "q": analyze_receiver(chain.q_chain, frequency_hz, **kwargs)}
        result = BudgetResult(_grid(frequency_hz), {}, [], {}, branches=branches)
        if chain.combining_convention is None:
            result.warnings.append("IQ branches reported separately; no combining convention supplied")
        else:
            if chain.noise_correlation != 0:
                raise ValueError("Power-sum combination requires explicit noise_correlation=0")
            a, b = (branches[k].aggregate for k in ("i", "q"))
            for key in ("output_power_dbm", "output_noise_dbm"):
                result.aggregate[key] = db(10 ** (a[key] / 10) + 10 ** (b[key] / 10))
            result.aggregate["snr_db"] = result.aggregate["output_power_dbm"] - result.aggregate["output_noise_dbm"]
            result.assumptions.append("IQ power sum uses explicitly uncorrelated branch noise")
        result.completeness = {k: all(r.completeness[k] for r in branches.values()) for k in branches['i'].completeness}
        return result
    if not np.isfinite(source_temperature_k) or source_temperature_k < 0:
        raise ValueError("Source temperature must be finite and nonnegative")
    if (bandwidth_hz is None) == (passband_hz is None):
        raise ValueError("Supply exactly one of bandwidth_hz or passband_hz")
    r = _analyze(chain, frequency_hz, input_power_dbm, state or OperatingState())
    a = r.aggregate
    noise_w_hz = K_B * (source_temperature_k + a['equivalent_noise_temperature_k'])
    a['input_noise_density_dbm_hz'] = db(noise_w_hz * 1000)
    if bandwidth_hz is not None:
        bw = _broadcast(bandwidth_hz, r.frequency_hz, "Bandwidth")
        if np.any(bw <= 0):
            raise ValueError("Bandwidth must be positive")
        input_noise_w = noise_w_hz * bw
        output_noise_w = input_noise_w * 10 ** (a['gain_db'] / 10)
        r.assumptions.append("ENBW calculation assumes locally constant gain and noise across bandwidth")
        for stage in r.stages:
            stage['signal_bandwidth_hz'] = bw.copy()
    else:
        lo, hi = passband_hz
        f = r.frequency_hz
        if len(f) < 2 or np.any(np.diff(f) <= 0) or not f[0] <= lo < hi <= f[-1]:
            raise ValueError("Passband requires a sorted frequency grid covering both edges")
        grid = np.concatenate(([lo], f[(f > lo) & (f < hi)], [hi]))
        integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        total = integrate(np.interp(grid, f, noise_w_hz * 10 ** (a['gain_db'] / 10)), grid)
        output_noise_w = np.full_like(f, total)
        input_noise_w = output_noise_w / 10 ** (a['gain_db'] / 10)
        r.assumptions.append("Passband is an ideal rectangular integration window on the input frequency grid")
        for stage in r.stages:
            stage['signal_bandwidth_hz'] = np.full_like(f, hi - lo)
    a['input_noise_dbm'] = db(input_noise_w * 1000)
    a['output_noise_dbm'] = db(output_noise_w * 1000)
    a['snr_db'] = a['output_power_dbm'] - a['output_noise_dbm']
    a['compression_dynamic_range_db'] = a['estimated_input_p1db_dbm'] - a['input_noise_dbm']
    a['sfdr3_db'] = (2 / 3) * (a['iip3_dbm'] - a['input_noise_dbm'])
    if tone_power_dbm is not None:
        tone = _broadcast(tone_power_dbm, r.frequency_hz, "Tone power")
        a['im3_input_dbm'] = 3 * tone - 2 * a['iip3_dbm']
        a['im3_output_dbm'] = a['im3_input_dbm'] + a['gain_db']
        r.assumptions.append("IM3 uses equal tones and a small-signal reciprocal-power IP3 cascade; tone phases are not modeled")
    if r.completeness['compression']:
        limits = np.stack([s['input_p1db_dbm'] - (s['cumulative_gain_db'] - s['gain_db']) for s in r.stages])
        a['compression_limiting_stage_index'] = np.argmin(limits, axis=0)
    return r


def analyze_transmitter(chain, frequency_hz, *, input_power_dbm, state=None):
    state = state or OperatingState('transmit')
    r = _analyze(chain, frequency_hz, input_power_dbm, state)
    r.aggregate['delivered_peak_power_dbm'] = r.aggregate['output_power_dbm'].copy()
    r.aggregate['average_output_power_dbm'] = r.aggregate['output_power_dbm'] + db(state.duty_cycle)
    r.assumptions.append("Average power assumes zero transmitted power between rectangular pulses")
    return r


def analyze_leakage(transmitter, coupling_path, receiver, frequency_hz, *, input_power_dbm, state=None):
    """Follow one explicitly ordered leakage path; isolation subtracts path gain."""
    state = state or OperatingState('transmit')
    # Isolation is an explicit loss at the coupling component input frequency.
    from .models import Component
    stages = []
    for c in coupling_path.components:
        if c.isolation_db is not None:
            iso = c.isolation_db
            values = iso.values if isinstance(iso, FrequencyTable) else iso
            if np.any(np.asarray(values) < 0):
                raise ValueError("Isolation must be nonnegative")
            loss = FrequencyTable(iso.frequency_hz, -iso.values, 'dB', iso.provenance) if isinstance(iso, FrequencyTable) else -iso
            stages.append(Component(c.name + '_isolation', loss, kind='passive'))
        stages.append(c)
    full = Chain([*transmitter.components, *stages, *receiver.components], 'leakage')
    r = _analyze(full, frequency_hz, input_power_dbm, state)
    rx_stages = r.stages[-len(receiver.components):]
    r.aggregate['receiver_input_leakage_dbm'] = rx_stages[0]['input_power_dbm']
    r.aggregate['leakage_compression_margin_db'] = np.min(np.stack([s['compression_headroom_db'] for s in rx_stages]), axis=0)
    r.aggregate['leakage_safe_input_margin_db'] = np.min(np.stack([s['safe_input_margin_db'] for s in rx_stages]), axis=0)
    r.completeness['receiver_protection'] = all(np.all(np.isfinite(s['safe_input_margin_db'])) for s in rx_stages)
    if state.recovery_time_s is None:
        r.warnings.append("Receiver recovery time is unknown; no dead-time prediction is available")
    else:
        r.assumptions.append(f"Supplied recovery time: {state.recovery_time_s:g} s; not simulated")
    return r
