import numpy as np
import pytest
from heterodyne import (Component, FrequencyTable, Chain, Mixer, OperatingState,
                        IQConversion, analyze_receiver, analyze_transmitter, analyze_leakage)


def amp(name='amp', **kwargs):
    specs = dict(gain_db=10, noise_figure_db=3, iip3_dbm=10,
                 input_p1db_dbm=0, max_safe_input_dbm=20)
    specs.update(kwargs)
    return Component(name, **specs)


def rx(components, **kwargs):
    return analyze_receiver(Chain(components), [34e9], bandwidth_hz=1e6, **kwargs)


def test_friis_and_gain():
    r = rx([amp('a'), amp('b', gain_db=20, noise_figure_db=6)])
    factor = 10**0.3 + (10**0.6 - 1)/10
    assert r.aggregate['gain_db'] == pytest.approx([30])
    assert r.aggregate['noise_figure_db'] == pytest.approx([10*np.log10(factor)])
    assert r.aggregate['output_noise_dbm'] == pytest.approx(
        [10*np.log10(1.380649e-23*290*factor*1e6*1000)+30])


@pytest.mark.parametrize('temperature', [0, 145, 290, 580])
def test_passive_temperature(temperature):
    r = rx([Component('loss', -10, kind='passive', temperature_k=temperature)])
    assert r.aggregate['noise_figure_db'] == pytest.approx([10*np.log10(1+9*temperature/290)])


def test_front_end_loss():
    r = rx([Component('loss', -3, kind='passive'), amp()])
    assert r.aggregate['noise_figure_db'] == pytest.approx([6])


def test_tables_and_csv(tmp_path):
    t = FrequencyTable([33e9, 35e9], [10, 20], 'dB', 'datasheet')
    t.to_csv(tmp_path/'spec.csv', 'gain_db')
    loaded = FrequencyTable.from_csv(tmp_path/'spec.csv', 'gain_db', 'dB')
    assert loaded.evaluate([34e9]) == pytest.approx([15])
    with pytest.raises(ValueError):
        t.evaluate([36e9])
    with pytest.raises(ValueError):
        FrequencyTable([1, 1], [2, 3], 'dB')
    r = rx([amp(gain_db=t)])
    r.to_csv(tmp_path/'budget.csv')
    assert 'amp.input_frequency_hz' in (tmp_path/'budget.csv').read_text()
    assert any('constant NF' in a for a in r.assumptions)


def test_unknowns_propagate():
    r = rx([Component('unknown', 10)])
    assert not any(r.completeness.values())
    for key in ['noise_figure_db', 'iip3_dbm', 'estimated_input_p1db_dbm']:
        assert np.isnan(r.aggregate[key]).all()


def test_ip3_compression_im3():
    r = rx([amp('a'), amp('b')], input_power_dbm=-20, tone_power_dbm=-30)
    ip3 = -10*np.log10(1/10+10/10)
    assert r.aggregate['iip3_dbm'] == pytest.approx([ip3])
    assert r.aggregate['im3_input_dbm'] == pytest.approx([-90-2*ip3])
    assert r.aggregate['estimated_input_p1db_dbm'] == pytest.approx([-10])
    assert r.aggregate['compression_limiting_stage_index'] == [1]
    converted = rx([amp(iip3_dbm=None, oip3_dbm=20,
                         input_p1db_dbm=None, output_p1db_dbm=9)])
    assert converted.aggregate['iip3_dbm'] == pytest.approx([10])
    assert converted.aggregate['estimated_input_p1db_dbm'] == pytest.approx([0])


def test_sfdr_bandwidth():
    c = Chain([amp()])
    a = analyze_receiver(c, [34e9], bandwidth_hz=1e6)
    b = analyze_receiver(c, [34e9], bandwidth_hz=1e7)
    assert a.aggregate['sfdr3_db']-b.aggregate['sfdr3_db'] == pytest.approx([20/3])


def test_conversion_and_dsb():
    m = Mixer('m', -6, noise_figure_db=8, lo_hz=33e9, noise_convention='SSB')
    r = rx([m, amp('if', operating_band_hz=(0.5e9, 1.5e9))])
    assert r.stages[1]['input_frequency_hz'] == [1e9]
    m.noise_convention = 'DSB'
    assert not rx([m]).completeness['noise']
    lower = Mixer('lower', -6, lo_hz=35e9, sideband='lower', noise_convention='SSB')
    assert lower.output_frequency([34e9]) == [1e9]
    up = Mixer('up', -6, lo_hz=33e9, conversion='up', noise_convention='SSB')
    assert up.output_frequency([1e9]) == [34e9]
    with pytest.raises(ValueError):
        m.output_frequency([32e9])


def test_passband_integration():
    c = Chain([amp()])
    a = analyze_receiver(c, [33e9, 34e9, 35e9], passband_hz=(33.5e9, 34.5e9))
    b = analyze_receiver(c, [34e9], bandwidth_hz=1e9)
    assert a.aggregate['output_noise_dbm'] == pytest.approx(np.repeat(b.aggregate['output_noise_dbm'], 3))


def test_iq():
    c = Chain([amp()])
    separate = analyze_receiver(IQConversion(c, c), [34e9], bandwidth_hz=1e6)
    assert set(separate.branches) == {'i', 'q'}
    assert separate.aggregate == {}
    with pytest.raises(ValueError):
        analyze_receiver(IQConversion(c, c, 'power_sum'), [34e9], bandwidth_hz=1e6)
    combined = analyze_receiver(IQConversion(c, c, 'power_sum', 0), [34e9], bandwidth_hz=1e6)
    assert combined.aggregate['snr_db'] == pytest.approx(separate.branches['i'].aggregate['snr_db'])


def test_transmit_and_leakage_states():
    tx = Chain([amp('tx', gain_db=30)])
    coupling = Chain([Component('coupling', -1, isolation_db=30, kind='passive')])
    receiver = Chain([amp('rx', state_gain_db={'transmit': -40})])
    t = analyze_transmitter(tx, [34e9], input_power_dbm=-10, state=OperatingState('transmit', .01))
    assert t.aggregate['average_output_power_dbm'] == pytest.approx([0])
    receive = analyze_leakage(tx, coupling, receiver, [34e9], input_power_dbm=-10)
    transmit = analyze_leakage(tx, coupling, receiver, [34e9], input_power_dbm=-10,
                               state=OperatingState('receive'))
    assert receive.aggregate['receiver_input_leakage_dbm'] == pytest.approx([-11])
    assert transmit.aggregate['output_power_dbm'] - receive.aggregate['output_power_dbm'] == pytest.approx([50])
    assert receive.completeness['receiver_protection']
    unknown = analyze_leakage(tx, coupling, Chain([Component('unknown', 0)]), [34e9], input_power_dbm=-10)
    assert not unknown.completeness['receiver_protection']
    assert np.isnan(unknown.aggregate['leakage_safe_input_margin_db']).all()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        OperatingState(duty_cycle=2)
    with pytest.raises(ValueError):
        rx([amp(operating_band_hz=(1e9, 2e9))])
    with pytest.raises(ValueError):
        rx([amp()], source_temperature_k=-1)
    with pytest.raises(ValueError):
        analyze_receiver(Chain([amp()]), [34e9], bandwidth_hz=0)


def test_noise_temperature_and_im3_scaling():
    c = Chain([amp(noise_figure_db=0)])
    a = analyze_receiver(c, [34e9], bandwidth_hz=1e6,
                         source_temperature_k=145, tone_power_dbm=-30)
    b = analyze_receiver(c, [34e9], bandwidth_hz=1e6,
                         source_temperature_k=290, tone_power_dbm=-29)
    assert b.aggregate['input_noise_dbm'] - a.aggregate['input_noise_dbm'] == pytest.approx([10*np.log10(2)])
    assert b.aggregate['im3_input_dbm'] - a.aggregate['im3_input_dbm'] == pytest.approx([3])


def test_variable_gain_noise_passband():
    gain = FrequencyTable([1e9, 2e9], [0, 10], 'dB')
    nf = FrequencyTable([1e9, 2e9], [0, 3], 'dB')
    r = analyze_receiver(Chain([amp(gain_db=gain, noise_figure_db=nf)]),
                         [1e9, 1.5e9, 2e9], passband_hz=(1e9, 2e9))
    density = 1.380649e-23*290*10**(np.array([0, 1.5, 3])/10)*10**(np.array([0, 5, 10])/10)
    total = (density[0]+2*density[1]+density[2])*0.25e9
    assert r.aggregate['output_noise_dbm'] == pytest.approx(np.repeat(10*np.log10(total*1000), 3))


def test_damage_is_distinct_from_compression():
    r = rx([amp()], input_power_dbm=5)
    assert r.stages[0]['compression_headroom_db'] == [-5]
    assert r.stages[0]['safe_input_margin_db'] == [15]
    assert any('unreliable' in warning for warning in r.warnings)
    with pytest.raises(ValueError):
        Mixer('undeclared', -6, lo_hz=33e9)
