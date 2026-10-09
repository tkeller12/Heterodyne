import numpy as np
import pytest

from heterodyne import (Attenuator, Component, FrequencyTable, Chain,
                        analyze_receiver, analyze_transmitter, analyze_leakage)


def receiver(*components):
    return analyze_receiver(Chain(list(components)), [34e9], bandwidth_hz=1e6)


def amplifier():
    return Component('amp', 20, noise_figure_db=3, iip3_dbm=10,
                     input_p1db_dbm=0, max_safe_input_dbm=20)


def test_loss_placement_and_linearity():
    cable = Attenuator('cable', 3, ideal_linear=True)
    before = receiver(cable, amplifier())
    after = receiver(amplifier(), cable)
    assert before.aggregate['gain_db'] == pytest.approx([17])
    assert before.aggregate['noise_figure_db'] == pytest.approx([6])
    factor = 10**0.3 + (10**0.3-1)/100
    assert after.aggregate['noise_figure_db'] == pytest.approx([10*np.log10(factor)])
    assert before.aggregate['iip3_dbm'] == pytest.approx([13])
    assert before.aggregate['estimated_input_p1db_dbm'] == pytest.approx([3])
    assert after.aggregate['iip3_dbm'] == pytest.approx([10])
    assert after.aggregate['estimated_input_p1db_dbm'] == pytest.approx([0])
    assert before.aggregate['compression_limiting_stage_index'] == [1]
    assert before.completeness['ip3'] and before.completeness['compression']
    assert not before.completeness['safe_input']
    assert 'explicitly ideal linear' in before.summary()


@pytest.mark.parametrize('temperature', [0, 145, 290, 580])
def test_thermal_noise(temperature):
    r = receiver(Attenuator('cable', 10, temperature_k=temperature))
    assert r.aggregate['noise_figure_db'] == pytest.approx([10*np.log10(1+9*temperature/290)])


def test_zero_loss_and_all_ideal_chain():
    r = receiver(Attenuator('cable', 0, ideal_linear=True))
    assert r.aggregate['gain_db'] == [0]
    assert r.aggregate['noise_figure_db'] == [0]
    assert np.isposinf(r.aggregate['iip3_dbm']).all()
    assert np.isnan(r.aggregate['compression_limiting_stage_index']).all()
    assert 'unknown' in r.summary()


def test_tables_csv_provenance_and_ranges(tmp_path):
    table = FrequencyTable([33e9, 35e9], [1, 3], 'dB', 'measured cable')
    cable = Attenuator('cable', table, operating_band_hz=(33e9, 35e9))
    r = receiver(cable)
    assert r.aggregate['gain_db'] == [-2]
    assert r.aggregate['noise_figure_db'] == pytest.approx([2])
    assert r.stages[0]['provenance']['gain_db'] == 'measured cable'
    assert r.stages[0]['provenance']['loss_db'] == 'measured cable'
    assert table.values == pytest.approx([1, 3])
    assert 'cable' in r.summary()
    r.to_csv(tmp_path/'budget.csv')
    assert 'cable.gain_db' in (tmp_path/'budget.csv').read_text()
    with pytest.raises(ValueError):
        cable.evaluate('gain_db', [36e9])
    with pytest.raises(ValueError):
        Attenuator('scalar', 1, operating_band_hz=(1e9, 2e9)).evaluate('gain_db', [34e9])


@pytest.mark.parametrize('loss', [-1, np.nan, np.inf, None, [1, 2]])
def test_invalid_loss(loss):
    with pytest.raises(ValueError):
        Attenuator('bad', loss)


def test_invalid_table_and_conflicts():
    for table in [FrequencyTable([1, 2], [1, 2], 'dBm'),
                  FrequencyTable([1, 2], [-1, 2], 'dB')]:
        with pytest.raises(ValueError):
            Attenuator('bad', table)
    for field in ['iip3_dbm', 'oip3_dbm', 'input_p1db_dbm', 'output_p1db_dbm']:
        with pytest.raises(ValueError):
            Attenuator('bad', 1, ideal_linear=True, **{field: 10})


def test_unknown_and_supplied_limits():
    unknown = receiver(Attenuator('cable', 1), amplifier())
    assert not unknown.completeness['ip3']
    assert not unknown.completeness['compression']
    assert np.isnan(unknown.aggregate['iip3_dbm']).all()
    specified = receiver(Attenuator('cable', 2, iip3_dbm=10,
                                    input_p1db_dbm=0, max_safe_input_dbm=20))
    assert all(specified.completeness.values())
    ideal = receiver(Attenuator('cable', 2, ideal_linear=True, max_safe_input_dbm=0))
    assert ideal.completeness['safe_input']
    assert np.isfinite(ideal.stages[0]['safe_input_margin_db']).all()


def test_transmitter_and_leakage():
    tx = Chain([Component('tx', 20), Attenuator('tx_cable', 2, ideal_linear=True)])
    coupling = Chain([Attenuator('coupling_loss', 30, ideal_linear=True)])
    rx = Chain([Attenuator('rx_cable', 1, ideal_linear=True), amplifier()])
    t = analyze_transmitter(tx, [34e9], input_power_dbm=0)
    assert t.aggregate['delivered_peak_power_dbm'] == [18]
    leak = analyze_leakage(tx, coupling, rx, [34e9], input_power_dbm=0)
    assert leak.aggregate['receiver_input_leakage_dbm'] == [-12]
    assert leak.stages[-1]['input_power_dbm'] == [-13]
    assert not leak.completeness['receiver_protection']
