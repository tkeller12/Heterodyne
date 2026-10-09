import io

import pytest

from heterodyne import Component, Chain, IQConversion, analyze_receiver


def result():
    return analyze_receiver(Chain([Component('LNA', 20, noise_figure_db=3)]),
                            [33e9, 34e9, 35e9], bandwidth_hz=1e6)


def test_report_frequency_units_and_unknowns():
    r = result()
    report = r.summary()
    assert '34.000000 GHz (point 2 of 3)' in report
    assert '33.000000 to 35.000000 GHz' in report
    assert '20.000 dB' in report
    assert 'dBm/Hz' in report
    assert 'unknown' in report
    assert 'Noise: complete' in report
    assert 'Compression: incomplete' in report
    assert 'Warnings:' in report
    assert 'Stage details' in report
    assert 'Safe margin dB' in report
    assert str(r) == report
    assert '33.000000 GHz (point 1 of 3)' in r.summary(0)
    assert 'Stage details' not in r.summary(include_stages=False)


def test_report_stream_and_stdout(capsys):
    r = result()
    stream = io.StringIO()
    r.print_summary(file=stream)
    assert stream.getvalue() == r.summary() + '\n'
    r.print_summary()
    assert capsys.readouterr().out == stream.getvalue()


@pytest.mark.parametrize('index', [-1, 3, 0.5])
def test_invalid_report_point(index):
    with pytest.raises(ValueError, match='frequency_index'):
        result().summary(index)


def test_iq_branch_reports_and_limiting_stage():
    chain = Chain([Component('LNA', 20, noise_figure_db=3, input_p1db_dbm=-10)])
    r = analyze_receiver(IQConversion(chain, chain), [34e9], bandwidth_hz=1e6)
    report = r.summary()
    assert 'I branch:' in report and 'Q branch:' in report
    assert 'Compression limiting stage index : LNA' in report
