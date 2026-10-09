"""Illustrative specifications only: these are not hardware recommendations."""
import numpy as np
from heterodyne import Component, Attenuator, FrequencyTable, Chain, Mixer, OperatingState
from heterodyne import analyze_receiver, analyze_transmitter, analyze_leakage


def main():
    f = np.linspace(33.5e9, 34.5e9, 101)
    band = (33e9, 35e9)
    assumed = {'all': 'Illustrative user assumptions; replace with datasheet values'}
    rx = Chain([
        Attenuator('receiver_cable_connectors', 1.0, ideal_linear=True,
                   operating_band_hz=band, provenance=assumed),
        Component('receive_switch', -1.5, kind='switch', operating_band_hz=band,
                  state_gain_db={'transmit': -50}, iip3_dbm=60,
                  input_p1db_dbm=30, max_safe_input_dbm=35, provenance=assumed),
        Component('lna', FrequencyTable([33e9, 35e9], [25, 23], 'dB'),
                  noise_figure_db=3, iip3_dbm=-5, input_p1db_dbm=-15,
                  max_safe_input_dbm=10, operating_band_hz=band, provenance=assumed),
        Mixer('downconverter', -8, noise_figure_db=9, iip3_dbm=15,
              input_p1db_dbm=5, max_safe_input_dbm=15, lo_hz=33e9,
              noise_convention='SSB', operating_band_hz=band, provenance=assumed),
        Component('if_amplifier', 35, noise_figure_db=4, iip3_dbm=0,
                  input_p1db_dbm=-10, max_safe_input_dbm=10,
                  operating_band_hz=(0.1e9, 2e9), provenance=assumed),
    ], 'Q-band receiver')
    tx = Chain([Component('power_amplifier', 30, input_p1db_dbm=5,
                          operating_band_hz=band, provenance=assumed),
                Attenuator('transmitter_cable_connectors', 1.5, ideal_linear=True,
                           operating_band_hz=band, provenance=assumed)], 'Q-band transmitter')
    coupling = Chain([Component('circulator_coupling', -1, isolation_db=30,
                               kind='passive', provenance=assumed)])
    receiver = analyze_receiver(rx, f, bandwidth_hz=20e6, input_power_dbm=-90,
                                tone_power_dbm=-70)
    transmitter = analyze_transmitter(tx, f, input_power_dbm=0,
                                      state=OperatingState('transmit', duty_cycle=0.01))
    leakage = analyze_leakage(tx, coupling, rx, f, input_power_dbm=0,
                              state=OperatingState('transmit', duty_cycle=0.01))
    for label, result in [('receiver', receiver), ('transmitter', transmitter), ('leakage', leakage)]:
        result.to_csv(label + '_budget.csv')
        print(f'\n{label.upper()}')
        result.print_summary()


if __name__ == '__main__':
    main()
