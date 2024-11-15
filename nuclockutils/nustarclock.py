import glob
import os
import shutil
from functools import lru_cache

import numpy as np
from astropy.table import Table, vstack
import pandas as pd
from astropy.time import Time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from .utils import NUSTAR_MJDREF, splitext_improved, sec_to_mjd
from .utils import filter_with_region, fix_byteorder, rolling_std
from .utils import measure_overall_trend, cross_two_gtis, get_rough_trend_fun
from .utils import spline_through_data, cubic_interpolation, robust_poly_fit
from .utils import get_temperature_parameters
from astropy.io import fits
import tqdm
from astropy import log
from statsmodels.robust import mad
import copy
import holoviews as hv
from holoviews.operation.datashader import datashade
from holoviews import opts
# import matplotlib.pyplot as plt

hv.extension('bokeh')

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def get_bad_points_db(db_file='BAD_POINTS_DB.dat'):
    if not os.path.exists(db_file):
        db_file = os.path.join(datadir, 'BAD_POINTS_DB.dat')

    return np.genfromtxt(db_file, dtype=np.longdouble)


def flag_bad_points(all_data, db_file='BAD_POINTS_DB.dat'):
    """

    Examples
    --------
    >>> db_file = 'dummy_bad_points.dat'
    >>> np.savetxt(db_file, np.array([-1, 3, 10]))
    >>> all_data = Table({'met': [0, 1, 2, 3, 4]})
    >>> all_data = flag_bad_points(all_data, db_file='dummy_bad_points.dat')
    INFO: ...
    >>> np.all(all_data['flag'] == [False, False, False, True, False])
    True
    """
    if not os.path.exists(db_file):
        return all_data
    log.info("Flagging bad points...")

    intv = [all_data['met'][0] - 0.5, all_data['met'][-1] + 0.5]
    ALL_BAD_POINTS = np.genfromtxt(db_file)
    ALL_BAD_POINTS.sort()
    ALL_BAD_POINTS = np.unique(ALL_BAD_POINTS)
    ALL_BAD_POINTS = ALL_BAD_POINTS[
        (ALL_BAD_POINTS > intv[0]) & (ALL_BAD_POINTS < intv[1])]

    idxs = all_data['met'].searchsorted(ALL_BAD_POINTS)

    if 'flag' in all_data.colnames:
        mask = np.array(all_data['flag'], dtype=bool)
    else:
        mask = np.zeros(len(all_data), dtype=bool)

    for idx in idxs:
        if idx >= mask.size:
            continue
        mask[idx] = True
    all_data['flag'] = mask
    return all_data


def find_good_time_intervals(temperature_table,
                             clock_jump_times=None):
    start_time = temperature_table['met'][0]
    stop_time = temperature_table['met'][-1]

    clock_gtis = no_jump_gtis(
        start_time, stop_time, clock_jump_times)

    if not 'gti' in temperature_table.meta:
        temp_gtis = temperature_gtis(temperature_table)
    else:
        temp_gtis = temperature_table.meta['gti']

    gtis = cross_two_gtis(temp_gtis, clock_gtis)

    return gtis


def calculate_stats(all_data):
    log.info("Calculating statistics")
    r_std = residual_roll_std(all_data['residual_detrend'])

    scatter = mad(all_data['residual_detrend'])
    print()
    print("----------------------------- Stats -----------------------------------")
    print()
    print(f"Overall MAD: {scatter * 1e6:.0f} us")
    print(f"Minimum scatter: ±{np.min(r_std) * 1e6:.0f} us")
    print()
    print("-----------------------------------------------------------------------")


def load_and_flag_clock_table(clockfile="latest_clock.dat", shift_non_malindi=False):
    clock_offset_table = load_clock_offset_table(clockfile,
                                                 shift_non_malindi=shift_non_malindi)
    clock_offset_table = flag_bad_points(
        clock_offset_table, db_file='BAD_POINTS_DB.dat')
    return clock_offset_table


def spline_detrending(clock_offset_table, temptable, outlier_cuts=None,
                      fixed_control_points=None):
    tempcorr_idx = np.searchsorted(temptable['met'], clock_offset_table['met'])
    tempcorr_idx[tempcorr_idx >= temptable['met'].size] = \
        temptable['met'].size - 1
    clock_residuals = \
        np.array(clock_offset_table['offset'] -
                 temptable['temp_corr'][tempcorr_idx])

    clock_mets = clock_offset_table['met']

    if outlier_cuts is not None:
        log.info("Cutting outliers...")
        better_points = np.array(clock_residuals == clock_residuals,
                                 dtype=bool)

        for i, cut in enumerate(outlier_cuts):
            mm = median_filter(clock_residuals, 11)
            wh = ((clock_residuals[better_points] - mm[better_points]) < outlier_cuts[
                i]) | ((clock_residuals[better_points] - mm[better_points]) <
                       outlier_cuts[0])
            better_points[better_points] = ~wh
        # Eliminate too recent flags, in the last month of solution.
        one_month = 86400 * 30
        do_not_flag = clock_mets > clock_mets.max() - one_month
        better_points[do_not_flag] = True

        clock_offset_table = clock_offset_table[better_points]
        clock_residuals = clock_residuals[better_points]

    detrend_fun = spline_through_data(
        clock_offset_table['met'], clock_residuals, downsample=20,
        fixed_control_points=fixed_control_points)

    r_std = residual_roll_std(
        clock_residuals - detrend_fun(clock_offset_table['met']))

    clidx = np.searchsorted(clock_offset_table['met'], temptable['met'])
    clidx[clidx == clock_offset_table['met'].size] = \
        clock_offset_table['met'].size - 1

    temptable['std'] = r_std[clidx]
    temptable['temp_corr_trend'] = detrend_fun(temptable['met'])
    temptable['temp_corr_detrend'] = \
        temptable['temp_corr'] + temptable['temp_corr_trend']

    return temptable


def eliminate_trends_in_residuals(temp_table, clock_offset_table,
                                  gtis, debug=False,
                                  fixed_control_points=None):

    # good = clock_offset_table['met'] < np.max(temp_table['met'])
    # clock_offset_table = clock_offset_table[good]
    temp_table['temp_corr_raw'] = temp_table['temp_corr']

    tempcorr_idx = np.searchsorted(temp_table['met'],
                                   clock_offset_table['met'])
    tempcorr_idx[tempcorr_idx == temp_table['met'].size] = \
        temp_table['met'].size - 1

    clock_residuals = \
        clock_offset_table['offset'] - temp_table['temp_corr'][tempcorr_idx]

    # Only use for interpolation Malindi points; however, during the Malindi
    # problem in 2013, use the other data for interpolation but subtracting
    # half a millisecond

    use_for_interpol, bad_malindi_time = \
        get_malindi_data_except_when_out(clock_offset_table)

    clock_residuals[bad_malindi_time] -= 0.0005

    good = (clock_residuals == clock_residuals) & ~clock_offset_table['flag'] & use_for_interpol

    clock_offset_table = clock_offset_table[good]
    clock_residuals = clock_residuals[good]

    for g in gtis:
        log.info(f"Treating data from METs {g[0]}--{g[1]}")
        start, stop = g

        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)

        if cl_idx_end - cl_idx_start == 0:
            continue

        temp_idx_start, temp_idx_end = \
            np.searchsorted(temp_table['met'], g)

        table_new = temp_table[temp_idx_start:temp_idx_end]
        cltable_new = clock_offset_table[cl_idx_start:cl_idx_end]
        met = cltable_new['met']

        if len(met) < 2:
            continue

        residuals = clock_residuals[cl_idx_start:cl_idx_end]
        met0 = met[0]
        met_rescale = (met - met0)/(met[-1] - met0)
        _, m, q = measure_overall_trend(met_rescale, residuals)
        # p_new = get_rough_trend_fun(met, residuals)
        #
        # if p_new is not None:
        #     p = p_new
        poly_order = min(met.size // 300 + 1, 2)
        p0 = np.zeros(poly_order + 1)
        p0[0] = q
        if p0.size > 1:
            p0[1] = m
        log.info(f"Fitting a polinomial of order {poly_order}")
        p = robust_poly_fit(met_rescale, residuals, order=poly_order,
                            p0=p0)
        # if poly_order >=2:
        import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(met_rescale, residuals)
        # plt.plot(met_rescale, p(met_rescale))
        # plt.plot(met_rescale, m * (met_rescale) + q)

        table_mets_rescale = (table_new['met'] - met0) / (met[-1] - met0)
        corr = p(table_mets_rescale)

        sub_residuals = residuals - p(met_rescale)
        m = (sub_residuals[-1] - sub_residuals[0]) / (met_rescale[-1] - met_rescale[0])
        q = sub_residuals[0]

        # plt.plot(table_mets_rescale, corr + m * (table_mets_rescale - met_rescale[0]) + q, lw=2)

        corr = corr + m * (table_mets_rescale - met_rescale[0]) + q
        table_new['temp_corr'] += corr

        if debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(table_new['met'], table_new['temp_corr'], alpha=0.5)
            plt.scatter(cltable_new['met'], cltable_new['offset'])
            plt.plot(table_new['met'], table_new['temp_corr'])
            plt.savefig(f'{int(start)}--{int(stop)}_detr.png')
            plt.close(fig)
        # plt.show()

        # print(f'df/f = {(p(stop) - p(start)) / (stop - start)}')

    bti_list = [[g0, g1] for g0, g1 in zip(gtis[:-1, 1], gtis[1:, 0])]
    bti_list += [[gtis[-1, 1], clock_offset_table['met'][-1] + 10]]
    btis = np.array(bti_list)

    # Interpolate the solution along bad time intervals
    for g in btis:
        start, stop = g
        log.info(f"Treating bad data from METs {start}--{stop}")

        temp_idx_start, temp_idx_end = \
            np.searchsorted(temp_table['met'], g)
        if temp_idx_end - temp_idx_start == 0 and \
                temp_idx_end < len(temp_table):
            continue
        table_new = temp_table[temp_idx_start:temp_idx_end]
        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)
        local_clockoff = clock_offset_table[cl_idx_start - 1:cl_idx_end + 1]
        clock_off = local_clockoff['offset']
        clock_tim = local_clockoff['met']

        last_good_tempcorr = temp_table['temp_corr'][temp_idx_start - 1]
        last_good_time = temp_table['met'][temp_idx_start - 1]
        if temp_idx_end < temp_table['temp_corr'].size:
            next_good_tempcorr = temp_table['temp_corr'][temp_idx_end + 1]
            next_good_time = temp_table['met'][temp_idx_end + 1]
            clock_off = np.concatenate(
                ([last_good_tempcorr], clock_off, [next_good_tempcorr]))
            clock_tim = np.concatenate(
                ([last_good_time], clock_tim, [next_good_time]))
        else:
            clock_off = np.concatenate(
                ([last_good_tempcorr], clock_off))
            clock_tim = np.concatenate(
                ([last_good_time], clock_tim))

            next_good_tempcorr = clock_off[-1]
            next_good_time = clock_tim[-1]

        if cl_idx_end - cl_idx_start < 2:
            log.info("Not enough good clock measurements. Interpolating")
            m = (next_good_tempcorr - last_good_tempcorr) / \
              (next_good_time - last_good_time)
            q = last_good_tempcorr
            table_new['temp_corr'][:] = \
                q + (table_new['met'] - last_good_time) * m
            continue

        order = np.argsort(clock_tim)

        clock_off_fun = interp1d(
            clock_tim[order], clock_off[order], kind='linear',
            assume_sorted=True)
        table_new['temp_corr'][:] = clock_off_fun(table_new['met'])

    log.info("Final detrending...")

    table_new = spline_detrending(
        clock_offset_table, temp_table,
        outlier_cuts=[-0.002, -0.001],
        fixed_control_points=fixed_control_points)

    return table_new


def residual_roll_std(residuals, window=30):
    """

    Examples
    --------
    >>> residuals = np.zeros(5000)
    >>> residuals[:4000] = np.random.normal(0, 1, 4000)
    >>> roll_std = residual_roll_std(residuals, window=500)
    >>> np.allclose(roll_std[:3500], 1., rtol=0.2)
    True
    >>> np.all(roll_std[4500:] == 0.)
    True
    """
    r_std = rolling_std(residuals, window)
    # r_std = rolling_std(np.diff(residuals), window) / np.sqrt(2)
    # return np.concatenate(([r_std[:1], r_std]))
    return r_std


def get_malindi_data_except_when_out(clock_offset_table):
    """Select offset measurements from Malindi, unless Malindi is out.

    In the time interval between METs 93681591 and 98051312, Malindi was out
    of work. For that time interval, we use all clock offset measurements
    available. In all other cases, we just use Malindi

    Parameters
    ----------
    clock_offset_table : :class:`Table` object
        Table containing the clock offset measurements. At least, it has to
        contain a 'met' and a 'station' columns.

    Returns
    -------
    use_for_interpol : array of ``bool``
        Mask of "trustworthy" time measurements
    bad_malindi_time : array of ``bool``
        Mask of time measurements during Malindi outage

    Example
    -------
    >>> clocktable = Table({'met': [93681592, 1e8, 1.5e8],
    ...                     'station': ['SNG', 'MLD', 'UHI']})
    >>> ufp, bmt = get_malindi_data_except_when_out(clocktable)
    >>> assert np.all(ufp == [True, True, False])
    >>> assert np.all(bmt == [True, False, False])
    """
    # Covers 2012/12/20 - 2013/02/08 Malindi outage
    # Also covers 2021/04/28 - 2021/05/06 issues with Malindi clock

    no_malindi_intvs = [[93681591, 98051312],[357295300, 357972500]]
    clock_mets = clock_offset_table['met']

    bad_malindi_time = np.zeros(len(clock_mets), dtype=bool)
    for nmi in no_malindi_intvs:
        bad_malindi_time = bad_malindi_time | (clock_mets >= nmi[0]) & (
                    clock_mets < nmi[1])

    malindi_stn = clock_offset_table['station'] == 'MLD'
    use_for_interpol = \
        (malindi_stn | bad_malindi_time)

    return use_for_interpol, bad_malindi_time


def _look_for_temptable():
    """
    Look for the default temperature table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_temptable() # doctest: +ELLIPSIS
    ...
    >>> tempt.endswith('tp_eps_ceu_txco_tmp.csv')
    True
    """
    name = 'tp_eps_ceu_txco_tmp.csv'
    fullpath = os.path.join(datadir, name)

    if not os.path.exists(fullpath):
        import shutil
        import subprocess as sp
        sp.check_call('wget --no-check-certificate https://www.dropbox.com/s/spkn4v018m5fvkf/tp_eps_ceu_txco_tmp.csv?dl=0 -O bu.csv'.split(" "))

        shutil.copyfile('bu.csv', fullpath)
    return fullpath


def _look_for_clock_offset_file():
    """
    Look for the default clock offset table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_clock_offset_file()
    >>> os.path.basename(tempt).startswith('nustar_clock_offsets')
    True
    """
    name = 'nustar_clock_offsets*.dat'
    clockoff_files = sorted(glob.glob(os.path.join(datadir, name)))

    assert len(clockoff_files) > 0, \
        ("Clock offset file not found. Have you run get_data.sh in "
         "the data directory?")

    return clockoff_files[-1]


def _look_for_freq_change_file():
    """
    Look for the default frequency change table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_clock_offset_file()
    >>> os.path.basename(tempt).startswith('nustar_clock_offsets')
    True
    """
    name = 'nustar_freq_changes*.dat'
    fchange_files = sorted(glob.glob(os.path.join(datadir, name)))

    assert len(fchange_files) > 0, \
        ("Frequency change file not found. Have you run get_data.sh in "
         "the data directory?")

    return fchange_files[-1]


def read_clock_offset_table(clockoffset_file=None, shift_non_malindi=False):
    """
    Parameters
    ----------
    clockoffset_file : str
        e.g. 'nustar_clock_offsets-2018-10-30.dat'

    Returns
    -------
    clock_offset_table : `astropy.table.Table` object
    """
    if clockoffset_file is None:
        clockoffset_file = _look_for_clock_offset_file()
    log.info(f"Reading clock offsets from {clockoffset_file}")
    clock_offset_table = Table.read(clockoffset_file,
                                    format='csv', delimiter=' ',
                                    names=['uxt', 'met', 'offset', 'divisor',
                                           'station'])
    if shift_non_malindi:
        log.info("Shifting non-Malindi clock offsets down by 0.5 ms")
        all_but_malindi = clock_offset_table['station'] != 'MLD'
        clock_offset_table['offset'][all_but_malindi] -= 0.0005
    clock_offset_table['mjd'] = sec_to_mjd(clock_offset_table['met'])
    clock_offset_table.remove_row(len(clock_offset_table) - 1)
    clock_offset_table['flag'] = np.zeros(len(clock_offset_table), dtype=bool)

    log.info("Flagging bad points...")
    ALL_BAD_POINTS = get_bad_points_db()

    for b in ALL_BAD_POINTS:
        nearest = np.argmin(np.abs(clock_offset_table['met'] - b))
        if np.abs(clock_offset_table['met'][nearest] - b) < 1:
            clock_offset_table['flag'][nearest] = True

    return clock_offset_table


FREQ_CHANGE_DB= {"delete": [77509247, 78720802],
                 "add": [(1023848462, 77506869, 24000340),
                         (1025060017, 78709124, 24000337),
                         (0, 102576709, 24000339),
                         (1051488464, 105149249, 24000336),
                         (0, 182421890, 24000334),
                         (1157021125, 210681910, 24000337),
                         ( 0, 215657278, 24000333),
                         (0, 215794126, 24000328),
                         (1174597307, 228258092, 24000334),
                         (1174759273, 228420058, 24000334)]}


def no_jump_gtis(start_time, stop_time, clock_jump_times=None):
    """
    Examples
    --------
    >>> gtis = no_jump_gtis(0, 3, [1, 1.1])
    >>> np.allclose(gtis, [[0, 1], [1, 1.1], [1.1, 3]])
    True
    >>> gtis = no_jump_gtis(0, 3)
    >>> np.allclose(gtis, [[0, 3]])
    True
    """
    if clock_jump_times is None:
        return [[start_time, stop_time]]

    clock_gtis = []
    current_start = start_time
    for jump in clock_jump_times:
        clock_gtis.append([current_start, jump])
        current_start = jump
    clock_gtis.append([current_start, stop_time])
    clock_gtis = np.array(clock_gtis)
    return clock_gtis


def temperature_gtis(temperature_table, max_distance=600):
    """

    Examples
    --------
    >>> temperature_table = Table({'met': [0, 1, 2, 10, 11, 12]})
    >>> gti = temperature_gtis(temperature_table, 5)
    >>> np.allclose(gti, [[0, 2], [10, 12]])
    True
    >>> temperature_table = Table({'met': [-10, 0, 1, 2, 10, 11, 12, 20]})
    >>> gti = temperature_gtis(temperature_table, 5)
    >>> np.allclose(gti, [[0, 2], [10, 12]])
    True
    """
    temp_condition = np.concatenate(
        ([False], np.diff(temperature_table['met']) > max_distance, [False]))

    temp_edges_l = np.concatenate((
        [temperature_table['met'][0]],
        temperature_table['met'][temp_condition[:-1]]))

    temp_edges_h = np.concatenate((
        [temperature_table['met'][temp_condition[1:]],
         [temperature_table['met'][-1]]]))

    temp_gtis = np.array(list(zip(
        temp_edges_l, temp_edges_h)))

    length = temp_gtis[:, 1] - temp_gtis[:, 0]
    return temp_gtis[length > 0]


def read_freq_changes_table(freqchange_file=None, filter_bad=True):
    """Read the table with the list of commanded divisor frequencies.

    Parameters
    ----------
    freqchange_file : str
        e.g. 'nustar_freq_changes-2018-10-30.dat'

    Returns
    -------
    freq_changes_table : `astropy.table.Table` object
    """
    if freqchange_file is None:
        freqchange_file = _look_for_freq_change_file()
    log.info(f"Reading frequency changes from {freqchange_file}")
    freq_changes_table = Table.read(freqchange_file,
                                format='csv', delimiter=' ',
                                comment="\s*#",
                                names=['uxt', 'met', 'divisor'])
    log.info("Correcting known bad frequency points")
    for time in FREQ_CHANGE_DB['delete']:
        bad_time_idx = freq_changes_table['met'] == time
        freq_changes_table[bad_time_idx] = [0, 0, 0]
    for line in FREQ_CHANGE_DB['add']:
        freq_changes_table.add_row(line)
    freq_changes_table = freq_changes_table[freq_changes_table['met'] > 0]
    freq_changes_table.sort('met')

    freq_changes_table['mjd'] = sec_to_mjd(freq_changes_table['met'])
    freq_changes_table.remove_row(len(freq_changes_table) - 1)
    freq_changes_table['flag'] = \
        np.abs(freq_changes_table['divisor'] - 2.400034e7) > 20
    if filter_bad:
        freq_changes_table = freq_changes_table[~freq_changes_table['flag']]

    return freq_changes_table


def _filter_table(tablefile, start_date=None, end_date=None, tmpfile='tmp.csv'):
    try:
        from datetime import timezone
    except ImportError:
        # Python 2
        import pytz as timezone
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999

    start_date = Time(start_date, format='mjd', scale='utc')
    start_str = start_date.to_datetime(timezone=timezone.utc).strftime('%Y:%j')

    start_yr, start_day = [float(n) for n in start_str.split(':')]

    end_date = Time(end_date, format='mjd', scale='utc')
    stop_str = end_date.to_datetime(timezone=timezone.utc).strftime('%Y:%j')
    new_str = ""
    with open(tablefile) as fobj:
        before = True
        for i, l in enumerate(fobj.readlines()):
            if i == 0:
                new_str += l
                continue
            l = l.strip()
            # Now, let's check if the start date is before the start of the
            # clock file. It's sufficient to do it for the first 2-3 line(s)
            # (3 if there are units)
            if i <=2:
                try:
                    yr, day = [float(n) for n in l.split(':')[:2]]
                except ValueError:
                    continue
                if start_yr <= yr and start_day <= day:
                    before = False
            if l.startswith(start_str) and before is True:
                before = False
            if before is False:
                new_str += l + "\n"
            if l.startswith(stop_str):
                break

    if new_str == "":
        raise ValueError(f"No temperature information is available for the "
                         "wanted time range in {temperature_file}")
    with open(tmpfile, "w") as fobj:
        print(new_str, file=fobj)

    return tmpfile


def read_csv_temptable(mjdstart=None, mjdstop=None, temperature_file=None):
    if mjdstart is not None or mjdstop is not None:
        mjdstart_use = mjdstart
        mjdstop_use = mjdstop
        if mjdstart is not None:
            mjdstart_use -= 10
        if mjdstop is not None:
            mjdstop_use += 10
        log.info("Filtering table...")
        tmpfile = _filter_table(temperature_file,
                                start_date=mjdstart_use,
                                end_date=mjdstop_use, tmpfile='tmp.csv')
        log.info("Done")
    else:
        tmpfile = temperature_file

    temptable = Table.read(tmpfile)
    temptable.remove_row(0)
    log.info("Converting times (it'll take a while)...")
    times_mjd = Time(temptable["Time"], scale='utc', format="yday",
                     in_subfmt="date_hms").mjd
    log.info("Done.")
    temptable["mjd"] = np.array(times_mjd)
    temptable['met'] = (temptable["mjd"] - NUSTAR_MJDREF) * 86400
    temptable.remove_column('Time')
    temptable.rename_column('tp_eps_ceu_txco_tmp', 'temperature')
    temptable["temperature"] = np.array(temptable["temperature"], dtype=float)
    if os.path.exists('tmp.csv'):
        os.unlink('tmp.csv')

    return temptable


def read_saved_temptable(mjdstart=None, mjdstop=None,
                         temperature_file='temptable.hdf5'):
    table = Table.read(temperature_file)
    if mjdstart is None and mjdstop is None:
        return table

    if 'mjd' not in table.colnames:
        table["mjd"] = sec_to_mjd(table['met'])

    if mjdstart is None:
        mjdstart = table['mjd'][0]
    if mjdstop is None:
        mjdstop = table['mjd'][-1]

    good = (table['mjd'] >= mjdstart - 10)&(table['mjd'] <= mjdstop + 10)
    if not np.any(good):
        raise ValueError(f"No temperature information is available for the "
                         "wanted time range in {temperature_file}")
    return table[good]


def read_fits_temptable(temperature_file):
    with fits.open(temperature_file) as hdul:
        temptable = Table.read(hdul['ENG_0x133'])
        temptable.rename_column('TIME', 'met')
        temptable.rename_column('sc_clock_ext_tmp', 'temperature')
        for col in temptable.colnames:
            if 'chu' in col:
                temptable.remove_column(col)
        temptable["mjd"] = sec_to_mjd(temptable['met'])
    return temptable


def interpolate_temptable(temptable, dt=10):
    time = temptable['met']
    temperature = temptable['temperature']
    new_times = np.arange(time[0], time[-1], dt)
    idxs = np.searchsorted(time, new_times)
    return Table({'met': new_times, 'temperature': temperature[idxs]})


def read_temptable(temperature_file=None, mjdstart=None, mjdstop=None,
                   dt=None, gti_tolerance=600):
    if temperature_file is None:
        temperature_file = _look_for_temptable()
    log.info(f"Reading temperature_information from {temperature_file}")
    ext = splitext_improved(temperature_file)[1]
    if ext in ['.csv']:
        temptable = read_csv_temptable(mjdstart, mjdstop, temperature_file)
    elif ext in ['.hk', '.hk.gz']:
        temptable = read_fits_temptable(temperature_file)
    elif ext in ['.hdf5', '.h5']:
        temptable = read_saved_temptable(mjdstart, mjdstop,
                                         temperature_file)
        temptable = fix_byteorder(temptable)
    else:
        raise ValueError('Unknown format for temperature file')

    temp_gtis = temperature_gtis(temptable, gti_tolerance)
    if dt is not None:
        temptable = interpolate_temptable(temptable, dt)
    else:
        good = np.diff(temptable['met']) > 0
        good = np.concatenate((good, [True]))
        temptable = temptable[good]
    temptable.meta['gti'] = temp_gtis

    window = np.median(1000 / np.diff(temptable['met']))
    window = int(window // 2 * 2 + 1)
    log.info(f"Smoothing temperature with a window of {window} points")
    temptable['temperature_smooth'] = \
        savgol_filter(temptable['temperature'], window, 3)
    temptable['temperature_smooth_gradient'] = \
        np.gradient(temptable['temperature_smooth'], temptable['met'],
                    edge_order=2)

    return temptable


@lru_cache(maxsize=64)
def load_temptable(temptable_name):
    log.info(f"Reading data from {temptable_name}")
    IS_CSV = temptable_name.endswith('.csv')
    hdf5_name = temptable_name.replace('.csv', '.hdf5')

    if IS_CSV and os.path.exists(hdf5_name):
        IS_CSV = False
        temptable_raw = read_temptable(hdf5_name)
    else:
        temptable_raw = read_temptable(temptable_name)

    if IS_CSV:
        log.info(f"Saving temperature data to {hdf5_name}")
        temptable_raw.write(hdf5_name, overwrite=True)
    return temptable_raw


@lru_cache(maxsize=64)
def load_freq_changes(freq_change_file):
    log.info(f"Reading data from {freq_change_file}")
    return read_freq_changes_table(freq_change_file)


@lru_cache(maxsize=64)
def load_clock_offset_table(clock_offset_file, shift_non_malindi=False):
    return read_clock_offset_table(clock_offset_file,
                                   shift_non_malindi=shift_non_malindi)


class ClockCorrection():
    def __init__(self, temperature_file, mjdstart=None, mjdstop=None,
                 temperature_dt=10, adjust_absolute_timing=False,
                 force_divisor=None, label="", additional_days=2,
                 clock_offset_file=None,
                 hdf_dump_file='dumped_data.hdf5',
                 freqchange_file=None,
                 spline_through_residuals=False):
        # hdf_dump_file_adj = hdf_dump_file.replace('.hdf5', '') + '_adj.hdf5'
        self.temperature_dt = temperature_dt
        self.temperature_file = temperature_file
        self.freqchange_file = freqchange_file

        # Initial value. it will be changed in the next steps
        self.mjdstart = mjdstart
        self.mjdstop = mjdstop

        self.read_temptable()

        if mjdstart is None:
            mjdstart = sec_to_mjd(self.temptable['met'].min())
        else:
            mjdstart = mjdstart - additional_days / 2

        self.clock_offset_file = clock_offset_file
        self.clock_offset_table = \
            read_clock_offset_table(self.clock_offset_file,
                                    shift_non_malindi=True)
        if mjdstop is None:
            last_met = max(self.temptable['met'].max(),
                           self.clock_offset_table['met'].max())
            mjdstop = sec_to_mjd(last_met)

            mjdstop = mjdstop + additional_days / 2

        self.mjdstart = mjdstart
        self.mjdstop = mjdstop

        self.met_start = (self.mjdstart - NUSTAR_MJDREF) * 86400
        self.met_stop = (self.mjdstop - NUSTAR_MJDREF) * 86400

        if label is None or label == "":
            label = f"{self.met_start}-{self.met_stop}"

        self.force_divisor = force_divisor
        self.adjust_absolute_timing = adjust_absolute_timing

        self.hdf_dump_file = hdf_dump_file
        self.plot_file = label + "_clock_adjustment.png"

        self.clock_jump_times = \
            np.array([78708320, 79657575, 81043985, 82055671, 293346772,
            392200784, 394825882, 395304135,407914525, 408299422])
        self.fixed_control_points = np.arange(291e6, 295e6, 86400)
        #  Sum 30 seconds to avoid to exclude these points
        #  from previous interval
        self.gtis = find_good_time_intervals(
            self.temptable, self.clock_jump_times + 30)

        # table_new = temperature_correction_table(
        #     met_start, met_stop, temptable=temptable_raw,
        #     freqchange_file=FREQFILE,
        #     time_resolution=10, craig_fit=False, hdf_dump_file='dump.hdf5')
        #
        # table_new = eliminate_trends_in_residuals(
        #     table_new, clock_offset_table_corr, gtis)

        self.temperature_correction_data = \
            temperature_correction_table(
                self.met_start, self.met_stop,
                # force_divisor=self.force_divisor,
                time_resolution=10,
                temptable = self.temptable,
                # hdf_dump_file=self.hdf_dump_file,
                freqchange_file=self.freqchange_file)

        if adjust_absolute_timing:
            log.info("Adjusting temperature correction")
            try:
                self.adjust_temperature_correction()
            except Exception:
                import traceback
                logfile = 'adjust_temperature_error.log'
                log.warn(f"Temperature adjustment failed. "
                         f"Full error stack in {logfile}")
                with open(logfile, 'w') as fobj:
                    traceback.print_last(file=logfile)

    def read_temptable(self, cache_temptable_name=None):
        if cache_temptable_name is not None and \
                os.path.exists(cache_temptable_name):
            self.temptable = Table.read(cache_temptable_name, path='temptable')
        else:
            self.temptable = \
                    read_temptable(temperature_file=self.temperature_file,
                                   mjdstart=self.mjdstart,
                                   mjdstop=self.mjdstop,
                                   dt=self.temperature_dt)
            if cache_temptable_name is not None:
                self.temptable.write(cache_temptable_name, path='temptable')

    def temperature_correction_fun(self, adjust=False):
        data = self.temperature_correction_data

        return interp1d(np.array(data['met']), np.array(data['temp_corr']),
                        fill_value="extrapolate", bounds_error=False)

    def adjust_temperature_correction(self):
        table_new = eliminate_trends_in_residuals(
            self.temperature_correction_data,
            load_clock_offset_table(
                self.clock_offset_file, shift_non_malindi=True), self.gtis,
            debug=False,
            fixed_control_points=self.fixed_control_points)
        table_new['temp_corr_nodetrend'] = table_new['temp_corr']
        table_new['temp_corr'] = table_new['temp_corr_detrend']
        table_new.remove_column('temp_corr_detrend')

        # 'temp_corr_detrend']
        self.temperature_correction_data = table_new

    def write_clock_file(self, filename=None, save_nodetrend=False,
                         shift_times=0., highres=False):
        from astropy.io.fits import Header, HDUList, BinTableHDU, PrimaryHDU
        from astropy.time import Time

        if filename is None:
            filename = 'new_clock_file.fits'
        table_new = self.temperature_correction_data
        clock_offset_table = self.clock_offset_table
        clock_offset_table = clock_offset_table[
            clock_offset_table['met'] < table_new['met'][-1]]

        tempcorr_idx = np.searchsorted(table_new['met'],
                                       clock_offset_table['met'])
        tempcorr_idx[tempcorr_idx >= table_new['met'].size] = \
            table_new['met'].size -1

        clock_residuals_detrend = clock_offset_table['offset'] - \
                                  table_new['temp_corr'][tempcorr_idx]

        good, _ = get_malindi_data_except_when_out(clock_offset_table) & ~clock_offset_table['flag']

        roll_std = residual_roll_std(clock_residuals_detrend[good])
        control_points = clock_offset_table['met'][good]
        clock_err_fun = interp1d(control_points, roll_std,
                                 assume_sorted=True,
                                 bounds_error=False, fill_value='extrapolate')

        clockerr = clock_err_fun(table_new['met'])

        bti_list = list(zip(self.gtis[:-1, 1], self.gtis[1:, 0]))
        btis = np.array(bti_list)

        for g in btis:
            start, stop = g
            log.info(f"Treating bad data from METs {start}--{stop}")

            temp_idx_start, temp_idx_end = \
                np.searchsorted(table_new['met'], g)
            if temp_idx_end - temp_idx_start == 0:
                continue
            clockerr[temp_idx_start:temp_idx_end] = 0.001
        clockerr[table_new['met'] > self.gtis[-1, 1]] = 0.001
        new_clock_table = Table(
            {'TIME': table_new['met'],
             'CLOCK_OFF_CORR': -table_new['temp_corr'] + shift_times,
             'CLOCK_FREQ_CORR': np.gradient(
                 -table_new['temp_corr'],
                 table_new['met'], edge_order=2),
             'CLOCK_ERR_CORR': clockerr})

        allmets = new_clock_table['TIME']

        if not highres:
            good_for_clockfile = np.zeros(allmets.size, dtype=bool)
            good_for_clockfile[::100] = True
            twodays = 86400 * 2
            for jumptime in self.clock_jump_times:
                idx0, idx1 = np.searchsorted(
                    allmets, [jumptime - twodays, jumptime + twodays])
                # print(idx0, idx1)
                good_for_clockfile[idx0:idx1:10] = True
            new_clock_table_subsample = new_clock_table[good_for_clockfile]
        else:
            new_clock_table_subsample = new_clock_table
        del new_clock_table

        if save_nodetrend:
            new_clock_table_nodetrend = Table(
                {'TIME': table_new['met'],
                 'CLOCK_OFF_CORR': -table_new['temp_corr_nodetrend'] + shift_times,
                 'CLOCK_FREQ_CORR': np.gradient(
                     -table_new['temp_corr'],
                     table_new['met'], edge_order=2),
                 'CLOCK_ERR_CORR': clock_err_fun(
                     table_new['met'])})
            if not highres:
                new_clock_table_subsample_nodetrend = \
                    new_clock_table_nodetrend[good_for_clockfile]
            else:
                new_clock_table_subsample_nodetrend = \
                    new_clock_table_nodetrend

            del new_clock_table_nodetrend

        t = Time.now()
        t.format = 'iso'
        t.out_subfmt = 'date'

        date = t.value
        clock_mets = clock_offset_table['met']
        clstart = clock_mets.min()
        clstop = clock_mets.max()
        tstart = clstart
        tstop = clstop

        header = Header()
        header["XTENSION"] = ('BINTABLE', "Written by Python")
        header["BITPIX"] = (8, "NAXIS   =                    2 /Binary table")
        header["TFIELDS"] = (4, "Number of columns")
        header["EXTNAME"] = ('NU_FINE_CLOCK', "NuSTAR fine clock correction")
        header["DATE"] = (date, "Creation date")
        header["ORIGIN"] = ('Caltech ', "Source of FITS file")
        header["TELESCOP"] = ('NuSTAR  ', "Telescope (mission) name")
        header["INSTRUME"] = (
        'FPM     ', "Instrument name (FPM=mission-level)")
        header["LONGSTRN"] = (
        'OGIP 1.0', "The OGIP long string convention may be used.")
        header["TFORM1"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE1"] = ('TIME    ', "Epoch of spline control point")
        header["TUNIT1"] = ('s       ', "Units of column 1")
        header["TFORM2"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE2"] = ('CLOCK_OFF_CORR', "Clock correction at epoch")
        header["TUNIT2"] = ('s       ', "Units of column 2")
        header["TFORM3"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE3"] = (
        'CLOCK_FREQ_CORR', "Clock freq correction [fractional]")
        header["TFORM4"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE4"] = ('CLOCK_ERR_CORR', "Approx. 1-sigma error of model")
        header["TUNIT4"] = ('s       ', "Units of column 4")
        header[
            'COMMENT'] = "This FITS file may contain long string keyword values that are"
        header[
            'COMMENT'] = "continued over multiple keywords.  This convention uses the  '&'"
        header[
            'COMMENT'] = "character at the end of a string which is then continued"
        header['COMMENT'] = "on subsequent keywords whose name = 'CONTINUE'."
        header['COMMENT'] = ""
        if shift_times != 0.:
            header['COMMENT'] = f"A systematic shift of {shift_times} s was applied to the correction"
        header["RADECSYS"] = ('FK5     ', "celestial coord system")
        header["TIMEUNIT"] = ('s       ', "Units of header time keywords")
        header["TASSIGN"] = ('SATELLITE', "Time assigned by onboard clock")
        header["TIMESYS"] = ('TT      ', "time measured from")
        header["MJDREFI"] = (55197, "MJD reference day")
        header["MJDREFF"] = (
        0.000766018520000, "MJD reference (fraction of day)")
        header["TIMEREF"] = ('LOCAL   ', "reference time")
        header["VERSION"] = (87, "Extension version number")
        header["CONTENT"] = ('NuSTAR Fine Clock Correction', "File content")
        header["CDES0001"] = (
        'NUSTAR FINE CLOCK CORRECTION FILE', "Description")
        header["CCLS0001"] = (
        'BCF     ', "Dataset is a Basic Calibration File")
        header["CDTP0001"] = ('DATA    ', "Calibration file contains data")
        header["CCNM0001"] = ('CLOCK', "Type of calibration data")

        header["CVSD0001"] = ('2010-01-01', "UCT date when file should first be used")
        header["CVST0001"] = ('00:00:00', "UCT time when file should first be used")

        header["CVTD0001"] = (date, "Last UTC date when the file should be used (same as DATE)")
        header["CVTT0001"] = (date, "Last UTC time when the file should be used")

        header["TSTART"] = (tstart, "[s] Start time [MET] of data in the offset file")
        header["TSTOP"] = (tstop, "[s] Stop time [MET] of data in the offset file")

        comment = """NuSTAR fine clock correction file

        This extension contains NuSTAR 'fine' clock correction, which improves
        the NuSTAR clock performance into the 10s of microsecond range.

        During a transition period both kinds of extensions, the standard and
        the fine clock correction, may appear in the same FITS file.
        Use the standard correction, or the fine correction, but not both.
        The task 'barycorr' version 2.2 and higher will automatically choose
        this extension in preference over the standard correction, if it is
        present.

        The extension contains the parameters of a cubic spline model.
        The spline model represents short term clock changes driven by
        the thermal environment of the NuSTAR observatory's clock, as
        well as longer term stochastic drifts.

        The spline is sampled irregularly depending on changes in the
        clock performance and thermal environment.

        Each TIME point in the table is considered a cublic spline
        control point.  At the control point, the value of the spline
        and its derivative are specified.  The correspond to the
        clock *offset* (CLOCK_OFFSET, in seconds) and *frequency*
        (CLOCK_FRACT_FREQ, expressed as a fractional quantity, or
        seconds per second).

        For a given epoch, t, locate the bracketing TIME samples
        in the table.  Use cubic interpolation with the given
        tabulated values to find the clock offset (and optionally
        frequency) at epoch.

        The resulting quantity is the clock correction value.  Thus,
        this value should be *added* from observatory timestamps
        recorded at that epoch.
        """

        for comm in comment.split("\n"):
            header['COMMENT'] = comm

        prihdu = PrimaryHDU()
        prihdu.header['DATE'] = date
        prihdu.header['TELESCOP'] = ('NuSTAR', 'Telescope (mission) name')
        prihdu.header['INSTRUME'] = \
            ('FPM', 'Instrument name (FPM=mission-level)')
        prihdu.header['ORIGIN'] = ('Caltech', 'Source of FITS file')
        prihdu.header['COMMENT'] = \
            "This FITS file may contain long string keyword values that are"
        prihdu.header['COMMENT'] = \
            "continued over multiple keywords.  This convention uses the  '&'"
        prihdu.header['COMMENT'] = \
            "character at the end of a string which is then continued"
        prihdu.header['COMMENT'] = \
            "on subsequent keywords whose name = 'CONTINUE'."

        hdu = BinTableHDU(new_clock_table_subsample, header=header)
        hdulist = [prihdu, hdu]
        if save_nodetrend:
            hdu_nodetrend = \
                BinTableHDU(new_clock_table_subsample_nodetrend,
                            header=header, name="NU_FINE_CLOCK_NODETREND")
            hdulist.append(hdu_nodetrend)

        HDUList(hdulist).writeto(filename, overwrite=True)


def interpolate_clock_function(new_clock_table, mets):
    tab_times = new_clock_table['TIME']
    good_mets = (mets > tab_times.min()) & (mets < tab_times.max())
    mets = mets[good_mets]
    tab_idxs = np.searchsorted(tab_times, mets, side='right') - 1

    clock_off_corr = new_clock_table['CLOCK_OFF_CORR']
    clock_freq_corr = new_clock_table['CLOCK_FREQ_CORR']

    x = np.array(mets)
    xtab = [tab_times[tab_idxs], tab_times[tab_idxs + 1]]
    ytab = [clock_off_corr[tab_idxs], clock_off_corr[tab_idxs + 1]]
    yptab = [clock_freq_corr[tab_idxs], clock_freq_corr[tab_idxs + 1]]

    return cubic_interpolation(x, xtab, ytab, yptab), good_mets


def plot_scatter(new_clock_table, clock_offset_table, shift_times=0,
                 debug=False):
    from bokeh.models import HoverTool
    yint, good_mets = interpolate_clock_function(new_clock_table,
                                                 clock_offset_table['met'])
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(clock_offset_table[good_mets]['met'],
                    clock_offset_table[good_mets]['offset'] - shift_times)
        plt.scatter(
            clock_offset_table[good_mets]['met'],
            clock_offset_table[good_mets]['offset'] - shift_times + yint)
        plt.plot(new_clock_table['TIME'], -new_clock_table['CLOCK_OFF_CORR'])
        plt.show()

    yint = - yint
    clock_offset_table = clock_offset_table[good_mets]
    clock_offset_table['offset'] -= shift_times
    clock_mets = clock_offset_table['met']
    clock_mjds = clock_offset_table['mjd']
    clock_residuals_detrend = clock_offset_table['offset'] - yint

    good, _ = \
        get_malindi_data_except_when_out(
            clock_offset_table) & ~clock_offset_table['flag']

    roll_std = residual_roll_std(clock_residuals_detrend[good])
    control_points = clock_offset_table['met'][good]

    dates = Time(clock_mjds, format='mjd')

    all_data = pd.DataFrame({'met': clock_mets,
                             'mjd': np.array(clock_mjds, dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'offset': clock_offset_table['offset'],
                             'station': clock_offset_table['station']})
    all_data = hv.Dataset(all_data, [('met', 'Mission Epoch Time'),
                                     ('station', 'Ground Station')],
                                    [('offset', 'Clock Offset (s)'),
                                     ('mjd', 'MJD'),
                                     ('doy', 'DOY'),
                                     ('utc', 'UT')])

    tooltips = [
        ('MET', '@met'),
        ('MJD', '@mjd'),
        ('DOY', '@doy'),
        ('UT', '@utc'),
    ]
    hover = HoverTool(tooltips=tooltips)

    plot_0 = all_data.to.scatter('met', ['offset', 'mjd', 'doy', 'utc'],
                                 groupby='station').options(
        color_index='station', alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station')
    plot_0a = hv.Curve(dict(x=clock_mets, y=yint),
                       group='station', label='Clock corr')

    plot_0_all = plot_0.opts(opts.Scatter(width=900, height=350, tools=[hover])).opts(
                             ylim=(-0.1, 0.8)) * plot_0a

    all_data_res = pd.DataFrame({'met': clock_mets,
                             'mjd': np.array(clock_mjds, dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'residual': clock_residuals_detrend * 1e6,
                             'station': clock_offset_table['station']})

    # Jump in and save this to disk here:
    all_data_res.to_pickle('all_data_res.pkl')

    all_data_res = hv.Dataset(all_data_res, [('met', 'Mission Epoch Time'),
                                     ('station', 'Ground Station')],
                                    [('residual', 'Residuals (us)'),
                                     ('mjd', 'MJD'),
                                     ('doy', 'DOY'),
                                     ('utc', 'UT')])
    plot_1 = all_data_res.to.scatter('met', ['residual', 'mjd', 'doy', 'utc'],
                                 groupby='station').options(
        color_index='station', alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station')

    plot_1b = hv.Curve({'x': new_clock_table['TIME'],
                        'y': new_clock_table['CLOCK_ERR_CORR'] * 1e6},
                       group='station', label='scatter up').opts(
        opts.Curve(color='k'))
    plot_1a = hv.Curve({'x': new_clock_table['TIME'],
                        'y': -new_clock_table['CLOCK_ERR_CORR'] * 1e6},
                       group='station', label='scatter down').opts(
        opts.Curve(color='k'))

    plot_1_all = plot_1.opts(
        opts.Scatter(width=900, height=350, tools=[hover])).opts(
                             ylim=(-700, 700)) * plot_1b * plot_1a

    rolling_data = pd.DataFrame({'met':control_points,
                                 'rolling_std':roll_std*1e6})
    rolling_data.to_pickle('rolling_data.pkl')

    text_top = hv.Div("""
        <p>
        Clock offsets measured at ground station passes throughout
        the mission (scatter points), compared to the thermal model for clock
        delays (line). Different colors indicate different ground stations
        (MLD: Malindi; SNG: Singapore; UHI: US Hawaii).
        <p>Use the tools on the top right to zoom in the plot.</p>
        """)

    text_bottom = hv.Div("""
        <p>
        Residuals of the clock offsets with respect to the thermal model.
        The black lines indicate the local scatter, calculated over a time
        span of approximately 5 days. The largest spikes indicate pathological
        intervals.</p>
        <p>
        For periods when there are no temperature measurements the code interpolates
        linearly between the raw clock offsets. We arbitrarily assign a 1-ms error to
        these periods, which should account for the typical deviation of the real clock
        delay with respect to the linear interpolation. This results in 1-ms "spikes" in
        the residuals and the rolling averages.
        </p>
        <p>Use the tools on the top right to zoom in the plot.</p>
        """)

    return hv.Layout((plot_0_all + text_top) +
                     (plot_1_all + text_bottom)).cols(2)


class NuSTARCorr():
    def __init__(self, events_file, outfile=None,
                 adjust=True, force_divisor=None,
                 temperature_file=None, hdf_dump_file=None):
        self.events_file = events_file
        root, ext = splitext_improved(os.path.basename(events_file))
        if outfile is None:
            outfile = events_file.replace(ext, "_tc" + ext)
        if outfile == events_file:
            raise ValueError("outfile == events_file")
        self.outfile = outfile
        self.adjust = adjust
        self.force_divisor = force_divisor
        self.temperature_file = temperature_file
        self.tstart = None
        self.tstop = None
        self.correction_start = None
        self.correction_stop = None
        self.read_observation_info()
        mjdstart, mjdstop = \
            sec_to_mjd([self.tstart, self.tstop])
        self.clock_correction = ClockCorrection(temperature_file,
                                                mjdstart=mjdstart,
                                                mjdstop=mjdstop,
                                                temperature_dt=10,
                                                adjust_absolute_timing=adjust,
                                                force_divisor=force_divisor,
                                                label=root, additional_days=2,
                                                clock_offset_file=None,
                                                hdf_dump_file=hdf_dump_file)
        self.temperature_correction_fun = \
            self.clock_correction.temperature_correction_fun(adjust=adjust)

    def read_observation_info(self):
        with fits.open(self.events_file) as hdul:
            hdr = hdul[1].header
            self.tstart, self.tstop = hdr['TSTART'], hdr['TSTOP']

    def apply_clock_correction(self):
        events_file = self.events_file
        outfile = self.outfile

        log.info(f"Opening {events_file}")

        shutil.copyfile(events_file, outfile)

        with fits.open(outfile) as hdul:
            event_times = hdul[1].data['TIME']
            log.info(f"Calculating temperature correction")
            corr_fun = \
                self.temperature_correction_fun
            hdul[1].data['TIME'] = event_times - corr_fun(event_times)
            hdul.writeto(outfile, overwrite=True)

        return outfile


def simpcumquad(x, y):
    n = len(x)

    if (n != len(y)):
        raise ValueError('ERROR: X and Y array dimensions must match')
    if n == 0:
        raise ValueError('ERROR: X not defined')
    if n == 1:
        return x * 0

    # Find the step size between samples - assume equally sampled
    h = x[1] - x[0]

    # Trapezoid rule for two points
    if n == 2:
        return h * (y[0] + y[1]) / 2

    ii = np.arange((n - 1) // 2, dtype=int) * 2 + 1

    # initialize the area array to zero
    area = np.zeros_like(y)

    area[ii] = 1.25 * y[ii - 1] + 2.0 * y[ii] - 0.25 * y[ii + 1]
    area[ii + 1] = -0.25 * y[ii - 1] + 2.0 * y[ii] + 1.25 * y[ii + 1]

    if n % 2 == 0:
        area[n - 1] = -0.25 * y[n - 3] + 2.0 * y[n - 2] + 1.25 * y[n - 1]

    area1 = (h / 3) * np.cumsum(area)

    return area1


def abs_des_fun(x, b0, b1, b2, b3, t0=77509250):
    """An absorption-desorption function"""
    x = (x - t0) / 86400. / 365.25
    return b0 * np.log(b1 * x + 1) + b2 * np.log(b3 * x + 1)


def clock_ppm_model(nustar_met, temperature, craig_fit=False, version=None):
    """Improved clock model

    Parameters
    ----------
    time : np.array of floats
        Times in MET
    temperature : np.array of floats
        TCXO Temperatures in C
    T0 : float
        Reference temperature
    ppm_vs_T_pars : list
        parameters of the ppm-temperature relation
    ppm_vs_T_pars : list
        parameters of the ppm-time relation (long-term clock decay)

    """
    if craig_fit:
        version = "craig"

    pars = get_temperature_parameters(version)

    T0 = pars['T0']
    offset = pars['offset']
    ppm_vs_T_pars = pars['ppm_vs_T_pars']
    ppm_vs_time_pars = pars['ppm_vs_time_pars']

    temp = (temperature - T0)
    ftemp = offset + ppm_vs_T_pars[0] * temp + \
            ppm_vs_T_pars[1] * temp ** 2  # Temperature dependence

    flongterm = abs_des_fun(nustar_met, *ppm_vs_time_pars)

    return ftemp + flongterm


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      debug=False, craig_fit=False,
                      time_resolution=10, version=None):
    table_times = temptable['met']

    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]

    temperature = temptable['temperature_smooth']

    temp_fun = interp1d(table_times, temperature,
                        assume_sorted=True)

    dt = time_resolution
    times_fine = np.arange(met_start, met_stop, dt)

    try:
        ppm_mod = clock_ppm_model(times_fine, temp_fun(times_fine),
                                  craig_fit=craig_fit, version=version)
    except:
        print(times_fine.min(), times_fine.max())
        print(table_times.min(), table_times.max())
        raise

    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1

    delay_sim = simpcumquad(times_fine, clock_rate_corr)
    return interp1d(times_fine, delay_sim, fill_value='extrapolate',
                    bounds_error=False)


def temperature_correction_table(met_start, met_stop,
                                 temptable=None,
                                 freqchange_file=None,
                                 hdf_dump_file='dump.hdf5',
                                 force_divisor=None,
                                 time_resolution=0.5,
                                 craig_fit=False,
                                 version=None):
    import six
    if hdf_dump_file is not None and os.path.exists(hdf_dump_file):
        log.info(f"Reading cached data from file {hdf_dump_file}")

        result_table = fix_byteorder(Table.read(hdf_dump_file))
        mets = np.array(result_table['met'])
        if (met_start > mets[10] or met_stop < mets[-20]) and (
                met_stop - met_start < 3 * 365 * 86400):
            log.warning(
                "Interval not fully included in cached data. Recalculating.")
        else:
            good = (mets >= met_start - 86400) & (mets < met_stop + 86400)
            return result_table[good]

    if temptable is None or isinstance(temptable, six.string_types):
        mjdstart, mjdstop = sec_to_mjd(met_start), sec_to_mjd(met_stop)
        temptable = read_temptable(mjdstart=mjdstart,
                                   mjdstop=mjdstop,
                                   temperature_file=temptable)
    if force_divisor is None:
        freq_changes_table = \
            read_freq_changes_table(freqchange_file=freqchange_file)
        allfreqtimes = np.array(freq_changes_table['met'])
        allfreqtimes = \
            np.concatenate([allfreqtimes, [allfreqtimes[-1] + 86400]])
        met_intervals = list(
            zip(allfreqtimes[:-1], allfreqtimes[1:]))
        divisors = freq_changes_table['divisor']
    else:
        met_intervals = [[met_start - 10, met_stop + 10]]
        divisors = [force_divisor]

    last_corr = 0
    last_time = met_intervals[0][0]
    N = int((met_stop - met_start) / time_resolution + 50000000)

    log.info(f"Allocating temperature table with {N} entries...")
    table = Table(names=['met', 'temp_corr', 'divisor'],
                  data=np.zeros((N, 3))
                  )
    log.info("Done")

    firstidx = 0
    log.info(f"Calculating temperature correction between "
             f"MET {met_start:.1f}--{met_stop:.1f}")

    for i, met_intv in tqdm.tqdm(enumerate(met_intervals),
                                 total=len(met_intervals)):
        if met_intv[1] < met_start:
            continue
        if met_intv[0] > met_stop:
            break

        start, stop = met_intv

        tempidx = np.searchsorted(temptable['met'], [start - 20, stop + 20])

        temptable_filt = temptable[tempidx[0]:tempidx[1] + 1]

        times_fine = np.arange(start, stop, time_resolution)

        if len(temptable_filt) < 5:
            log.warning(
                f"Too few temperature points in interval "
                f"{start} to {stop} (MET)")
            temp_corr = np.zeros_like(times_fine)
        else:
            delay_function = \
                temperature_delay(temptable_filt, divisors[i], craig_fit=craig_fit,
                                  time_resolution=time_resolution,version=version)

            temp_corr = \
                delay_function(times_fine) + last_corr - delay_function(last_time)
            temp_corr[temp_corr != temp_corr] = 0

        new_data = Table(dict(met=times_fine,
                              temp_corr=temp_corr,
                              divisor=np.zeros_like(times_fine) + divisors[i]))

        # if np.any(temp_corr != temp_corr):
        #     log.error("Invalid data in temperature table")
        #     break

        N = len(times_fine)
        table[firstidx:firstidx + N] = new_data
        firstidx = firstidx + N

        last_corr = temp_corr[-1]
        last_time = times_fine[-1]

    log.info("Interpolation done.")
    table = table[:firstidx]

    if hdf_dump_file is not None:
        log.info(f"Saving intermediate data to {hdf_dump_file}...")
        table.write(hdf_dump_file, overwrite=True)
        log.info(f"Done.")
    return table


def main_tempcorr(args=None):
    import argparse
    description = ('Apply experimental temperature correction to NuSTAR'
                   'event files. For very recent ToO observations not covered'
                   'by the temperature history file or the frequency change '
                   'file, use the -t, --no-adjust and -D options.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name (default <inputfname>_tc.evt)")
    parser.add_argument("-t", "--tempfile", default=None,
                        help="Temperature file (e.g. the nu<OBSID>_eng.hk.gz "
                             "file in the auxil/directory "
                             "or the tp_tcxo*.csv file)")
    parser.add_argument("--cache", default=None,
                        help="HDF5 dump file used as cache (ext. hdf5)")
    parser.add_argument("--no-adjust",
                        help="Do not adjust using tabulated clock offsets",
                        action='store_true', default=False)
    parser.add_argument("-D", "--force-divisor", default=None, type=float,
                        help="Force frequency divisor to this value. Typical "
                             "values are around 24000330")
    parser.add_argument("-r", "--region", default=None, type=str,
                        help="Filter with ds9-compatible region file. MUST be"
                             " a circular region in the FK5 frame")

    args = parser.parse_args(args)

    if args.region is not None:
        args.file = filter_with_region(args.file)

    observation = NuSTARCorr(args.file, outfile=args.outfile,
                             adjust=not args.no_adjust,
                             force_divisor=args.force_divisor,
                             temperature_file=args.tempfile,
                             hdf_dump_file=args.cache)

    outfile = observation.apply_clock_correction()
    return outfile


def main_create_clockfile(args=None):
    import argparse
    description = ('Calculate experimental clock file for NuSTAR, using a '
                   'temperature-driven correction for the onboard TCXO.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("tempfile", type=str,
                        help="Temperature file (e.g. tp_tcxo*.csv)")
    parser.add_argument("offsets", type=str,
                        help="Table of clock offsets")
    parser.add_argument("frequency_changes", type=str,
                        help="Table of divisor changes")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name")
    parser.add_argument("--cache", default=None,
                        help="HDF5 dump file used as cache (ext. hdf5)")
    parser.add_argument("--shift-times", default=4.9e-3, type=float,
                        help="Shift times by this amount")
    parser.add_argument("--save-nodetrend",
                        help="Save un-detrended correction in separate FITS "
                             "extension",
                        action='store_true', default=False)
    parser.add_argument("--high-resolution",
                        help="Create a high-resolution file "
                             "(100 times larger)",
                        action='store_true', default=False)

    args = parser.parse_args(args)

    clockcorr = ClockCorrection(temperature_file=args.tempfile,
                                adjust_absolute_timing=True,
                                clock_offset_file=args.offsets,
                                hdf_dump_file=args.cache,
                                freqchange_file=args.frequency_changes)
    clockcorr.write_clock_file(args.outfile,
                               save_nodetrend=args.save_nodetrend,
                               shift_times=args.shift_times,
                               highres=args.high_resolution)

    clock_offset_table = read_clock_offset_table(args.offsets)
    plot = plot_scatter(Table.read(args.outfile, hdu="NU_FINE_CLOCK"),
                        clock_offset_table,
                        shift_times=args.shift_times)
    from bokeh.io import output_file, save, show
    outfig = args.outfile.replace(".gz", "").replace(".fits", "")
    renderer = hv.renderer('bokeh')
    renderer.save(plot, outfig)


def main_update_temptable(args=None):
    import argparse
    description = ('Calculate experimental clock file for NuSTAR, using a '
                   'temperature-driven correction for the onboard TCXO.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("tempfile", type=str,
                        help="Temperature file (e.g. tp_tcxo*.csv)")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output HDF5 file name")

    args = parser.parse_args(args)

    last_measurement = None
    existing_table = None
    if args.outfile is not None and os.path.exists(args.outfile):
        log.info("Reading existing temperature table")
        existing_table = Table.read(args.outfile)
        last_measurement = existing_table['mjd'][-1] + 0.001 / 86400

    log.info("Reading new temperature values")
    new_table = read_csv_temptable(temperature_file=args.tempfile,
                                   mjdstart=last_measurement)

    if last_measurement is not None:
        new_values = new_table['mjd'] > last_measurement
        if not np.any(new_values):
            log.info("Temperature table is up to date")
            return

        log.info("Updating temperature table")
        new_table = vstack((existing_table, new_table[new_values]))

    window = np.median(310 / np.diff(new_table['met']))
    window = int(window // 2 * 2 + 1)
    new_table['temperature_smooth'] = \
        savgol_filter(new_table['temperature'], window, 2)

    outfile = args.outfile
    if args.outfile is None:
        outfile = os.path.splitext(args.tempfile)[0] + '.hdf5'

    log.info(f"Saving to {outfile}")
    new_table.write(outfile, path="temptable", overwrite=True)


def main_plot_diagnostics(args=None):
    import argparse
    description = ('Plot diagnostic information about the newly produced '
                   'clock file.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("clockcorr", type=str,
                        help="Clock correction file")
    parser.add_argument("clockoff", default=None,
                        help="Clock offset table")

    args = parser.parse_args(args)

    clock_offset_table = read_clock_offset_table(args.clockoff)
    plot = plot_scatter(Table.read(args.clockcorr, hdu="NU_FINE_CLOCK"),
                        clock_offset_table)

    outfig = args.clockcorr.replace(".gz", "").replace(".fits", "")
    renderer = hv.renderer('bokeh')
    renderer.save(plot, outfig)
