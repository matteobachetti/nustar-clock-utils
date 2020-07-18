import os
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy import log
import pint.models
import pint.toa as toa
from pint.models import StandardTimingModel
from pint.observatory.nustar_obs import NuSTARObs
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
import nuclockutils
from .utils import filter_with_region
from .nustarclock import interpolate_clock_function


class OrbitalFunctions():
    lat_fun = None
    lon_fun = None
    alt_fun = None
    lst_fun = None


def get_orbital_functions(orbfile):
    from astropy.time import Time
    import astropy.units as u
    orbtable = Table.read(orbfile)
    mjdref = orbtable.meta['MJDREFF'] + orbtable.meta['MJDREFI']

    times = Time(np.array(orbtable['TIME'] / 86400 + mjdref), format='mjd')
    if 'GEODETIC' in orbtable.colnames:
        geod = np.array(orbtable['GEODETIC'])
        lat, lon, alt = geod[:, 0] * u.deg, geod[:, 1] * u.deg, geod[:,
                                                                2] * u.m
    else:
        geod = np.array(orbtable['POLAR'])
        lat, lon, alt = (geod[:, 0] * u.rad).to(u.deg), (
                    geod[:, 1] * u.rad).to(u.deg), geod[:, 2] * 1000 * u.m

    lat_fun = interp1d(times.mjd, lat, bounds_error=False,
                       fill_value='extrapolate')
    lon_fun = interp1d(times.mjd, lon, bounds_error=False,
                       fill_value='extrapolate')
    alt_fun = interp1d(times.mjd, alt, bounds_error=False,
                       fill_value='extrapolate')
    gst = times.sidereal_time('apparent', 'greenwich')
    lst = lon.to(u.hourangle) + gst.to(u.hourangle)
    lst[lst.value > 24] -= 24 * u.hourangle
    lst[lst.value < 0] += 24 * u.hourangle
    lst_fun = interp1d(times.mjd, lst, bounds_error=False,
                       fill_value='extrapolate')

    orbfunc = OrbitalFunctions()
    orbfunc.lat_fun = lat_fun
    orbfunc.lon_fun = lon_fun
    orbfunc.alt_fun = alt_fun
    orbfunc.lst_fun = lst_fun

    return orbfunc


def get_dummy_parfile_for_position(orbfile):

    # Construct model by hand
    with fits.open(orbfile, memmap=True) as hdul:
        label = '_NOM'
        if 'RA_OBJ' in hdul[1].header:
            label = '_OBJ'
        ra = hdul[1].header[f'RA{label}']
        dec = hdul[1].header[f'DEC{label}']

    modelin = StandardTimingModel
    # Should check if 12:13:14.2 syntax is used and support that as well!
    modelin.RAJ.quantity = Angle(ra, unit="deg")
    modelin.DECJ.quantity = Angle(dec, unit="deg")
    modelin.DM.quantity = 0
    return modelin


def get_barycentric_correction(orbfile, parfile, dt=5, ephem='DE421'):
    no = NuSTARObs(name="NuSTAR", FPorbname=orbfile, tt2tdb_mode="pint")
    with fits.open(orbfile) as hdul:
        mjdref = hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF']

    mjds = np.arange(no.X.x[1], no.X.x[-2], dt / 86400)
    mets = (mjds - mjdref) * 86400

    obs, scale = 'nustar', "tt"
    toalist = [None] * len(mjds)

    for i in range(len(mjds)):
        # Create TOA list
        toalist[i] = toa.TOA(mjds[i], obs=obs, scale=scale)

    if parfile is not None and os.path.exists(parfile):
        modelin = pint.models.get_model(parfile)
    else:
        modelin = get_dummy_parfile_for_position(orbfile)

    ts = toa.get_TOAs_list(
        toalist,
        ephem=ephem,
        include_bipm=False,
        include_gps=False,
        planets=False,
        tdb_method='default',
    )
    bats = modelin.get_barycentric_toas(ts)
    return interp1d(mets, (bats.value - mjds) * 86400,
        assume_sorted=True, bounds_error=False, fill_value='extrapolate')


def correct_times(times, bary_fun, clock_fun=None):
    if clock_fun is not None:
        times += clock_fun(times)
    times += bary_fun(times)

    return times


def apply_clock_correction(
    fname, orbfile, outfile='bary.evt', clockfile=None,
    parfile=None, ephem='DE421', radecsys='ICRS', overwrite=False):
    version = nuclockutils.__version__

    bary_fun = get_barycentric_correction(orbfile, parfile, ephem=ephem)
    with fits.open(fname, memmap=True) as hdul:
        times = hdul[1].data['TIME']
        clock_fun = None
        if clockfile is not None and os.path.exists(clockfile):
            clocktable = Table.read(clockfile)
            clock_corr, _ = interpolate_clock_function(clocktable, times)

            clock_fun = interp1d(times, clock_corr,
                assume_sorted=True, bounds_error=False, fill_value='extrapolate')

        for hdu in hdul:
            log.info(f"Updating HDU {hdu.name}")
            for keyname in ['TIME', 'START', 'STOP', 'TSTART', 'TSTOP']:
                if hdu.data is not None and keyname in hdu.data.names:
                    log.info(f"Updating column {keyname}")
                    hdu.data[keyname] = \
                        correct_times(hdu.data[keyname], bary_fun, clock_fun)
                if keyname in hdu.header:
                    log.info(f"Updating header keyword {keyname}")
                    hdu.header[keyname] = \
                        correct_times(hdu.header[keyname], bary_fun, clock_fun)

            hdu.header['CREATOR'] = f'NuSTAR Clock Utils - v. {version}'
            hdu.header['DATE'] = Time.now().fits
            hdu.header['PLEPHEM'] = f'JPL-{ephem}'
            hdu.header['RADECSYS'] = radecsys
            hdu.header['TIMEREF'] = 'SOLARSYSTEM'
            hdu.header['TIMESYS'] = 'TDB'
            hdu.header['TIMEZERO'] = 0.0
            hdu.header['TREFDIR'] = 'RA_OBJ,DEC_OBJ'
            hdu.header['TREFPOS'] = 'BARYCENTER'
        hdul.writeto(outfile, overwrite=overwrite)


def _default_out_file(args):
    outfile = 'bary'
    if not os.path.exists(args.clockfile):
        outfile += '_noclock'
    if not os.path.exists(args.parfile):
        outfile += '_nopar'
    outfile += '.evt'

    return outfile


def main_barycorr(args=None):
    import argparse
    description = ('Apply the barycenter correction to NuSTAR'
                   'event files')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("orbitfile", help="Orbit file")
    parser.add_argument("-p", "--parfile",
                        help="Parameter file in TEMPO/TEMPO2 "
                             "format (for precise coordinates)",
                        default=None, type=str)
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name (default bary_<opts>.evt)")
    parser.add_argument("-c", "--clockfile", default=None,
                        help="Clock correction file")
    parser.add_argument("--overwrite",
                        help="Overwrite existing data",
                        action='store_true', default=False)
    parser.add_argument("-r", "--region", default=None, type=str,
                        help="Filter with ds9-compatible region file. MUST be"
                             " a circular region in the FK5 frame")

    args = parser.parse_args(args)

    outfile = args.outfile
    if outfile is None:
        outfile = _default_out_file(args)

    if args.region is not None:
        args.file = filter_with_region(args.file)

    apply_clock_correction(
        args.file, args.orbitfile, parfile=args.parfile, outfile=args.outfile,
        overwrite=args.overwrite)

    return outfile


if __name__ == '__main__':
    main_barycorr()
