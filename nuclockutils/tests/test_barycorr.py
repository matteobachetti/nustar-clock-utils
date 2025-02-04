from nuclockutils import get_orbital_functions
from nuclockutils import main_barycorr
import os
import numpy as np
import pytest


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6

        self.orbfile = os.path.join(datadir, "dummy_orb.fits")
        self.parfile = os.path.join(datadir, "dummy_par.par")
        self.evfile = os.path.join(datadir, "dummy_evt.evt")

    @pytest.mark.remote_data
    def test_read_orb_funcs(self):
        res = get_orbital_functions(self.orbfile)

    @pytest.mark.remote_data
    def test_barycorr(self):
        outfile = main_barycorr([self.evfile, self.orbfile, "-p", self.parfile])
        assert os.path.exists(outfile)
        with pytest.raises(RuntimeError):
            main_barycorr([self.evfile, self.orbfile, "-p", self.parfile,
                           '-o', outfile])
        main_barycorr([self.evfile, self.orbfile, "-p", self.parfile,
                       '-o', outfile, '--overwrite'])

        os.unlink(outfile)

    @pytest.mark.remote_data
    def test_barycorr_no_bary(self):
        outfile = main_barycorr([self.evfile, "--no-bary"])
        assert os.path.exists(outfile)

        os.unlink(outfile)
