import pytest

from nuclockutils.utils import get_obsid_list_from_heasarc


@pytest.mark.remote_data
def test_get_obsid_list():
    table = get_obsid_list_from_heasarc()
    assert 'MET' in table.colnames
    assert '80002092002' in table['OBSID']


@pytest.mark.internet_off
def test_get_obsid_list_fail():
    table = get_obsid_list_from_heasarc()
    assert 'MET' in table.colnames
    assert '80002092002' in table['OBSID']

