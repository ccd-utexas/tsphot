#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pytest style tests for tsphot/read_spe.py

"""


from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, '.')
import read_spe


def test_read_spe_load_footer_metadata(fname_spe='tests/data/test_lightbox_10s 2014-05-20 21_56_08.spe',
                                       xml_first_40=r'<SpeFormat version="3.0" xmlns="http://w',
                                       xml_last_40=r'39Z" /></GeneralInformation></SpeFormat>'):
    """pytest style test for read_spe.File._load_footer_metadata

    """
    footer_metadata = read_spe.File(fname=fname_spe).footer_metadata
    assert footer_metadata[:40] == xml_first_40
    assert footer_metadata[-40:] == xml_last_40
    return None



