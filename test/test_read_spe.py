#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pytest style tests for tsphot/read_spe.py

"""
# TODO: move data files to separate subdir


from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, '.')
import read_spe


def test_read_spe_load_footer_metadata(fname='test_lightbox_10s 2014-05-20 21_56_08.spe', fname_footer='test_footer.txt'):
    """pytest style test for read_spe.File._load_footer_metadata

    """
    assert read_spe.File(fname=fname).footer_metadata == open(fname_footer, 'rb').read().strip()
    return None
