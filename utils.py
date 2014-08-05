#!/usr/bin/env python
"""
Utilities for time-series photometry.
"""

from __future__ import print_function, division
import os
import sys
import csv
import pickle
import astropy
import ccdproc
import read_spe
import numpy as np
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def create_master_calib_from_spe(fpath):
    """
    Create master calibration frame from SPE file.
    Median-combine individual calibration frames.
    Return an astropy.ccddata object with all metadata.
    """
    # TODO:
    # - Use multiprocessing to side-step global interpreter lock and parallelize.
    #   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing
    # STH, 20140716
    spe = read_spe.File(fpath)
    combiner_list = []
    fidx_meta = {}
    for frame_idx in xrange(spe.get_num_frames()):
        (data, meta) = spe.get_frame(frame_idx)
        ccddata = ccdproc.CCDData(data=data, meta=meta, unit=astropy.units.adu)
        combiner_list.append(ccddata)
        fidx_meta[frame_idx] = ccddata.meta
    ccddata = ccdproc.Combiner(combiner_list).median_combine()
    ccddata.meta['fidx_meta'] = fidx_meta
    ccddata.meta['footer_xml'] = spe.footer_metadata
    spe.close()
    return ccddata
