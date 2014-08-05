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

def spe_to_dict(fpath):
    """
    Load an SPE file into a dict of ccdproc.ccddata.
    """
    spe = read_spe.File(fpath)
    object_ccddata = {}
    object_ccddata['footer_xml'] = spe.footer_metadata
    for fidx in xrange(spe.get_num_frames()):
        (data, meta) = spe.get_frame(fidx)
        object_ccddata[fidx] = ccdproc.CCDData(data=data, meta=meta, unit=astropy.units.adu)
    spe.close()
    return object_ccddata
    
def create_master_calib(dobj):
    """
    Create master calibration frame from dict of ccdproc.ccddata.
    Median-combine individual calibration frames and retain all metadata.
    """
    # TODO:
    # - Use multiprocessing to side-step global interpreter lock and parallelize.
    #   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing
    # STH, 20140716
    combiner_list = []
    noncombiner_list = []
    fidx_meta = {}
    for key in dobj:
        # If the key is an index for a CCDData frame...
        if isinstance(dobj[key], ccdproc.CCDData):
            combiner_list.append(dobj[key])
            fidx_meta[key] = dobj[key].meta
        # ...otherwise save it as metadata.
        else:
            noncombiner_list.append(key)
    ccddata = ccdproc.Combiner(combiner_list).median_combine()
    ccddata.meta['fidx_meta'] = fidx_meta
    for key in noncombiner_list:
        ccddata.meta[key] = dobj[key]
    return ccddata
