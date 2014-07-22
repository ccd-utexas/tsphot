from __future__ import print_function, division
import os
import read_spe
import ccdproc
from bs4 import BeautifulSoup
import datetime as dt
import numpy as np
import pandas as pd
import csv
import pickle
import sys

mcalib_mccddata = {}
root_dir = "/Users/harrold/Downloads/test_tsphot/lightfield/"
fobject = os.path.join(root_dir, "test_eclipse_clouds_UWCrB.spe")
# TODO: read in file locations from config file
# Also write out master fits.
# STH 2014-07-16
mcalib_fpath = {}
mcalib_fpath['bias'] = os.path.join(root_dir, "master_bias.pkl")
mcalib_fpath['dark'] = os.path.join(root_dir, "master_dark.pkl")
mcalib_fpath['flat'] = os.path.join(root_dir, "master_flat.pkl")
for mcalib in mcalib_fpath:
    with open(mcalib_fpath[mcalib], 'rb') as fpkl:
        mcalib_mccddata[mcalib] = pickle.load(fpkl)
        
master = mcalib_mccddata['dark']
footer_xml = BeautifulSoup(master.meta['footer_xml'], 'xml')

# Docstring for 'verify_timestamps'
# Note:
# - The ProEM shortcuts non-integer exposure times by <= 0.02 seconds
#   because of tradeoffs necessary for precision timing. Electronics details and
#   reasons were proprietary to Princeton Instruments. The shortcut is not affected
#   by changing CleanBeforeExposure. From tests and emails with PI tech support, fall 2012, STH.
# - The response time of the camera to EXT SYNC trigger
#   is within 10 us from tests with LOGIC OUT in fall 2012, STH.
# - numpy "NaT" == numpy.timedelta64('NaT', 'ns')
#               == numpy.timedelta64(-2**63, 'ns')
# - Convert to type datetime.timedelta before converting to numpy.timedelta64
#   to avoid hardcoding units for numpy.timedelta64. Use division from Python 3.

# TODO: Assert that all pass and log verifications: https://docs.python.org/2/library/logging.html
# TODO: create config file for verify_timestamps module: https://docs.python.org/2/library/configparser.html
# TODO: For pandas v0.14, there is a bug with pandas row-wise operations and type conversions.
#   The bug has been patched and will be in the 0.15 release.
#   Link to Pandas issue: https://github.com/pydata/pandas/issues/7778
# TODO: Accomodate different trigger responses within case statement (use dict as case statement). STH 2014-07-18
# TODO: When upgrading to Python 3, use modulo operation % or rounding with timedeltas.
# TODO: give -f, --force option to override verification.
# TODO: 
# - check temp for darks
# - check readout time < exp time
# - get serial number, model
# - check sensor information between calibration frames
#   by writing to file with json dumps then using https://docs.python.org/2/library/filecmp.html
# TODO: verify against leapseconds.

