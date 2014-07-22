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

# Define common tolerances and required values for verifying timestamps.
# - For integer-second exposures with minimal deadtime, triggered per-frame,
#   use an upper bound for the ProEM shortcut for non-integer expsosure times, 0.02 seconds.
# - For the ProEM's internal counter/timer card, a typical drift is ~ -6 us/s.
#   Counter/timer card drifts are temperature dependent. +/- 20 us/s is a safe upper bound tolerance.
# - For the first frame to be triggered from the GPS receiver,
#   the trigger response must be "ReadoutPerTrigger" or "StartOnSingleTrigger"
# - For *all* frames to be triggered from the GPS reciever,
#   the trigger response must be "ReadoutPerTrigger".

required_values = {}
required_values['td_shortcut_tol'] = np.timedelta64(20, 'ms')
required_values['td_ctrtmr_tol'] = np.timedelta64(20, 'us') # Use as us/s, not absolute.
required_values['first_frame_trig'] = ['ReadoutPerTrigger', 'StartOnSingleTrigger']
required_values['all_frame_trig'] = ['ReadoutPerTrigger']
print(required_values)
trig_response = footer_xml.find(name='TriggerResponse').contents[0]
print(trig_response)

# Verify absolute timestamp within a tolerance.
# - The absolute timestamp and resolution are from the SPE file's XML footer.
# - Verify that the absolute timestamp is within a tolerance of NTP.
#   From Domain Time Client, 99% of timestamps are within 0.01 sec of NTP. 0.02 sec is a safe upper bound.
#   If the timestamp is more than 0.5 sec off from NTP, the timestamps should be manually inspected.

# TODO: test with created time "2014-05-31T01:23:19.9380367Z" STH 2014-07-18
ts_begin = np.datetime64(footer_xml.find(name='TimeStamp', event='ExposureStarted').attrs['absoluteTime'])
resolution = int(footer_xml.find(name='TimeStamp', event='ExposureStarted').attrs['resolution'])
# TODO: get DTC uncertainty in timestamp. Look for both types of DTC logs. 2014-07-21, STH.
# TODO: verify if w/in 0.02 seconds of NTP with required_values['td_shortcut_tol']

#  Create dataframe of timestamps.
# - Compute per-frame datetimes from elapsed timedeltas for each exposure. 
#   Elapsed timedeltas are from per-frame metadata
#   and were created by the ProEM's internal counter/timer card.
df_metadata = pd.DataFrame.from_dict(master.meta['fidx_meta'], orient='index')
df_metadata = df_metadata.set_index(keys='frame_tracking_number')
df_metadata = df_metadata[['time_stamp_exposure_started', 'time_stamp_exposure_ended']].applymap(lambda x: ts_begin + np.timedelta64(dt.timedelta(seconds = (x / resolution))))

# Verify that all frames exposed for the programmed exposure time within a tolerance.
# - The programmed exposure time and resolution are from the SPE file's XML footer.
# - Compute acutal exposure time from exposure timestamps: exp_end - exp_start
# - Compute the offsets between actual exposure time and programmed exposure time: exp_actual - exp_prog
# - Exposure time resolution is from the shutter resolution, 1000 ticks per second. Exposure time 
#   can be defined as precisely as the internal counter/timer card, 1E6 ticks per second.
# - Verify that abs(exp_actual - exp_prog) are within a tolerance.
#   Offsets between actual and programmed exposure times are limited
#   by the ProEM's shortcut for non-integer exposures, 0.02 sec.
exp_prog = int(footer_xml.find(name='ExposureTime').contents[0])
exp_prog_res = int(footer_xml.find(name='DelayResolution').contents[0])
td_exp_prog = np.timedelta64(dt.timedelta(seconds=(exp_prog / exp_prog_res)))
df_metadata['exp_actual'] = df_metadata['time_stamp_exposure_ended'] - df_metadata['time_stamp_exposure_started']
df_metadata['diff_exp_actual-prog'] = df_metadata['exp_actual'] - td_exp_prog
df_metadata['exp_verified'] = abs(df_metadata['diff_exp_actual-prog']) <= required_values['td_shortcut_tol']
print(("INFO: Programmed exposure time: {exp}").format(exp=str(td_exp_prog)))

# Verify that no triggers were missed. Assuming triggers are from GPS receiver.
# - Check that trigger response is "ReadoutPerTrigger" from the SPE file's XML footer.
# - Compute deadtime between successive frames as timedeltas between exposure timestamps: exp_start1 - exp_end0
#   "NaT" for first frame in sequence.
# - Verify trigger response and that deadtime is within a tolerance.
#   Deadtime is limited by the ProEM's shortcut for non-integer exposures, 0.02 sec.
#   Long deadtime indicates pulse may have been missed.
df_metadata['deadtime'] = df_metadata['time_stamp_exposure_started'] - df_metadata['time_stamp_exposure_ended'].shift()
df_metadata['trig_verified'] = np.NaN
# If the first frame exists, it was triggered.
if trig_response in required_values['first_frame_trig']:
    df_metadata['trig_verified'].iloc[0] = True
else:
    df_metadata['trig_verified'].iloc[0] = False
# Check that subsequent frames were triggered at the correct times.
if trig_response in required_values['all_frame_trig']:
    for (fnum, td_dtime) in df_metadata[1:][['deadtime']].itertuples():
        if td_dtime <= required_values['td_shortcut_tol']:
            df_metadata.loc[fnum, 'trig_verified'] = True
        else:
            df_metadata.loc[fnum, 'trig_verified'] = False
else:
    df_metadata['trig_verified'].iloc[1:] = False
    

