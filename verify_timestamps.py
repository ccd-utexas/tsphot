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
    
# Verify computer-camera delay of first triggered frame within a tolerance and correct all frames.
# Assuming trigger is from GPS.
# - Offset of first timestamp from UTC is caused by delay between fetching computer system timestamp
#   and beginning ProEM internal counter/timer card. delay = LF_timestamp - ctrtmr_begin.
# - Verify that the computer-camera delay is within a tolerance. A typical upper bound for the delay is 0.02 sec.
# - Subtract computer-camera delay from all timestamps.
# TODO: When converting to Python 3, use division, multiplication, modulus for numpy.timedeltas.
# TODO: check that dtc verified utc. round to nearest millisecond.
td_0s  = np.timedelta64(dt.timedelta(seconds = (0.0*resolution / resolution)))
td_05s = np.timedelta64(dt.timedelta(seconds = (0.5*resolution / resolution)))
td_1s  = np.timedelta64(dt.timedelta(seconds = (1.0*resolution / resolution)))
if trig_response in required_values['first_frame_trig']:
    ts_firstframestart = df_metadata['time_stamp_exposure_started'].iloc[0]
    # Computer timestamp has default precision of 500 nanoseconds.
    td_remainder = np.timedelta64(ts_firstframestart.microsecond, 'us') + np.timedelta64(ts_firstframestart.nanosecond, 'ns')
    if not ((td_0s <= td_remainder) and (td_remainder < td_1s)):
        raise AssertionError(("Timestamps are not being verified correctly.\n"
                              +"Required:\n"
                              +"  0s <= td_remainder < 1s\n"
                              +"Actual:\n"
                              +"  td_0s        = {td_0s}\n"
                              +"  td_remainder = {td_rem}\n"
                              +"  td_1s        = {td_1s}").format(td_0s=td_0s,
                                                                    td_rem=td_remainder,
                                                                    td_1s=td_1s))
    ts_floor = ts_firstframestart - td_remainder
    ts_ceil = ts_floor + td_1s
    if td_remainder <= td_05s:
        print("ProEM counter/timer started before system time fetched.")
        ts_corrected = ts_floor
        td_correction = -1*td_remainder
        if not ((ts_floor ==  ts_corrected)
                and (ts_corrected <=  ts_firstframestart)
                and (ts_firstframestart < ts_floor + td_05s)
                and (ts_floor + td_05s < ts_ceil)):
            raise AssertionError(("Timestamps are not being verified correctly.\n"
                          +"Required:\n"
                          +"  ts_floor\n"
                          +"    == ts_corrected\n"
                          +"    <= ts_firstframestart\n"
                          +"    <  ts_floor + td_05s\n"
                          +"    <  ts_ceil\n"
                          +"Actual:\n"
                          +"  ts_floor           = {ts0}\n"
                          +"  ts_corrected       = {ts1}\n"
                          +"  ts_firstframestart = {ts2}\n"
                          +"  ts_floor + td_05s  = {ts3}\n"
                          +"  ts_ceil            = {ts4}").format(ts0=ts_floor,
                                                                ts1=ts_corrected,
                                                                ts2=ts_firstframestart,
                                                                ts3=(ts_floor + td_05s),
                                                                ts4=ts_ceil))
    else: # then td_remainder > td_05s:
        # TODO: test.
        print("System time fetched before ProEM counter/timer started.")
        ts_corrected = ts_ceil
        td_correction = td_1s - td_remainder
        if not ((ts_floor <  ts_floor + td_05s)
                and (ts_floor + td_05s <= ts_firstframestart)
                and (ts_firstframestart <= ts_corrected)
                and (ts_corrected == ts_ceil)):
            raise AssertionError(("Timestamps are not being verified correctly.\n"
                          +"Required:\n"
                          +"  ts_floor\n"
                          +"    <  ts_floor + td_05s\n"
                          +"    <= ts_firstframestart\n"
                          +"    <= ts_corrected\n"
                          +"    == ts_ceil\n"
                          +"Actual:\n"
                          +"  ts_floor           = {ts0}\n"
                          +"  ts_floor + td_05s  = {ts1}\n"
                          +"  ts_firstframestart = {ts2}\n"
                          +"  ts_corrected       = {ts3}\n"
                          +"  ts_ceil            = {ts4}").format(ts0=ts_floor,
                                                                ts1=(ts_floor + td_05s),
                                                                ts2=ts_firstframestart,
                                                                ts3=ts_corrected,
                                                                ts4=ts_ceil))
    if abs(td_correction) <= required_values['td_shortcut_tol']:
        df_metadata['delay_verified'] = True
        df_metadata['expstart_corr_delay'] = df_metadata['time_stamp_exposure_started'] + td_correction
        df_metadata['expended_corr_delay'] = df_metadata['time_stamp_exposure_ended'] + td_correction
    else:
        df_metadata['delay_verified'] = False
        print(("WARNING: Computer-camera timestamp delay was outside expected tolerance.\n"
               +"  No correction was applied for computer-camera timestamp delay.\n"
               +"  Required tolerance: +/-{req}\n"
               +"  Actual delay:       {act}").format(req=required_values['td_shortcut_tol'],
                                                      act=td_correction), file=sys.stderr)
else:
    print(("WARNING: First frame was not triggered. No correction was applied for computer-camera timestamp delay."
           +"  Required trigger responses: {req}\n"
           +"  Actual trigger response:    {act}").format(req=required_values['first_frame_trig'],
                                                      act=trig_response), file=sys.stderr)
                                                      
# Verify counter/timer card drift of triggered frames within a tolerance and correct all frames.
# - After correcting for computer-camera delay, offsets of subsequent triggered frames 
#   from integer UTC timestamps are due to drift from the ProEM's internal counter/timer card.
# - Typical drifts from counter/timer cards are ~ -6 us/s and are temperature dependent.
# - Verify that the counter/timer card drift is within a tolerance.
#   +/- 20 us/s is a safe upper bound tolerance for counter/timer card drift.
# - Subtract the counter/timer card drift from all timestamps after the first frame.
# Note: Numpy will omit nanoseconds when doing datetime - datetime.
#       Only do timedelta - timedelta to compute timedeltas to nearest integer seconds.
df_metadata['drift_verified'] = np.NaN
df_metadata['expstart_corr_drift'] = np.NaN
df_metadata['expended_corr_drift'] = np.NaN
if trig_response in required_values['all_frame_trig']:
    # Both tolerance and corrections are cumulative to account for cumulative drift.
    # Define cumulative tolerance.
    td_tot_tol = np.timedelta64(0, 's')
    # TODO: For pandas v0.14, there is a bug with pandas row-wise operations and type conversions.
    #   The bug has been patched and will be in the 0.15 release.
    # Code for pandas v0.15:
    # for (fnum, row) in df_metadata.iterrows():
    for fnum in df_metadata.index:
        row = df_metadata.loc[fnum]
        if row['delay_verified'] == True:
            ts_framestart = row['expstart_corr_delay']
            td_remainder = np.timedelta64(ts_framestart.microsecond, 'us') + np.timedelta64(ts_framestart.nanosecond, 'ns')
            if not ((td_0s <= td_remainder) and (td_remainder < td_1s)):
                raise AssertionError(("Timestamps are not being verified correctly.\n"
                              +"Required:\n"
                              +"  0s <= td_remainder < 1s\n"
                              +"Actual:\n"
                              +"  td_0s        = {td_0s}\n"
                              +"  td_remainder = {td_rem}\n"
                              +"  td_1s        = {td_1s}").format(td_0s=td_0s,
                                                                    td_rem=td_remainder,
                                                                    td_1s=td_1s))
            ts_floor = ts_framestart - td_remainder
            ts_ceil = ts_floor + td_1s
            # ProEM counter/timer card drift >= 0 us/s.
            if td_remainder <= td_05s:
                ts_corrected = ts_floor
                td_correction = -1*td_remainder
                if not ((ts_floor ==  ts_corrected)
                        and (ts_corrected <=  ts_framestart)
                        and (ts_framestart < ts_floor + td_05s)
                        and (ts_floor + td_05s < ts_ceil)):
                    raise AssertionError(("Timestamps are not being verified correctly.\n"
                          +"Required:\n"
                          +"  ts_floor\n"
                          +"    == ts_corrected\n"
                          +"    <= ts_framestart\n"
                          +"    <  ts_floor + td_05s\n"
                          +"    <  ts_ceil\n"
                          +"Actual:\n"
                          +"  ts_floor           = {ts0}\n"
                          +"  ts_corrected       = {ts1}\n"
                          +"  ts_framestart      = {ts2}\n"
                          +"  ts_floor + td_05s  = {ts3}\n"
                          +"  ts_ceil            = {ts4}").format(ts0=ts_floor,
                                                                ts1=ts_corrected,
                                                                ts2=ts_framestart,
                                                                ts3=(ts_floor + td_05s),
                                                                ts4=ts_ceil))
            # ProEM counter/timer card drift < 0 us/s.
            else: # then td_remainder > td_05s:
                ts_corrected = ts_ceil
                td_correction = td_1s - td_remainder
                if not ((ts_floor <  ts_floor + td_05s)
                    and (ts_floor + td_05s <= ts_framestart)
                    and (ts_framestart <= ts_corrected)
                    and (ts_corrected == ts_ceil)):
                    raise AssertionError(("Timestamps are not being verified correctly.\n"
                          +"Required:\n"
                          +"  ts_floor\n"
                          +"    <  ts_floor + td_05s\n"
                          +"    <= ts_framestart\n"
                          +"    <= ts_corrected\n"
                          +"    == ts_ceil\n"
                          +"Actual:\n"
                          +"  ts_floor           = {ts0}\n"
                          +"  ts_floor + td_05s  = {ts1}\n"
                          +"  ts_framestart      = {ts2}\n"
                          +"  ts_corrected       = {ts3}\n"
                          +"  ts_ceil            = {ts4}").format(ts0=ts_floor,
                                                                ts1=(ts_floor + td_05s),
                                                                ts2=ts_framestart,
                                                                ts3=ts_corrected,
                                                                ts4=ts_ceil))

            # Calculate allowed tolerance for new frame from actual exposure time.
            # Compare cumulative timestamp correction with cumulative tolerance
            # including tolerance estimate for new frame.
            td_exp_actual = row['exp_actual']
            td_new_tol = ((td_exp_actual / td_1s) * required_values['td_ctrtmr_tol'])
            if abs(td_correction) <= td_new_tol + td_tot_tol:
                df_metadata['drift_verified'].loc[fnum] = True
                df_metadata['expstart_corr_drift'].loc[fnum] = df_metadata['expstart_corr_delay'].loc[fnum] + td_correction
                df_metadata['expended_corr_drift'].loc[fnum] = df_metadata['expended_corr_delay'].loc[fnum] + td_correction
                # Update cumulative tolerance with last permitted correction.
                td_tot_tol += td_correction
            else:
                # TODO: test
                df_metadata['drift_verified'].loc[fnum] = False
                print(("WARNING: Counter/timer card drift was outside expected tolerance.\n"
                   +"  Stopped applying corrections at frame tracking number: {fnum}\n"
                   +"  Required tolerance: +/-{req}\n"
                   +"  Actual drift:       {act}").format(fnum=fnum,
                                                          req=str(td_tol),
                                                          act=str(td_correction), file=sys.stderr))
                break
        else:
            # TODO: test
            df_metadata['drift_verified'].loc[fnum] = False
            print(("WARNING: Computer-camera delay was not previously corrected."
                   +"  Stopped applying counter/timer drift corrections at frame tracking number: {fnum}\n"
                   +"  Required: drift_verified = True\n"
                   +"  Actual:   drift_verified = False").format(fnum=fnum), file=sys.stderr)
            break
else:
    print(("WARNING: Not all frames were triggered. No correction was applied for counter/timer card timestamp drift."
           +"  Actual trigger response: {tr}"
           +"  Required trigger response: {rtr}").format(tr=trig_response,
                                                       rtr=required_values['all_frame_trig']), file=sys.stderr)
                                                       
# TODO:
# - redo timestamp correction so that time_stamp_exposure_started/ended are just raw values
# - expstart, expended are timedeltas
# - exp_actual, diff_exp_actual-prog, exp_verified are ok
# - deadtime, trig_verified are ok
# - delay_verified, expstart/ended_corr_delay are timedeltas
# - drift_verified, expstart/ended_corr_drift are timedeltas
# - did a leapsecond occur while taking data? see http://www.ietf.org/timezones/data/leap-seconds.list
#   if so, add to all timestamps after it occurred. test with 2012.

# todo: convert programmed exptime, expstart, expended to int
# todo: check expstart + exptime == expended
# todo: check expended_0 == expstart_1
# tood: check expstart_0 + num_frames*exptime = expended_n

# how to do unix time
print(ts_framestart)
np.datetime64(ts_framestart).astype('uint64')/10**6

# TODO: Make module to add to FITS files:
#   KEYWORD  VALUE                # COMMENT  
#   FRAMENUM 10                   # Frame tracking number.
#   VERIFYTS TRUE                 # Verified timestamps (T/F).
#   EXPSTART YYYYMMDDTHHMMSS.SSSZ # Timestamp of exposure start (ISO 8601, UTC).
#   EXPEND   YYYYMMDDTHHMMSS.SSSZ # Timestamp of exposure end (ISO 8601, UTC).
#   USE ARGOS KEYWORDS FOR BEG, END EXPOSURE.
#   EXPTIME  5                    # Exposure time in integer seconds.
#   UNIXTIME 1347782351.SSS       # Unix timestamp of exposure start.

