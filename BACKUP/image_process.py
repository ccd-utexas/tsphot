#!/usr/bin/env python
"""
Read .spe file and do aperture photometry.
"""

from __future__ import division

from astropy.io import fits
from astropy.time import Time
import glob
import os
import photutils
import scipy.optimize as sco
import numpy as np

import argparse
import sys
import read_spe
import datetime as dt
import pandas as pd
import csv

def center(xx):
    global imdata
    # The aperture size for finding the location of the stars is arbitrarily set here
    app = [photutils.CircularAperture(7.)]
    flux = -photutils.aperture_photometry(imdata, xx[0], xx[1], app, method='exact',subpixels=10)
    return flux

# The derivative of the Gaussian PSF with respect to x and y in pixels. eps = 0.2 pixels
def der_center(xx):
    eps = 0.2
    # The aperture size for finding the location of the stars is arbitrarily set here
    app = 7.
    der = np.zeros_like(xx)
    fxp1 =  center([ xx[0] + eps, xx[1] ])
    fxm1 =  center([ xx[0] - eps, xx[1] ])
    fyp1 =  center([ xx[0], xx[1] + eps ])
    fym1 =  center([ xx[0], xx[1] - eps ])

    der[0] = (fxp1-fxm1)/(2.*eps)
    der[1] = (fyp1-fym1)/(2.*eps)
    return der

# TODO: separate computing aperture from calculating timestamps.
def aperture(image, dt_expstart, fcoords):
    """
    The main function for doing aperture photometry on individual FITS files
    app = -1 means the results are for a PSF fit instead of aperture photometry
    
    Arguments:
    image       = Image data as 2D numpy.ndarray
    dt_expstart = UTC timestamp of start of exposure as datetime.datetime object.
    fcoords     = Text file with pixel coordinates for stars in first frame. Format:
     targx targy
     compx compy
     compx compy
    """
    global iap, pguess_old, nstars, svec
    dnorm = 500.
    rann1 = 18.
    dann = 2.
    rann2 = rann1 + dann
    # variable apertures fail for data on 2014-07-01
    # restricting apertures to 4 pix for now
    # STH, 2014-07-01
    # app_min = 1.
    # app_max = 19.
    app_min = 3.
    app_max = 8.
    dapp = 1.
    app_sizes = np.arange(app_min,app_max,dapp)

    # If first time through, read in "guesses" for locations of stars
    # TODO: just read a csv
    if iap == 0:
        var = np.loadtxt(fcoords)
        xvec = var[:,0]
        yvec = var[:,1]
        nstars = len(xvec)
    else:
        xvec = svec[:,0]
        yvec = svec[:,1]

    # Find locations of stars
    # dxx0 is size of bounding box around star location.
    # Box edge is 2*dxx0
    # TODO: make dxx0 = prelim guess of fwhm
    dxx0 = 10.
    for i in range(nstars):
        xx0 = [xvec[i], yvec[i]]
        xbounds = (xx0[0]-dxx0,xx0[0]+dxx0)
        ybounds = (xx0[1]-dxx0,xx0[1]+dxx0)
        # TODO: when next released by photutils, use:
        # centroid = photutils.detection.morphology.centroid_2dg()
        res = sco.minimize(center, xx0, method='L-BFGS-B', bounds=(xbounds,ybounds),jac=der_center)
        xx0=res.x
        xvec[i] = xx0[0]
        yvec[i] = xx0[1]

    # Calculate sky around stars
    anns = [photutils.CircularAnnulus(rann1, rann2)] * len(xvec)
    sky  = photutils.aperture_photometry(image, xvec, yvec, anns, method='exact',subpixels=10)

    # Do psf fits to stars. Results are stored in arrays fwhm, pflux, psky, psf_x, and psf_y
    # TODO: Move FWHM calculation here, from lc_online. STH, 2014-07-15
    fwhm  = np.zeros(nstars)

    # Make stacked array of star positions from aperture photometry
    svec = np.dstack((xvec,yvec))[0]

    # Make stacked array of star positions from PSF fitting
    # pvec = np.dstack((psf_x,psf_y))[0]
    pvec = svec

    iap = iap + 1

    starr = []
    apvec = []
    app=-1.0

    # Get time of observation from the header
    #date = hdr['DATE-OBS']   # for Argos files
    #utc  = hdr['UTC']         # for Argos files
    # date = hdr['UTC-DATE']    # for Pro-EM files
    # utc  = hdr['UTC-BEG']     # for Pro-EM files
    # times = date + "  " + utc
    # t = Time(times, format='iso', scale='utc')
    # TODO: astropy 0.3.2 fully supports datetime objects
    # When we can upgrade photutils and thus numpy, use datetime objects.
    try:
        t = Time(dt_expstart, scale='utc')
    # ValueError if we haven't upgraded astropy.
    except ValueError:
        t = Time(dt_expstart.isoformat(), format='isot', scale='utc')
    # Calculate Julian Date of observation
    jd  = t.jd
    # TODO: As of 2014-06-01, photutils allows vectors of apertures. Use them. STH
    for radius in app_sizes:
        app = [photutils.CircularAperture(radius)]
        flux = [photutils.aperture_photometry(image, x, y, app, method='exact',subpixels=10)
                for (x, y) in zip(xvec, yvec)]
        skyc = sky*radius**2/(rann2**2 - rann1**2)
        fluxc = flux  - skyc
        starr.append([fluxc,skyc,fwhm])
        apvec.append(radius)
    starr = np.array(starr)
    apvec = np.array(apvec)
    return jd,svec,pvec,apvec,starr

def head_write(ffile,object,nstars):
    dform0='#   Aperture reductions for target {0}. Total number of stars is {1}\n'.format(object,nstars)
    ffile.write(dform0)

    # Format header for the general case of nstars stars
    eform0='#    time (JD)      App (pix)   Target Counts'
    'Comparison Counts     Sky Counts     Target Position    Comp Position    FWHM        Fits File\n'
    for i in range(1,nstars):
        eform0 = eform0 + '      Comp {0} Counts'.format(i)
    eform0 = eform0 + '       Target Sky'
    for i in range(1,nstars):
        eform0 = eform0 + '         Comp {0} Sky'.format(i)
    eform0 = eform0 + '     Target Position'
    for i in range(1,nstars):
        eform0 = eform0 + '  Comp {0} Position'.format(i)
    eform0 = eform0 + '  Target FWHM'
    for i in range(1,nstars):
        eform0 = eform0 + ' Comp {0} FWHM'.format(i)
    eform0 = eform0 + '   FITS File\n'
    ffile.write(eform0)
    return None

def app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2):
    # TODO: use pandas and write out to csv. STH 2014-07-05
    for i in range(0,ndim):
        if apvec[i] >= 0.0:
            eform = '{0:18.8f}  {1:7.2f} '.format(jd,apvec[i])
            for j in range(0,nstars):
                eform = eform + '{0:17.8f}  '.format(var2[i,0,j])
            for j in range(0,nstars):
                eform = eform + '{0:17.8f}  '.format(var2[i,1,j])
            for j in range(0,nstars):
                eform = eform + '{0:8.2f} {1:7.2f} '.format(svec[j,0],svec[j,1])
            for j in range(0,nstars):
                eform = eform + '{0:8.3f}    '.format(var2[i,2,j])
            eform = eform + fname_base + '\n'
        else:
            eform = '{0:18.8f}  {1:7.2f} '.format(jd,apvec[i])
            for j in range(0,nstars):
                eform = eform + '{0:17.8f}  '.format(var2[i,0,j])
            for j in range(0,nstars):
                eform = eform + '{0:17.8f}  '.format(var2[i,1,j])
            for j in range(0,nstars):
                eform = eform + '{0:8.2f} {1:7.2f} '.format(pvec[j,0],pvec[j,1])
            for j in range(0,nstars):
                eform = eform + '{0:8.3f}    '.format(var2[i,2,j])
            eform = eform + fname_base + '\n'
        efout.write(eform)

def main(args):
    """
    Do aperture photometry and write out lightcurve.
    """
    # TODO: if file ext is .spe, use spe funcs; if fits then fits funcs
    # STH 2014-07-15
    
    # TODO: use classes to retain state information.
    global imdata, iap, nstars, fname_base
    # TODO: use .csv
    efout=open(args.flc,'a')

    #print 'Calculating apertures:'
    iap = 0
    icount = 1
    
    # frame_idx is Python indexed and begins at 0.
    # frame_tracking_number from LightField begins at 1.
    is_first_iter = True
    spe = read_spe.File(args.fpath)
    num_frames = spe.get_num_frames()
    # Hack to get around replotting
    # must be redone for blocking out clouds.
    # STH, 2014-07-06
    if args.frame_start == 0:
        bool_write_lc_hdr = True
    else:
        bool_write_lc_hdr = False
    if args.frame_end == -1:
        args.frame_end = num_frames - 1
    # Add one to frame_end since python indexing is non-inclusive for end index.
    for frame_idx in xrange(args.frame_start, args.frame_end + 1):
        if args.verbose: print "INFO: Processing frame_idx: {num}".format(num=frame_idx)
        if is_first_iter:
            fname_base = os.path.basename(args.fpath)
        icount = icount + 1
        print "TEST: icount = {icount}".format(icount=icount)
        # Read SPE file
        (imdata, metadata) = spe.get_frame(frame_idx)
        # Time_stamps from the ProEM's internal timer-counter card are in 1E6 ticks per second.
        # Ticks per second from XML footer metadata using previous LightField experiments:
        # 1 tick = 1 microsecond ; 1E6 ticks per second.
        # 0 ticks is when "Acquire" was first clicked on LightField.
        # Without footer metadata, assume "Acquire" was clicked when the .SPE file was created.
        # File creation time is in seconds since epoch, Jan 1 1970 UTC.
        # Note: Only relevant for online analysis. Not accurate for reductions.
        if is_first_iter:
            dt_fctime_abs = dt.datetime.utcfromtimestamp(os.path.getctime(args.fpath))
        ticks_per_second = 1000000.
        expstart_rel_sec = metadata["time_stamp_exposure_started"] / ticks_per_second
        # Convert ticks from ProEM to seconds since timer-counter card in ProEM
        # counts higher than microseconds argument in datetime.timedelta
        dt_expstart_rel = dt.timedelta(seconds=expstart_rel_sec)
        dt_expstart_abs = dt_fctime_abs + dt_expstart_rel
        # Call aperture photometry routine. Get times, positions, and fluxes
        # jd = "Julian Date"
        # svec = "star vector", ndarray with positions of stars.
        # pvec = "position vector", ndarray with positions of stars from PSF fitting.
        # apvec = "aperture vector", ndarray with range of apertures for photometry.
        # var2 = "variables (2)", 3-nested lists with flux counts, sky counts, fwhm (not calculated)
        #   for each star by aperture. var2 contains the list [fluxc,skyc,fwhm])
        #   fluxc, skyc, and fwhm are all lists of length nstars.
        # Hack to get around non convergence for non-first frame. STH, 2014-07-15
        # TODO: Change this pending output of photutils function when clouds happen.
        try:
            # old version
            (jd, svec, pvec, apvec, var2) = aperture(image=imdata,
                                                     dt_expstart=dt_expstart_abs,
                                                     fcoords=args.fcoords)
            print "TEST: jd = {jd}".format(jd=jd)
            print "TEST: svec = {svec}".format(svec=svec)
            print "TEST: pvec = {pvec}".format(pvec=pvec)
            print "TEST: apvec = {apvec}".format(apvec=apvec)
            print "TEST: var2 = {var2}".format(var2=var2)
            (jd_old, svec_old, pvec_old, apvec_old, var2_old) = (jd, svec, pvec, apvec, var2)
            # new version
            # TODO: check that julian date from astropy.Time is barycentric corrected. STH 2014-07-16
            jul_date = Time(dt_expstart_abs, scale='utc').jd
            print "TEST: jul_date = {jul_date}".format(jul_date=jul_date)
            var = np.loadtxt(fcoords)
        except RuntimeError:
            (jd, svec, pvec, apvec, var2) = (jd_old, svec_old, pvec_old, apvec_old, var2_old)            
        ndim = len(apvec)
        # First time through write header
        # TODO: Hack for autoguiding. make this file modular to avoid hack
        # end of lightcurve.txt file has many repeats of the last data point with this hack
        # until more data is acquired
        # STH, 2014-07-06
        if is_first_iter and bool_write_lc_hdr:
            head_write(efout,fname_base,nstars)
        # Write out results for all apertures
        app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2)
        # df.to_csv(path_or_buf=efout, quoting=csv.QUOTE_NONNUMERIC)
        is_first_iter = False
    spe.close()
    return None

if __name__ == '__main__':
    # TODO: read parameters from config file. STH, 2014-07-15
    arg_default_map = {}
    arg_default_map['fcoords'] = "phot_coords.txt"
    arg_default_map['flc']     = "lightcurve.csv"
    arg_default_map['frame_start'] = 0
    arg_default_map['frame_end']   = -1
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Read .spe file and do aperture photometry."
                                                  +" Output fixed-width-format text file."))
    parser.add_argument("--fpath",
                        required=True,
                        help=("Path to single input .spe file.\n"
                              +"Example: /path/to/file.spe"))
    parser.add_argument("--fcoords",
                        default=arg_default_map['fcoords'],
                        help=(("Input text file with pixel coordinates of stars in first frame.\n"
                               +"Default: {fname}\n"
                               +"Format:\n"
                               +"targx targy\n"
                               +"compx compy\n"
                               +"compx compy\n").format(fname=arg_default_map['fcoords'])))
    parser.add_argument("--flc",
                        default=arg_default_map['flc'],
                        help=(("Output file with columns of star intensities by aperture radius and timestamp.\n"
                               +"Default: {fname}").format(fname=arg_default_map['flc'])))
    parser.add_argument("--frame_start",
                        default=arg_default_map['frame_start'],
                        type=int,
                        help=(("Index of first frame to read. 0 is first frame.\n"
                               +"Default: {default}").format(default=arg_default_map['frame_start'])))
    parser.add_argument("--frame_end",
                        default=arg_default_map['frame_end'],
                        type=int,
                        help=(("Index of last frame to read. -1 is last frame.\n"
                               +"Default: {default}").format(default=arg_default_map['frame_end'])))
    parser.add_argument("--verbose", "-v",
                        action='store_true',
                        help=("Print 'INFO:' messages to stdout."))
    args = parser.parse_args()
    if args.verbose:
        print "INFO: Arguments:"
        for arg in args.__dict__:
            print '   ', arg, args.__dict__[arg]
    if not os.path.isfile(args.fcoords):
        raise IOError(("File does not exist: {fname}").format(fname=args.fcoords))
    main(args)
