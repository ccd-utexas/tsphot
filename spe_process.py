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

def check_spe(fpath):
    """
    Check that the data file exists and is .spe.
    """
    # TODO: check if ver 3.0, warn if not
    if not os.path.isfile(self._fname):
        raise IOError(("File does not exist: {fname}").format(fname=self._fname))
    (fbase, fext) = os.path.splitext(self._fname)
    if fext != '.spe':
        raise IOError(("File extension not '.spe': {fname}").format(fname=self._fname))
    return None

# Gaussian functional form assumed for PSF fits
def psf((xx,yy),s0,s1,x0,y0,w):
    elow = -50.
    arg = - w * ( (xx-x0)**2 + (yy-y0)**2 )
    arg[arg <= elow] = elow
    intensity = s0 + s1 * np.exp( arg )
    fwhm = 2. * np.sqrt(np.log(2.)/np.abs(w))
    # Turn 2D intensity array into 1D array
    return intensity.ravel()

# Total integrated flux and fwhm for the assumed Gaussian PSF
def psf_flux(s0,s1,x0,y0,w):
    flux = np.pi * s1/np.abs(w)
    fwhm = 2. * np.sqrt(np.log(2.)/np.abs(w))
    return fwhm, flux

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

# Not called.
# def combine(pattern,fileout):
#     files = glob.glob(pattern)
#     print 'files= ',files
#     n=0
#     print "  Combining frames: "
#     for file in files:
#         pa,fi = os.path.split(file)
#         print fi,
#         if ( (n+1)/4 )*4 == n+1:
#             print " "
#         list = fits.open(file)
#         imdata = list[0].data
#         hdr    = list[0].header
#         if n==0:
#             exptime0 = hdr['exptime']
#         exptime = hdr['exptime']
#         if exptime != exptime0:
#             print "\nError: Exposure time mismatch in",file,"\n"
#             exit()
#         if file == files[0]:
#             comb = imdata
#         else:
#             comb = comb + imdata
#         list.close()
#         n=n+1

#     print ""
#     comb = comb/float(n)
#     out = fits.open(files[0])
#     hdr = out[0].header
#     #hdr.rename_keyword('NTP:GPS','NTP-GPS')  # Argos
#     combdata = out[0].data
#     combdata = comb
#     out.writeto(fileout)
#     out.close()
#     return exptime, files, comb

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

def app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2):
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
    # TODO: use classes to retain state information.
    global imdata, iap, nstars, fname_base
    # TODO: use .csv and .txt. ".app" has special meaning on Mac OS.
    efout=open(args.flc,'w')

    #print 'Calculating apertures:'

    iap = 0
    icount = 1
    
    spe = read_spe.File(args.fpath)
    num_frames = spe.get_num_frames()
    is_first_iter = True
    # frame_idx is Python indexed and begins at 0.
    # frame_tracking_number from LightField begins at 1.
    for frame_idx in xrange(num_frames):
        if args.verbose: print "INFO: Processing frame_idx: {num}".format(num=frame_idx)
        if is_first_iter:
            fname_base = os.path.basename(args.fpath)
        icount = icount + 1

        # # open FITS file
        # list = fits.open(file)
        # imdata = list[0].data
        # Read SPE file
        (imdata, metadata) = spe.get_frame(frame_idx)

        # hdr = list[0].header

        # Call aperture photometry routine. Get times, positions, and fluxes
        # var2 contains the list [fluxc,skyc,fwhm])
        # jd,svec,pvec,apvec,starr
        # fluxc, skyc, and fwhm are all lists of length nstars
        # jd, svec, pvec, apvec, var2 = aperture(imdata,hdr,dnorm)
        # jd, svec, pvec, apvec, var2 = aperture(imdata,hdr)

        # Give aperture a hdr dict with keys 'UTC-DATE', 'UTC-BEG'.
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

        (jd, svec, pvec, apvec, var2) = aperture(image=imdata, dt_expstart=dt_expstart_abs, fcoords=args.fcoords)
        ndim = len(apvec)

        # First time through write header
        if icount == 2:
            # head_write(efout,object,nstars)
            # TODO: object is a reserved word. Don't use.
            head_write(efout,fname_base,nstars)

        # Write out results for all apertures
        app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2)

        is_first_iter = False

    spe.close()
    return None

if __name__ == '__main__':
    defaults = {}
    defaults['fcoords'] = "phot_coords"
    defaults['flc']     = "lightcurve.app"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Read .spe file and do aperture photometry."
                                                  +" Output fixed-width-format text file."))
    parser.add_argument("--fpath",
                        required=True,
                        help=("Path to single input .spe file.\n"
                              +"Example: /path/to/file.spe"))
    parser.add_argument("--fcoords",
                        default=defaults['fcoords'],
                        help=(("Input text file with pixel coordinates of stars in first frame.\n"
                               +"Default: {fname}\n"
                               +"Format:\n"
                               +"targx targy\n"
                               +"compx compy\n"
                               +"compx compy\n").format(fname=defaults['fcoords'])))
    parser.add_argument("--flc",
                        default=defaults['flc'],
                        help=(("Output fixed-width-format text file"
                               +" with columns of star intensities by aperture radius.\n"
                               +"Default: {fname}").format(fname=defaults['flc'])))
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
