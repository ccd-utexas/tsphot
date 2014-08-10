#!/usr/bin/env python
"""Utilities for time-series photometry.

"""

from __future__ import division, absolute_import, print_function

import sys
import inspect
import math

import read_spe
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import astropy
import ccdproc
import imageutils
import photutils
from photutils.detection import morphology
from astroML import stats
import scipy
from skimage import feature
import matplotlib.pyplot as plt

def create_config(fjson='config.json'):
    """Create configuration file for data reduction.
    
    """
    # TODO: make config file for reductions
    pass

def spe_to_dict(fpath):
    """Load an SPE file into a dict of ccdproc.ccddata.
    
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
    """Create master calibration frame from dict of ccdproc.ccddata.
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

def get_exptime_prog(spe_footer_xml):
    """Get the programmed exposure time in seconds
    from the string XML footer of an SPE file.
    
    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = int(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = int(footer_xml.find(name='DelayResolution').contents[0])
    return (exptime_prog / exptime_prog_res)

def reduce_ccddata_dict(dobj, bias=None, dark=None, flat=None,
                        dobj_exptime=None, dark_exptime=None, flat_exptime=None):
    """Reduce a dict of object data frames using the master calibration frames
    for bias, dark, and flats. All frames must be type ccdproc.CCDData.
    Requires exposure times (seconds) for object data frames, master dark, and master flat.
    Operations (from sec 4.5, Basic CCD Reduction, of Howell, 2006, Handbook of CCD Astronomy):
    - subtract master bias from master dark
    - subtract master bias from master flat
    - scale and subract master dark from master flat
    - subtract master bias from object
    - scale and subtract master dark from object
    - divide object by normalized master flat
    
    """
    # TODO:
    # - parallelize
    # - Compute and correct ccdgain
    #   STH, 20140805
    # Check input.
    iframe = inspect.currentframe()
    (args, varargs, keywords, ilocals) = inspect.getargvalues(iframe)
    for arg in args:
        if ilocals[arg] == None:
            print(("INFO: {arg} is None.").format(arg=arg))
    # Operations:
    # - subtract master bias from master dark
    # - subtract master bias from master flat
    if bias != None:
        if dark != None:
            dark = ccdproc.subtract_bias(dark, bias)
        if flat != None:
            flat = ccdproc.subtract_bias(flat, bias)
    # Operations:
    # - scale and subract master dark from master flat
    if ((dark != None) and
        (flat != None)):
        flat = ccdproc.subtract_dark(flat, dark,
                                     dark_exposure=dark_exptime,
                                     data_exposure=flat_exptime,
                                     scale=True)
    # Operations:
    # - subtract master bias from object
    # - scale and subtract master dark from object
    # - divide object by normalized master flat
    for fidx in dobj:
        if isinstance(dobj[fidx], ccdproc.CCDData):
            if bias != None:
                dobj[fidx] = ccdproc.subtract_bias(dobj[fidx], bias)
            if dark != None:
                dobj[fidx] = ccdproc.subtract_dark(dobj[fidx], dark,
                                                   dark_exposure=dark_exptime,
                                                   data_exposure=dobj_exptime)
            if flat != None:
                dobj[fidx] = ccdproc.flat_correct(dobj[fidx], flat)
    # Remove cosmic rays
    for fidx in dobj:
        if isinstance(dobj[fidx], ccdproc.CCDData):
            dobj[fidx] = ccdproc.cosmicray_lacosmic(dobj[fidx], thresh=5, mbox=11, rbox=11, gbox=5)
    return dobj

def normalize(array):
    """Normalize an array in a robust way.

    The function flattens an array then normalizes in a way that is 
    insensitive to outliers (i.e. ignore stars on an image of the night sky).
    Following [1]_, the function uses `sigmaG` as a width estimator and
    uses the median as an estimator for the mean.
    
    Parameters
    ----------
    array : array_like
        Array can be flat or nested.

    Returns
    -------
    array_normd : array_like
        Normalized version of `array`.

    Notes
    -----
    `array_normd` = (`array` - median(`array`)) / `sigmaG`
    `sigmaG` = 0.7413(q75(`array`) - q50(`array`))
    q50, q75 = 50th, 75th quartiles (q50 == median)
    See [1]_.

    References
    ----------
     .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    
    """
    sigmaG = stats.sigmaG(array)
    median = np.median(array)
    return (array - median) / sigmaG

def sigma_to_fwhm(sigma):
    """Convert the standard deviation sigma of a Gaussian into
    the full-width-at-half-maximum.

    Parameters
    ----------
    sigma : number_like
        ``number_like``, e.g. ``float`` or ``int``

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Full_width_at_half_maximum
    
    """
    fwhm = 2.0*math.sqrt(2.0*np.log(2.0))*sigma
    return fwhm

def find_stars(image,
               blobargs=dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=2)):
    """Find stars in an image and return as a dataframe.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_
    to find star-like blobs. Method can also find extended sources by modifying `blobargs`,
    however this pipeline is taylored for stars.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobargs : {dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=2)}, optional
        Dict of keyword arguments for `skimage.feature.blob_log` [3]_.
        Because image is normalized, `threshold` is the number of stdandard deviations
        above background for counts per pixel.
        Example for extended sources:
            `blobargs`=dict(`min_sigma`=1, `max_sigma`=30, `num_sigma`=10, `threshold`=2)
    
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : Integer index labeling each found star.
        Columns:
            `x_pix` : x-coordinate (pixels) of found star.
            `y_pix` : y-coordinate (pixels) of found star (pixels).
            `sigma_pix` : Standard deviation (pixels) of the Gaussian kernel
                that detected the blob (usually 1 pixel).

    Notes
    -----
    - Can generalize to extended sources but for increased execution time.
      Execution times for 256x256 image:
      - For example for extended sources above: 0.33 sec/frame
      - For default above: 0.02 sec/frame

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    # Normalize image then find stars. Order by x,y,sigma.
    image_normd = normalize(image)
    stars = pd.DataFrame(feature.blob_log(image_normd, **blobargs),
                         columns=['y_pix', 'x_pix', 'sigma_pix'])
    return stars[['x_pix', 'y_pix', 'sigma_pix']]

def plot_stars(image, stars, radius=3):
    """Plot detected stars overlayed on image.

    Overlay circles around stars and label.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
    radius : {3}, optional, number_like
        ``number_like``, e.g. ``float`` or ``int``. The radius of the circle around each star in pixels.

    Returns
    -------
    None
        
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    (fig, ax) = plt.subplots(1, 1)
    ax.imshow(image, interpolation='none')
    for (idx, x_pix, y_pix) in stars[['x_pix', 'y_pix']].itertuples():
        circle = plt.Circle((x_pix, y_pix), radius=radius,
                            color='yellow', linewidth=1, fill=False)
        ax.add_patch(circle)
        ax.annotate(str(idx), xy=(x_pix, y_pix), xycoords='data',
                    xytext=(0,0), textcoords='offset points',
                    color='yellow', fontsize=12, rotation=0)
    plt.show()

def is_odd(num):
    """Determine if a number is odd.

    Parameters
    ----------
    num : number_like
        ``number_like``, e.g. ``float`` or ``int``

    """
    rint = np.rint(num)
    diff = rint - num
    # If num is an integer test if odd...
    if np.equal(diff, 0):
        is_odd = ((num % 2) != 0)
    # ...otherwise num is not odd
    else:
        is_odd = False
    return is_odd
    
def center_stars(image, stars, box_sigma=7, method='centroid_2dg'):
    """Compute centroids of pre-identified stars in an image and return as a dataframe.

    Extract a square subframe around each star. Side-length of the subframe box is sigma_pix*box_sigma.
    With the given method, return a dataframe with sub-pixel coordinates of the centroid.

    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
            `sigma_pix` : Standard deviation (pixels) of a rough 2D Gaussian fit to the star (usually 1 pixel).
    box_sigma : {7}, int, optional
        `box_sigma`*`sigma` x `box_sigma`*`sigma` are the dimensions for a subframe around the source.
        `box_sigma` must be odd and >= 3 so that the center pixel of the subframe is initial `x_pix`, `y_pix`.
        Except for `fit_max_phot_flux`, all methods agree within +/- 0.02 pix for centroid solution
            with 7 <= `box_sigma` <= 11 given `sigma`=1, i.e. 7x7 <= `subframe` size <= 11x11.
    method : {centroid_2dg, centroid_com, fit_bivariate_normal, fit_max_phot_flux}, optional
        The method by which to compute the centroids.
        `centroid_2dg` : Method is from photutils [1]_. Return the centroid from fitting a 2D Gaussian to the
            intensity distribution. Method is fast, accurate, and insensitive to outliers for all `box_sigma`. 
        `centroid_com` : Method is from photutils [1]_. Return the centroid from computing the image moments.
            Method is very fast but only accurate between 7 <= `box_sigma` <= 11 given `sigma`=1 due to
            sensitivity to outliers.
        `fit_bivariate_normal` : Model the photon counts wihtin each pixel of the subframe as from a uniform
            distribution [2]_. Return the centroid from fitting a bivariate normal (Gaussian) distribution to
            the modeled the photon count distribution [3]_. Method is slow but statistically robust. Given `sigma`=1,
            method agrees within +/- 0.02 pix with `centroid_com` for `box_sigma` <= 7 and agrees within +/- 0.02 pix
            with `centroid_2dg` for `box_sigma` >= 7.
        `fit_max_phot_flux` : Method is from Mike Montgomery, UT Austin, 2014. Return the centroid from computing the
            centroid that yields the largest photometric flux. Method is fast, but, as of 2014-08-08 (STH), implementation
            is inaccurate by ~0.1 pix (given `sigma`=1, `box_sigma`=7), and method is possibly sensitive to outliers.
        
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : (same as input `idx`).
        Columns:
            `x_pix` : Sub-pixel x-coordinate (pixels) of centroid.
            `y_pix` : Sub-pixel y-coordinate (pixels) of centroid.

    Notes
    -----
    - Pipeline: Run `find_stars` first. Use the output of `find_stars` as the input of `center_stars`.
            
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/photutils/morphology.html#centroiding-an-object
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.3.1., "The Uniform Distribution"
    .. [3] http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html
    
    """
    # Check input
    if ((not is_odd(box_sigma)) or
        (box_sigma < 3)):
        raise IOError(("Side length for square subframe must odd and >= 3.\n"+
                       "box_sigma = {box_sigma}").format(box_sigma=box_sigma))
    valid_methods = ['centroid_2dg', 'centroid_com', 'fit_bivariate_normal', 'fit_max_phot_flux']
    if method not in valid_methods:
        raise IOError(("Invalid method: {meth}\n"+
                       "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    # Make square subframes and compute centroids by chosed method.
    # Each star may have a different sigma. Store results in a dataframe.
    stars_init = stars.copy()
    stars_finl = pd.DataFrame(np.NaN, index=stars_init.index, columns=['x_pix', 'y_pix'])
    for (idx, x_init, y_init, sigma_init) in stars_init[['x_pix', 'y_pix', 'sigma_pix']].itertuples():
        width = np.rint(box_sigma*sigma_init)
        height = width
        # Note:
        # - Subframe may be shortened due to proximity to frame edge.
        # - width, height order is reverse of position x, y order
        # - numpy.ndarrays are ordered by row_idx (y) then col_idx (x)
        # - (0,0) is in upper left.
        subframe = imageutils.extract_array_2d(array_large=image,
                                               shape=(height, width),
                                               position=(x_init, y_init))
        # Compute the iniitial position for the star relative to the subframe.
        # The star is in an integer pixel.
        (height_actl, width_actl) = subframe.shape
        if ((width_actl != width) or
            (height_actl != height)):
            # TODO: log events. STH, 2014-08-08
            print(("ERROR: Star is too close to edge of frame. Subframe could not be extracted.\n"+
                   "  idx = {idx}\n"+
                   "  box_sigma = {box_sigma}").format(idx=idx, box_sigma=box_sigma), file=sys.stderr)
            continue
        else:
            x_init_sub = (width_actl - 1) / 2
            y_init_sub = (height_actl - 1) / 2
        # Compute the final position for the star relative to the subframe
        # using the selected method.
        if method == 'centroid_2dg':
            # Test results:
            # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix; seeing = 1.4 arcsec.
            # - For varying subframes, method converges to within +/- 0.01 pix of final centroid solution at 7x7 subframe,
            #   and final centroid solution agrees with fit_bivariate_normal final centroid solution within +/- 0.02 pix.
            # - For all subframes, method is closest to final centroid solution.
            # - For 7x7 subframe, centroid solution agrees with fit_bivariate_normal, centroid_com methods' centroid solutions
            #   for 7x7 subframe to within +/- 0.01 pix. Method is least susceptible to outliers.
            # - For 7x7 subframe, method takes ~20 ms. Method scales \propto box_sigma.
            (x_finl_sub, y_finl_sub) = morphology.centroid_2dg(subframe)
            print(morphology.fit_2dgaussian(subframe))
        elif method == 'centroid_com':
            # Test results:
            # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix; seeing = 1.4 arcsec.
            # - For varying subframes, method does not converge to final centroid solution.
            # - For 7x7 to 11x11 subframes, centroid solution agrees with centroid_2dg centroid solution within
            #   +/- 0.01 pix, but then diverges from solution with larger subframes. Method is susceptible to outliers.
            # - For 7x7 subframes, method takes ~3 ms per subframe. Method is invariant to box_sigma and alwyas takes ~3 ms.
            (x_finl_sub, y_finl_sub) = morphology.centroid_com(subframe)
        elif method == 'fit_bivariate_normal':
            # Test results:
            # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix; seeing = 1.4 arcsec.
            # - For varying subframes, method converges to within +/- 0.02 pix of final centroid solution at 7x7 subframe,
            #   and final centoid solution agrees with centroid_2dg final centroid solution within +/- 0.02 pix.
            # - For subframes <= 7x7, centroid solution follows centroid_com within +/- 0.02 pix.
            # - For subframes >= 7x7, centroid solution follows centroid_2dg within +/- 0.02 pix.
            # - For 7x7 subframes, method takes ~350 ms. Method scales \propto box_sigma**2.
            # Method description:
            # - Model the photons hitting the pixels of the subframe and
            #   robustly fit a bivariate normal distribution.
            # - Conservatively assume that photons hit each pixel, even those of the star,
            #   with a uniform distribution. See [2]_, [3]_.
            # - To compute sigma, add variances since modeling coordinate (x,y)
            #   as sum of vectors x, y. Prior PCA makes covariance ~ 0 (sec 3.5.1 of Ivezic 2014 [2]_).
            # - Seed the random number generator for reproducibility.
            x_dist = []
            y_dist = []
            (height_actl, width_actl) = subframe.shape
            for y_idx in xrange(height_actl):
                for x_idx in xrange(width_actl):
                    pixel_counts = np.round(subframe[y_idx, x_idx])
                    np.random.seed(0)
                    x_dist_pix = scipy.stats.uniform(x_idx - 0.5, 1)
                    x_dist.extend(x_dist_pix.rvs(pixel_counts))
                    np.random.seed(0)
                    y_dist_pix = scipy.stats.uniform(y_idx - 0.5, 1)
                    y_dist.extend(y_dist_pix.rvs(pixel_counts))
            (mu, sigma1, sigma2, alpha) = stats.fit_bivariate_normal(x_dist, y_dist, robust=True)
            (x_finl_sub, y_finl_sub) = mu
        elif method == 'fit_max_phot_flux':
            # Test results:
            # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix; seeing = 1.4 arcsec.
            # - For varying subframes, method converges to within +/- 0.0001 pix of final centroid solution at 7x7 subframe,
            #   however final centoid solution disagrees with other methods' centroid solutions.
            # - For 7x7 subframe, centroid solution disagrees with centroid_2dg centroid solution for 7x7 subframe
            #   by ~0.1 pix. Method may be susceptible to outliers.
            # - For 7x7 subframe, method takes ~130 ms. Method scales \propto box_sigma.
            # TODO: Test different minimization methods
            def obj_func(subframe, position, radius):
                """Objective function to minimize: -1*photometric flux from star.
                Assumed to follow a 2D Gaussian point-spread function.

                Parameters
                ----------
                subframe : array_like
                    2D subframe of image. Used only by `obj_func`.
                position : list or array of a tuple
                    Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``, i.e. [x_pix, y_pix] [1]_, [2]_.
                    Used by both `obj_func` and `jac_func`.
                radius : float
                    The radius of the aperture [1]_. Used only by `obj_func`.

                Returns
                -------
                flux_neg : float
                    Negative flux computed by photutils [2]_.

                References
                ----------
                .. [1] http://photutils.readthedocs.org/en/latest/api/photutils.CircularAperture.html#photutils.CircularAperture
                .. [2] http://photutils.readthedocs.org/en/latest/api/photutils.aperture_photometry.html#photutils.aperture_photometry
                .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html

                """
                aperture = ('circular', radius)
                (flux_table, aux_dict) = photutils.aperture_photometry(subframe, position, aperture)
                flux_neg = -1. * flux_table['aperture_sum'].data
                return flux_neg

            def jac_func(subframe, position, radius, eps=0.005):
                """Jacobian of the objective function for fixed radius.
                Assumed to follow a 2D Gaussian point-spread function.

                Parameters
                ----------
                subframe : array_like
                    2D subframe of image. Used only by `obj_func`
                position : list or array of a tuple
                    Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``, i.e. [x_pix, y_pix] [1]_, [2]_.
                    Used by both `obj_func` and `jac_func`.
                radius : float
                    The radius of the aperture [1]_. Used only by `obj_func`.
                eps : float
                    Epsilon value for computing the change in the gradient. Used only by `jac_func`.

                Returns
                -------
                jac : numpy.ndarray
                    Jacobian of obj_func as ``numpy.ndarray`` [dx, dy].

                References
                ----------
                .. [1] http://photutils.readthedocs.org/en/latest/api/photutils.CircularAperture.html#photutils.CircularAperture
                .. [2] http://photutils.readthedocs.org/en/latest/api/photutils.aperture_photometry.html#photutils.aperture_photometry
                .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
                
                """
                try:
                    [x_pix, y_pix] = position
                except ValueError:
                    raise ValueError(("'position' must have the format [x_pix, y_pix]\n"+
                                      "  position = {pos}").format(pos=position))
                jac = np.zeros(len(position))
                fxp1 = obj_func(subframe, (x_pix + eps, y_pix), radius)
                fxm1 = obj_func(subframe, (x_pix - eps, y_pix), radius)
                fyp1 = obj_func(subframe, (x_pix, y_pix + eps), radius)
                fym1 = obj_func(subframe, (x_pix, y_pix - eps), radius)
                jac[0] = (fxp1-fxm1)/(2.*eps)
                jac[1] = (fyp1-fym1)/(2.*eps)
                return jac

            position = [x_init_sub, y_init_sub]
            radius = sigma_to_fwhm(sigma_init)
            res = scipy.optimize.minimize(fun=(lambda pos: obj_func(subframe, pos, radius)),
                                          x0=position,
                                          method='L-BFGS-B',
                                          jac=(lambda pos: jac_func(subframe, pos, radius)),
                                          bounds=((0, width), (0, height)))
            (x_finl_sub, y_finl_sub) = res.x
        else:
            raise AssertionError(("Program error. Input method not accounted for: {meth}").format(meth=method))
        # Compute the centroid coordinates relative to the entire image.
        (x_offset, y_offset) = (x_finl_sub - x_init_sub,
                                y_finl_sub - y_init_sub)
        (x_finl, y_finl) = (x_init + x_offset,
                            y_init + y_offset)
        stars_finl.loc[idx, ['x_pix', 'y_pix']] = (x_finl, y_finl)
    return stars_finl

