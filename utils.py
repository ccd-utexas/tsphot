#!/usr/bin/env python
"""Utilities for time-series photometry.

"""

from __future__ import division, absolute_import, print_function

import math
import inspect

import read_spe
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import astropy
import ccdproc
import imageutils
import photutils
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
    `normd_array = (array - median) / sigmaG`
    `sigmaG = 0.7413(q75 - q50)`
    q50, q75 = 50th, 75th quartiles
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
    to find star-like blobs. Method can also find galaxies by modifying `blobargs`,
    however this pipeline is taylored for stars.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobargs : {dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=2)}, optional
        Dict of keyword arguments for `skimage.feature.blob_log` [3]_.
        Can generalize to extended sources but for increased execution time.
        On 256x256 image: for example below, 0.33 sec/frame; for default above, 0.02 sec/frame.
        Because image is normalized, `threshold` is the number of stdandard deviations
        above background for counts per pixel.
        Example for galaxies: `blobargs=dict(min_sigma=1, max_sigma=30, num_sigma=10, threshold=2)`
    
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : Integer index labeling found star.
        Columns:
            `x_pix` : x-coordinate of found star in image pixels.
            `y_pix` : y-coordinate of found star in image pixels.
            `sigma_pix` : Standard deviation of the Gaussian kernel that detected the blob (usually 1 pixel).
    
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

def plot_stars(image, stars):
    """Plot detected stars overlayed on image.

    Overlay image markers for stars. Convert star sigma to radius.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : Index labeling each star.
        Columns:
            `x_pix` : x-coordinate of star (pixels).
            `y_pix` : y-coordinate of star (pixels).
            `sigma_pix` : Standard deviation Gaussian fit to the star (pixels).

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
    for (idx, x_pix, y_pix, sigma_pix) in stars[['x_pix', 'y_pix', 'sigma_pix']].itertuples():
        fwhm_pix = sigma_to_fwhm(sigma_pix)
        radius_pix = fwhm_pix / 2.0
        circle = plt.Circle((x_pix, y_pix), radius=radius_pix,
                            color='yellow', linewidth=2, fill=False)
        ax.add_patch(circle)
        ax.annotate(str(idx), xy=(x_pix, y_pix), xycoords='data',
                    xytext=(0,0), textcoords='offset points',
                    color='yellow', fontsize=12, rotation=0)
    plt.show()

def center_stars(image, stars, box_sigma=5, method='fit_bivariate_normal'):
    """Compute star centroids in an image with identified stars.

    Extract a subframe around each star. Side-length of the subframe box is sigma_pix*box_sigma.
    From the given method, return a dataframe with sub-pixel coordinates and sigma.

    new category `center_stars `x_pix`, `y_pix`, `sigma_pix`for sub-pixel
    x-, y-coordinates of centroid and standard deviation sigma of intensity distribution.

    
    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each star.
        Columns:
            `x_pix`, `y`, `sigma`.
        `sigma` is the standard deviation of the Gaussian kernel that detected the star (usually 1 pixel).
    box_sigma : {5}, int, optional
        The dimensions for a subframe around the source for centering are
        `box_sigma*sigma` x `box_sigma*sigma`. `sigma` is from `stars` (usually 1 pixel).
        Number should be odd so that center pixel of subframe is initial `x`, `y`.
    method : {fit_bivariate_normal}, optional
        The method by which to compute the centroids.
        `fit_bivariate_normal` : Return the centroid and sigma from robustly fitting a bivariate normal distribution [1]_, [2]_.
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each star and added columns `x_ctrd`, `y_ctrd`, `sigma_ctrd`.

    TODO
    ----
    Include other methods. https://github.com/ccd-utexas/tsphot/issues/69
    Restructure columns to be nested: http://pandas.pydata.org/pandas-docs/stable/indexing.html#hierarchical-indexing-multiindex
    
    References
    ----------
    .. [1] http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.3.1., "The Uniform Distribution"
    
    """
    # Check input
    valid_methods = ['fit_bivariate_normal']
    assert method in valid_methods
    # if method not in valid_methods:
    #     raise IOError(("Invalid method: {meth}\n"+
    #                    "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    # Make box subframes and compute centroids per chosed method. Each star may have a different sigma.
    # Subframe may be shortened due to proximity to frame edge.
    (stars['x_ctrd'], stars['y_ctrd'], stars['sigma_ctrd']) = (np.NaN, np.NaN, np.NaN)
    for (idx, x_star, y_star, sigma_star) in stars[['x_star', 'y_star', 'sigma_star']].itertuples():
        width = np.round(box_sigma*sigma_star)
        height = width
        # width, height order is reverse of position x, y order
        subframe = imageutils.extract_array_2d(array_large=image,
                                               shape=(height, width),
                                               position=(x_star, y_star))
        if method == 'fit_bivariate_normal':
            # Model the photons hitting the pixels of the subframe and
            # robustly fit a bivariate normal distribution.
            # Photons hit each pixel with a uniform distribution. See [1]_, [2]_.
            # Process takes ~90 ms per subframe.
            x_dist = []
            y_dist = []
            for x_idx in xrange(len(subframe)):
                for y_idx in xrange(len(subframe[x_idx])):
                    pixel_counts = np.round(subframe[x_idx][y_idx])
                    np.random.seed(0)
                    x_pix_dist = scipy.stats.uniform(x_idx - 0.5, 1)
                    x_dist.extend(x_pix_dist.rvs(pixel_counts))
                    y_pix_dist = scipy.stats.uniform(y_idx - 0.5, 1)
                    y_dist.extend(y_pix_dist.rvs(pixel_counts))
            (mu, sigma1, sigma2, alpha) = stats.fit_bivariate_normal(x_dist, y_dist, robust=True)
            (x_sub_ctrd, y_sub_ctrd) = mu
            # Modeling coordinate (x,y) as sum of uncorrelated vectors x, y, so
            # adding variances per sec 3.5.1 of Ivezic 2014 [2]_.
            sigma_ctrd = math.sqrt(sigma1**2 + sigma2**2)
        else:
            raise AssertionError(("Program error. Input method not accounted for: {meth}").format(meth=method))
        # Compute the centroid coordinates relative to the star centers.
        (x_sub_star, y_sub_star) = map(lambda dim: (np.round(dim) - 1)/2, (width, height))
        (x_offset, y_offset) = (x_sub_ctrd - x_sub_star,
                                y_sub_ctrd - y_sub_star)
        stars.loc[idx, ['x_ctrd', 'y_ctrd', 'sigma_ctrd']] = (x_star + x_offset,
                                                              y_star + y_offset,
                                                              sigma_ctrd)
    return stars

