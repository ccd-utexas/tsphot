#!/usr/bin/env python
"""Utilities for pipelining time-series photometry.

See Also
--------
read_spe : Module for reading SPE files.
main : Top-level module.

Notes
-----
noinspection : Comments are created by PyCharm to flag permitted code inspection violations.
docstrings : This module's documentation follows the `numpy` doc example [1]_.
TODO : Flag all to-do items with 'TODO:' in the code body (not the docstring) so that they are flagged when using
    an IDE.
'See Also' : Methods describe their relationships to each other within their docstrings under the 'See Also' section.
    All methods should be connected to at least one other method within this module [2]_.
PIPELINE_SEQUENCE_NUMBER : Methods are labeled like semantic versioning [3]_ within their docstrings under the 'Notes'
    section. The sequence number identifies in what order the functions are usually called by higher-level scripts.
    - Major numbers (..., -1.0, 0.0, 1.0, 2.0, ...) identify functions that are computation/IO-intensive and/or are
        critical to the pipeline.
    - Minor numbers (..., x.0.1, x.1, x.1.1, , x.2, ...) identify functions that are not computation/IO-intensive,
        are optional to the pipeline, and/or are diagnostic.
    - All functions within this module should have a sequence number since they should all have a role in the
        pipeline [2]_.

References
----------
.. [1] https://github.com/numpy/numpy/blob/master/doc/example.py
.. [2] http://en.wikipedia.org/wiki/Pipeline_(software)
.. [3] http://semver.org/

"""
# TODO: Include FITS processing.
# TODO: Write 'Raises' docstring sections.
# TODO: Write 'Examples' docstring sections.

# Forwards compatibility imports.
from __future__ import division, absolute_import, print_function

# Standard library imports.
import os
import sys
import pdb
import math
import json
import logging
import collections

# External package imports. Grouped procedurally then categorically.
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import scipy
import skimage
from skimage import feature
import matplotlib.pyplot as plt
import astropy
import ccdproc
import imageutils
from photutils.detection import morphology, lacosmic
# noinspection PyPep8Naming
from astroML import stats as astroML_stats

# Internal package imports.
import read_spe


# TODO: def create_logging_config (for logging dictconfig)


def create_reduce_config(fjson='reduce_config.json'):
    """Create JSON configuration file for data reduction.

    Parameters
    ----------
    fjson : {'config.json'}, string, optional
        Path to write default JSON configuration file.

    Returns
    -------
    None

    See Also
    --------
    spe_to_dict : Next step in pipeline. Run `create_reduce_config` to create a JSON configuration file. Edit the file
        and use as the input to `spe_to_dict`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER: -1.0

    """
    # TODO: Describe key, value pairs in docstring.
    # To omit an argument in the config file, set it to `None`.
    config_settings = collections.OrderedDict()
    config_settings['comments'] = ["Insert multiline comments here. For formatting, see http://json.org/",
                                   "Use JSON `null`/`true`/`false` for empty/T/F values.",
                                   "  Example in ['master']['dark'].",
                                   "For ['logging']['level'], choices are (from most to least verbose):",
                                   "  ['DEBUG','INFO','WARNING','ERROR','CRITICAL']",
                                   "  See https://docs.python.org/2/library/logging.html#logging-levels"]
    config_settings['logging'] = collections.OrderedDict()
    config_settings['logging']['filename'] = "tsphot.log"
    config_settings['logging']['level'] = "INFO"
    config_settings['calib'] = collections.OrderedDict()
    config_settings['calib']['bias'] = "calib_bias.spe"
    config_settings['calib']['dark'] = "calib_dark.spe"
    config_settings['calib']['flat'] = "calib_flat.spe"
    config_settings['master'] = collections.OrderedDict()
    config_settings['master']['bias'] = "master_bias.pkl"
    config_settings['master']['dark'] = None
    config_settings['master']['flat'] = "master_flat.pkl"
    config_settings['object'] = collections.OrderedDict()
    config_settings['object']['raw'] = "object_raw.spe"
    config_settings['object']['reduced'] = "object_reduced.pkl"
    # Use binary read-write for cross-platform compatibility. Use Python-style indents in the JSON file.
    with open(fjson, 'wb') as fp:
        json.dump(config_settings, fp, sort_keys=False, indent=4)
    return None


def check_reduce_config(dobj):
    """Check configuration settings for data reduction.

    Parameters
    ----------
    dobj : dict
        ``dict`` of configuration settings.

    Returns
    -------
    None

    Raises
    ------
    IOError
        Raised when file doesn't exists, file extension is wrong, or keywords are missing.

    See Also
    --------
    create_reduce_config : Previous step in pipeline. Run `create_reduce_config` to create a JSON configuration file.
        Edit the file and use as the input to `check_reduce_config`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : -0.9

    """
    # TODO: Describe conditionals in docstring.
    # Logging file path need not be defined, but if it is then it must be .log.
    fname = dobj['logging']['filename']
    if fname is not None:
        (fbase, ext) = os.path.splitext(os.path.basename(fname))
        if ext != '.log':
            raise IOError("Logging filename extension is not '.log': {fpath}".format(fpath=fname))
    # Logging level must be set and be one of https://docs.python.org/2/library/logging.html#logging-levels
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level = dobj['logging']['level']
    if level not in valid_levels:
        raise IOError(("Invalid logging level: {level}\n" +
                       "Valid logging levels: {vlevels}").format(level=level, vlevels=valid_levels))
    # Calibration image file paths need not be defined, but if they are then they must exist and be .spe.
    calib_fpath = dobj['calib']
    for imtype in calib_fpath:
        cfpath = calib_fpath[imtype]
        if cfpath is not None:
            if not os.path.isfile(cfpath):
                raise IOError("Calibration image file does not exist: {fpath}".format(fpath=cfpath))
            (fbase, ext) = os.path.splitext(os.path.basename(cfpath))
            if ext != '.spe':
                raise IOError("Calibration image file extension is not '.spe': {fpath}".format(fpath=cfpath))
    # Master calibration image file paths need not be defined, but if they are then they must be .pkl.
    master_fpath = dobj['master']
    for imtype in master_fpath:
        mfpath = master_fpath[imtype]
        if mfpath is not None:
            (fbase, ext) = os.path.splitext(os.path.basename(mfpath))
            if ext != '.pkl':
                raise IOError("Master calibration image file extension is not '.pkl': {fpath}".format(fpath=mfpath))
    # All calibration image image types must have a corresponding type for master image,
    # even if the file path is not defined.
    for imtype in calib_fpath:
        if imtype not in master_fpath:
            raise IOError(("Calibration image type is not in master calibration image types.\n" +
                           "calibration image type: {imtype}\n" +
                           "master image types: {imtypes}").format(imtype=imtype,
                                                                   imtypes=master_fpath.keys()))
    # Raw object image file path must exist and must be .spe.
    rawfpath = dobj['object']['raw']
    if (rawfpath is None) or (not os.path.isfile(rawfpath)):
        raise IOError("Raw object image file does not exist: {fpath}".format(fpath=rawfpath))
    (fbase, ext) = os.path.splitext(os.path.basename(rawfpath))
    if ext != '.spe':
        raise IOError("Raw object image file extension is not '.spe': {fpath}".format(fpath=rawfpath))
    # Reduced object image file path need not be defined, but if it is then it must be .pkl.
    redfpath = dobj['object']['reduced']
    if redfpath is not None:
        (fbase, ext) = os.path.splitext(os.path.basename(redfpath))
        if ext != '.pkl':
            raise IOError("Reduced object image file extension is not '.pkl': {fpath}".format(fpath=redfpath))
    return None


# ######################################################################################################################
# CREATE LOGGER
# Note: Use logger only after checking configuration file.
# Note: For non-root-level loggers, use `getLogger(__name__)`
# http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)


# noinspection PyPep8Naming
def dict_to_class(dobj):
    """Convert keys of a ``dict`` into attributes of a ``Class``.

    Useful for passing arbitrary arguments to functions.

    Parameters
    ----------
    dobj : dict
        ``dict`` with keys and values.

    Returns
    -------
    dclass : Class
        ``Class`` where dclass.key = value.

    See Also
    --------
    create_reduce_config : Previous step in pipeline. Run `create_reduce_config` to create a JSON configuration file.
        Edit the file and use as the input to `dict_to_class`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : -0.8.9

    """
    Dclass = collections.namedtuple('Dclass', dobj.keys())
    return Dclass(**dobj)


# noinspection PyDictCreation
def spe_to_dict(fpath):
    """Load an SPE file into a ``dict`` of `ccdproc.CCDData` with metadata.

    Parameters
    ----------
    fpath : string
        Path to SPE file [1]_.

    Returns
    -------
    object_ccddata : dict
        ``dict`` with `ccdproc.CCDData`. Per-image metadata is stored as `ccdproc.CCDData.meta`.
        SPE file footer is stored under `object_ccddata['footer_xml']`.

    See Also
    --------
    create_reduce_config : Previous step in pipeline. Run `create_reduce_config` to a create JSON configuration file.
        Edit the file and use as the input to `spe_to_dict`.
    create_master_calib : Next step in pipeline. Run `spe_to_dict` then use the output
        in the input to `create_master_calib`.
    read_spe : Module for reading SPE files.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 0.0

    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
           SPE%203.0%20File%20Format%20Specification.pdf

    """
    # TODO : Return SPE header as well.
    spe = read_spe.File(fpath)
    object_ccddata = {}
    object_ccddata['footer_xml'] = spe.footer_metadata
    for fidx in xrange(spe.get_num_frames()):
        (data, meta) = spe.get_frame(fidx)
        object_ccddata[fidx] = ccdproc.CCDData(data=data, meta=meta, unit=astropy.units.adu)
    spe.close()
    return object_ccddata


def create_master_calib(dobj):
    """Create a master calibration image from a ``dict`` of `ccdproc.CCDData`.
    Median-combine individual calibration images and retain all metadata.

    Parameters
    ----------
    dobj : dict with ccdproc.CCDData
        ``dict`` keys with non-`ccdproc.CCDData` values are retained as metadata.

    Returns
    -------
    ccddata : ccdproc.CCDData
        A single master calibration image.
        For `dobj` keys with non-`ccdproc.CCDData` values, the values
        are returned in `ccddata.meta` under the same keys.
        For `dobj` keys with `ccdproc.CCDData` values, the `dobj[key].meta` values
        are returned  are returned as a ``dict`` of metadata.

    See Also
    --------
    spe_to_dict : Previous step in pipeline. Run `spe_to_dict` then use the output
        in the input to `create_master_calib`.
    reduce_ccddata : Next step in pipeline. Run `create_master_calib` to create master
        bias, dark, flat calibration images as input to `reduce_ccddata`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 1.0

    References
    ----------
    
    """
    combiner_list = []
    noncombiner_list = []
    fidx_meta = {}
    for key in dobj:
        # If the key is an index for a CCDData image...
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


def sigma_to_fwhm(sigma):
    """Convert the standard deviation sigma of a Gaussian into
    the full width at half maximum (FWHM).

    Parameters
    ----------
    sigma : float or int

    Returns
    -------
    fwhm : float
        fwhm = 2*sqrt(2*ln(2))*sigma [1]_.

    See Also
    --------
    center_stars : Within `center_stars`, the centroid fitting method that
        maximizes the flux yielded from an aperture calls `sigma_to_fwhm`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 1.0.1

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Full_width_at_half_maximum

    """
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    return fwhm


# noinspection PyPep8Naming
def gain_readnoise_from_master(bias, flat):
    """Calculate the gain and readnoise from a master bias image and a master flat image.

    Parameters
    ----------
    bias : array_like
        2D array of a master bias image.
    flat : array_like
        2D array of a master flat image.

    Returns
    -------
    gain : float
        Gain of the camera in electrons/ADU.
    readnoise : float
        Readout noise of the camera in electrons.

    See Also
    --------
    create_master_calib : Previous step in pipeline. Run `create_master_calib` then use the master bias, flat
        calibration images as input to `gain_readnoise_from_master`.
    gain_readnoise_from_random : Independent method of computing gain and readnoise from random bias and flat images.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER = 1.1
    from [1]_:
        fwhm_bias = readnoise / gain
    from [2]_:
        fwhm_flat = sqrt(mean_flat * gain) / gain
    Solving for gain and readnoise:
        gain = mean_flat / fwhm_flat**2
        readnoise = gain * fwhm_bias
    from [3]_:
        Using the median as estimator of average because robust to outliers.
        Using sigmaG as estimator of standard deviation because robust to outliers.

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 3.7 "Overscan and bias"
    .. [2] Howell, 2006, "Handbook of CCD Astronomy", sec 4.3 "Calculation of read noise and gain"
    .. [3] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"

    """
    # TODO: As of 2014-08-18, gain_readnoise_from_master and gain_readnoise_from_random do not agree. Check formulas.
    sigmaG_bias = astroML_stats.sigmaG(bias)
    fwhm_bias = sigma_to_fwhm(sigmaG_bias)
    sigmaG_flat = astroML_stats.sigmaG(flat)
    fwhm_flat = sigma_to_fwhm(sigmaG_flat)
    median_flat = np.median(flat)
    gain = (median_flat / (fwhm_flat ** 2.0))
    readnoise = (gain * fwhm_bias)
    return (gain * (astropy.units.electron / astropy.units.adu),
            readnoise * astropy.units.electron)


# noinspection PyPep8Naming
def gain_readnoise_from_random(bias1, bias2, flat1, flat2):
    """Calculate gain and readnoise from a pair of random bias images and a pair of random flat images.

    Parameters
    ----------
    bias1, bias2 : array_like
        2D arrays of bias images.
    flat1, flat2 : array_like
        2D arrays of flat images.

    Returns
    -------
    gain : float
        Gain of the camera in electrons/ADU.
    readnoise : float
        Readout noise of the camera in electrons.

    See Also
    --------
    spe_to_dict : Previous step in pipeline. Run `spe_to_dict` then use the bias and flat calibration images as input
        to `gain_readnoise_from_random`.
    gain_readnoise_from_master : Independent method of computing gain and readnoise from master bias and flat images.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER = 1.2
    from [1]_:
        (b1, b2) = (bias1_mean, bias2_mean)
        diff_b12 = b1 - b2
        sig_db12 = stddev(diff_b12)
        (f1, f2) = (flat1_mean, flat2_mean)
        diff_f12 = f1 - f2
        sig_df12 = stddev(diff_f12)
        gain = ((f1 + f2) - (b1 + b2)) / (sig_df12**2 - sig_db12**2)
        readnoise = gain * sig_db12 / sqrt(2)
    from [2]_:
        Using the median as estimator of average because robust to outliers.
        Using sigmaG as estimator of standard deviation because robust to outliers.
    from [3]_:
        The distribution of the difference of two normally distributed variables is the normal difference distribution
            with sigma12**2 = sigma1**2 + sigma2**2

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 4.3 "Calculation of read noise and gain"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    .. [3] http://mathworld.wolfram.com/NormalDifferenceDistribution.html

    """
    # TODO: As of 2014-08-18, gain_readnoise_from_master and gain_readnoise_from_random do not agree. Check formulas.
    # Note: As of 2014-08-18, `ccdproc.CCDData` objects give `numpy.ndarrays` that with 16-bit unsigned ints.
    # Subtracting these arrays gives values close to 0 and close to 65535. Add variance to circumvent unsigned ints.
    b1 = np.median(bias1)
    b2 = np.median(bias2)
    sigmaG_diff_b12 = math.sqrt(astroML_stats.sigmaG(bias1) ** 2.0 + astroML_stats.sigmaG(bias2) ** 2.0)
    f1 = np.median(flat1)
    f2 = np.median(flat2)
    sigmaG_diff_f12 = math.sqrt(astroML_stats.sigmaG(flat1) ** 2.0 + astroML_stats.sigmaG(flat2) ** 2.0)
    gain = (((f1 + f2) - (b1 + b2)) /
            (sigmaG_diff_f12 ** 2.0 - sigmaG_diff_b12 ** 2.0))
    readnoise = gain * sigmaG_diff_b12 / math.sqrt(2.0)
    return (gain * (astropy.units.electron / astropy.units.adu),
            readnoise * astropy.units.electron)


# TODO: Once gain_readnoise_from_masters and gain_readnoise_from_random agree, fix and use check_gain_readnoise
"""
def check_gain_readnoise(bias_dobj, flat_dobj, bias_master = None, flat_master = None,
max_iters=30, max_successes=3, tol_gain=0.01, tol_readnoise = 0.1):
"""
"""Calculate gain and readnoise using both master images
    and random images.
      Compare with image difference/sum method also from
      sec 4.3. Calculation of read noise and gain, Howell
      Needed by cosmic ray cleaner.
"""
"""
    def success_crit(gain_master, gain_new, gain_old, tol_acc_gain, tol_pre_gain,
                     readnoise_master, readnoise_new, readnoise_old, tol_acc_readnoise, tol_pre_readnoise):
"""
"""
        sc = ((abs(gain_new - gain_master) < tol_acc_gain) and
              (abs(gain_new - gain_old)    < tol_pre_gain) and
              (abs(readnoise_new - readnoise_master) < tol_acc_readnoise) and
              (abs(readnoise_new - readnoise_old)    < tol_pre_readnoise))
        return sc
    # randomly select 2 bias images and 2 flat images
    # Accuracy and precision are set to same.
    # tol_readnoise in electrons. From differences in ProEM cameras on calibration sheet.
    # tol_gain in electrons/ADU. From differences in ProEM cameras on calibration sheet.
    # Initialize
    np.random.seed(0)
    is_first_iter = True
    is_converged = False
    num_consec_success = 0
    (gain_finl, readnoise_finl) = (None, None)
    sc_kwargs = {}
    (sc_kwargs['tol_acc_gain'], sc_kwargs['tol_pre_gain']) = (tol_gain, tol_gain)
    (sc_kwargs['tol_acc_readnoise'], sc_kwargs['tol_pre_readnoise']) = (tol_readnoise, tol_readnoise)
    (sc_kwargs['gain_old'], sc_kwargs['readnoise_old']) = (None, None)
    # TODO: calc masters from dobjs if None.
    (sc_kwargs['gain_master'], sc_kwargs['readnoise_master']) = gain_readnoise_from_master(bias_master, flat_master)
    # TODO: Collect an array of values.
    # TODO: redo new, old. new is new median. old is old median.
    for iter in xrange(max_iters):
        # TODO: Use bootstrap sample
        (sc_kwargs['gain_new'], sc_kwargs['readnoise_new']) = gain_readnoise_from_random(bias1, bias2, flat1, flat2)
        if not is_first_iter:
            if (success_crit(**sc_kwargs)):
                num_consec_success += 1
            else:
                num_consec_success = 0
        if num_consec_success >= max_successes:
            is_converged = True
            break
        # Ready for next iteration.
        (sc_kwargs['gain_old'], sc_kwargs['readnoise_old']) = (sc_kwargs['gain_new'], sc_kwargs['readnoise_new'])
        is_first_iter = False
    # After loop.
    if is_converged:
        # todo: give details
        assert iter+1 > max_successes
        assert ((abs(gain_new - gain_master) < tol_acc_gain) and
                (abs(gain_new - gain_old)    < tol_pre_gain) and
                (abs(readnoise_new - readnoise_master) < tol_acc_readnoise) and
                (abs(readnoise_new - readnoise_old)    < tol_pre_readnoise))
        logging.info("Calculations for gain and readnoise converged.")
        (gain_finl, readnoise_finl) = (gain_master, readnoise_master)
    else:
        # todo: assertion error statement
        assert iter == (max_iters - 1)
        # todo: warning stderr description.
        logging.warning("Calculations for gain and readnoise did not converge")
        (gain_finl, readnoise_finl) = (None, None)
    return(gain_finl, readnoise_finl)
"""


def get_exptime_prog(spe_footer_xml):
    """Get the programmed exposure time in seconds from
    the string XML footer of an SPE file.

    Parameters
    ----------
    spe_foooter_xml : string
        ``string`` must be properly formatted XML from a
        Princeton Instruments SPE file footer [1]_.

    Returns
    -------
    exptime_prog_sec : float
        Programmed exposure time in seconds (i.e. the input exposure time from the observer).

    See Also
    --------
    reduce_ccddata : Requires exposure times for images.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 1.3
    Method uses `bs4.BeautifulSoup` to parse the XML ``string``.
    Converts exposure time to seconds from 'ExposureTime' and 'DelayResolution' XML keywords.

    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
           SPE%203.0%20File%20Format%20Specification.pdf

    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = float(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = float(footer_xml.find(name='DelayResolution').contents[0])
    exptime_prog_sec = (exptime_prog / exptime_prog_res)
    return exptime_prog_sec


# noinspection PyUnresolvedReferences
def reduce_ccddata(dobj, dobj_exptime=None,
                   bias=None,
                   dark=None, dark_exptime=None,
                   flat=None, flat_exptime=None):
    """Reduce a dict of object dataframes using the master calibration images
    for bias, dark, and flat.

    All images must be type `ccdproc.CCDData`. Method will do all reductions possible
    with given master calibration images. Method operates on a ``dict``
    in order to minimize the number of pre-reduction operations:
    `dark` - `bias`, `flat` - `bias`, `flat` - `dark`.
    Requires exposure time (seconds) for object dataframes.
    If master dark image is provided, requires exposure time for master dark image.
    If master flat image is provided, requires exposure time for master flat image.

    Parameters
    ----------
    dobj : dict with ccdproc.CCDData
         ``dict`` keys with non-`ccdproc.CCDData` values are retained as metadata.
    dobj_exptime : {None}, float or int, optional
         Exposure time of images within `dobj`. All images must have the same exposure time.
         Required if `dark` is provided.
    bias : {None}, ccdproc.CCDData, optional
        Master bias image.
    dark : {None}, ccdproc.CCDData, optional
        Master dark image. Will be scaled to match exposure time for `dobj` images and `flat` image.
    dark_exptime : {None}, float or int, optional
        Exposure time of `dark`. Required if `dark` is provided.
    flat : {None}, ccdproc.CCDData, optional
        Master flat image.
    flat_exptime : {None}, float or int, optional
        Exposure time of `flat`. Required if `flat` is provided.

    Returns
    -------
    dobj_reduced : dict with ccdproc.CCDData
        `dobj` with `ccdproc.CCDData` images reduced. ``dict`` keys with non-`ccdproc.CCDData` values
        are also returned in `dobj_reduced`.

    See Also
    --------
    create_master_calib : Previous step in pipeline. Run `create_master_calib` to create master
        bias, dark, flat calibration images and input to `reduce_ccddata`.
    remove_cosmic_rays : Next step in pipeline. Run `reduce_ccddata` then use the output
        in the input to `remove_cosmic_rays`.
    get_exptime_prog : Get programmed exposure time from an SPE footer XML string.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 2.0
    As of 2014-08-20, correlated errors in image images are not supported by astropy.
    Sequence of operations (following sec 4.5, "Basic CCD Reduction" [1]_):
    - subtract master bias from master dark
    - subtract master bias from master flat
    - scale and subtract master dark from master flat
    - subtract master bias from each object image
    - scale and subtract master dark from each object image
    - divide each object image by corrected master flat

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy"
    
    """
    # Check input.
    if bias is not None:
        has_bias = True
    else:
        has_bias = False
    if dark is not None:
        has_dark = True
    else:
        has_dark = False
    if flat is not None:
        has_flat = True
    else:
        has_flat = False
    # If there is a `dark` but no `dobj_exptime` or `dark_exptime`:
    if has_dark:
        if (dobj_exptime is None) or (dark_exptime is None):
            raise IOError("If `dark` is provided, both `dobj_exptime` and `dark_exptime` must also be provided.")
    # If there is a `flat` but no `flat_exptime`:
    if has_flat:
        if flat_exptime is None:
            raise IOError("If `flat` is provided, `flat_exptime` must also be provided.")
    # Silence warnings about correlated errors.
    astropy.nddata.conf.warn_unsupported_correlated = False
    # Note: Modify images in-place to reduce memory overhead.
    # Operations:
    # - subtract master bias from master dark
    # - subtract master bias from master flat
    # - scale and subtract master dark from master flat
    if has_bias:
        if has_dark:
            logger.info("Subtracting master bias from master dark.")
            dark = ccdproc.subtract_bias(ccd=dark, master=bias)
        if has_flat:
            logger.info("Subtracting master bias from master flat.")
            flat = ccdproc.subtract_bias(ccd=flat, master=bias)
    if has_dark and has_flat:
        logger.info("Subtracting master dark from master flat.")
        flat = ccdproc.subtract_dark(ccd=flat, master=dark,
                                     dark_exposure=dark_exptime,
                                     data_exposure=flat_exptime,
                                     scale=True)
    # Print progress through dict.
    # Operations:
    # - subtract master bias from object image
    # - scale and subtract master dark from object image
    # - divide object image by corrected master flat
    # TODO: Make a class to track progress.

    key_sortedlist = sorted([key for key in dobj.keys() if isinstance(dobj[key], ccdproc.CCDData)])
    key_len = len(key_sortedlist)
    prog_interval = 0.05
    prog_divs = int(math.ceil(1.0 / prog_interval))
    key_progress = {}
    for idx in xrange(0, prog_divs + 1):
        progress = (idx / prog_divs)
        key_idx = int(math.ceil((key_len - 1) * progress))
        key = key_sortedlist[key_idx]
        key_progress[key] = progress

    logger.info("Reducing object images.")
    logger.info("Subtracting master bias from object images: {tf}".format(tf=has_bias))
    logger.info("Subtracting master dark from object images: {tf}".format(tf=has_dark))
    logger.info("Correcting with master flat for object images: {tf}".format(tf=has_flat))
    for key in sorted(dobj):
        if isinstance(dobj[key], ccdproc.CCDData):
            if has_bias:
                logger.debug("Subtracting master bias from object image: {key}".format(key=key))
                dobj[key] = ccdproc.subtract_bias(ccd=dobj[key], master=bias)
            if has_dark:
                logger.debug("Subtracting master dark from object image: {key}".format(key=key))
                dobj[key] = ccdproc.subtract_dark(ccd=dobj[key], master=dark,
                                                  dark_exposure=dark_exptime,
                                                  data_exposure=dobj_exptime,
                                                  scale=True)
            if has_flat:
                logger.debug("Correcting with master flat for object image: {key}".format(key=key))
                dobj[key] = ccdproc.flat_correct(ccd=dobj[key], flat=flat)
            if key in key_progress:
                logger.info("Progress (%): {pct}".format(pct=int(key_progress[key] * 100)))
    return dobj


def remove_cosmic_rays(image, contrast=2.0, cr_threshold=4.5, neighbor_threshold=0.45, gain=0.85, readnoise=6.1,
                       **kwargs):
    """Remove cosmic rays from an image.

    Method uses the `photutils` implementation of the LA-Cosmic algorithm [1]_.

    Parameters
    ----------
    image : array_like
        2D array of image.
    contrast : {2.0}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from [1]_, and Fig 4 of [2]_.
    cr_threshold : {4.5}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from test script referenced in [3]_.
    neighbor_threshold : {0.45}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from test script referenced in [3]_.
    gain : {0.85}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. In electrons/ADU. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.
    readnoise : {6.1}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. In electrons. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.
    **kwargs :
        Other keyword arguments for `photutils.detection.lacosmic` [1]_.

    Returns
    -------
    image_cleaned : numpy.ndarray
        `image` cleaned of cosmic rays as ``numpy.ndarray``.
    ray_mask : numpy.ndarray of bool
        ``numpy.ndarray`` with same dimensions as `image_cleaned` with only ``True``/``False`` values. Pixels where
        cosmic rays were removed are ``True``.
        
    See Also
    --------
    reduce_ccddata : Previous step in pipeline. Run `reduce_ccddata` then use the output in the input to
        `remove_cosmic_rays`.
    find_stars : Next step in pipeline. Run `remove_cosmic_rays` then use the output in the input to `find_stars`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 3.0
    Use LA-Cosmic algorithm from `photutils` rather than `ccdproc` or `imageutils` until `ccdproc` issue #130 is
        closed [3]_.
    `photutils.detection.lacosmic` is verbose in stdout and stderr.
    
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/_modules/photutils/detection/lacosmic.html
    .. [2] van Dokkum, 2001. http://adsabs.harvard.edu/abs/2001PASP..113.1420V
    .. [3] https://github.com/astropy/ccdproc/issues/130
    .. [4] Princeton Instruments Certificate of Performance for ProEM 1024B EMCCDs with Traditional Amplifier,
        1 MHz readout speed, gain setting #3 (highest).
    
    """
    # TODO: Silence `photutils.detection.lacosmic`. Hack: http://stackoverflow.com/questions/14058453 and 19425736
    # TODO: Silence `photutils.detection.lacosmic`. directing stdout, stderr causes lacosmic to hang at end.
    tmp_kwargs = dict(contrast=contrast, cr_threshold=cr_threshold, neighbor_threshold=neighbor_threshold,
                      gain=gain, readnoise=readnoise, **kwargs)
    logger.debug("LA-Cosmic keyword arguments: {tmp_kwargs}".format(tmp_kwargs=tmp_kwargs))
    (image_cleaned, ray_mask) = lacosmic.lacosmic(image, contrast=contrast, cr_threshold=cr_threshold,
                                                  neighbor_threshold=neighbor_threshold, gain=gain, readnoise=readnoise,
                                                  **kwargs)
    return image_cleaned, ray_mask


# noinspection PyPep8Naming
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
    array_normd : numpy.ndarray
        Normalized `array` as ``numpy.ndarray``.

    See Also
    -------
    find_stars : `find_stars` calls `normalize` to normalize images before
        searching for stars.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 3.1
    `array_normd` = (`array` - median(`array`)) / `sigmaG`
    `sigmaG` = 0.7413(q75(`array`) - q50(`array`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
          sec 3.2, "Descriptive Statistics"
    
    """
    array_np = np.array(array)
    median = np.median(array_np)
    sigmaG = astroML_stats.sigmaG(array_np)
    if sigmaG == 0:
        logger.warning("SigmaG = 0. Normalized array will be all numpy.NaN")
    array_normd = (array_np - median) / sigmaG
    return array_normd


# noinspection PyUnresolvedReferences
def find_stars(image, min_sigma=1, max_sigma=1, num_sigma=1, threshold=3, **kwargs):
    """Find stars in an image and return as a dataframe.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_ to find star-like blobs.
    Method can also find extended sources by modifying `blobargs`, however this pipeline is taylored for stars.
    If focus is poor or if PSF is oversampled (FWHM is many pixels), method may find multiple small stars within a
    single star. Use `center_stars` then `condense_stars` to resolve degeneracy in coordinates.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    min_sigma : {2}, float, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Smallest sigma (pixels) to use for Gaussian kernel.
    max_sigma : {2}, float, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Largest sigma (pixels) to use for Gaussian kernel.
    num_sigma : {1}, float, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Number sigma between smallest and largest sigmas (pixels)
        to use for Gaussian kernel.
    threshold : {3}, float, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Because image is normalized, `threshold` is the number
        of standard deviations above the image median for counts per source pixel.
    **kwargs:
        Other keyword arguments for `skimage.feature.blob_log` [3]_.

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

    See Also
    --------
    remove_cosmic_rays : Previous step in pipeline. Run `remove_cosmic_rays`
        then use the output in the input to `find_stars`.
    center_stars : Next step in pipeline. Run `find_stars` then use the output
        in the input to `center_stars`.
    normalize : `find_stars` calls `normalize` to normalize the image before
        finding stars.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.0
    Can generalize to extended sources but for increased execution time.
        Execution times for 256x256 image:
        Example for extended sources:
        - For default above: 0.02 sec/image
        - For extended sources example below: 0.33 sec/image
        extended_sources = find_stars(image, min_sigma=1, max_sigma=1, num_sigma=1, threshold=3)
    Use `find_stars` after removing cosmic rays to prevent spurious sources.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    # Normalize image then find stars. Order by x,y,sigma.
    image_normd = normalize(image)
    stars = pd.DataFrame(feature.blob_log(image_normd, min_sigma=min_sigma, max_sigma=max_sigma,
                                          num_sigma=num_sigma, threshold=threshold, **kwargs),
                         columns=['y_pix', 'x_pix', 'sigma_pix'])
    return stars[['x_pix', 'y_pix', 'sigma_pix']]


def plot_stars(image, stars, radius=3, interpolation='none', **kwargs):
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
    radius : {3}, optional, float or int
        The radius of the circle around each star in pixels.
    interpolation : 'none', string, optional
        Keyword argument for `matplotlib.pyplot.imshow`.
    **kwargs :
        Other keyword arguments for `matplotlib.pyplot.imshow`.

    Returns
    -------
    None : None
        Displays a plot with labeled stars using `matplotlib.pyplot.imshow`.

    See Also
    --------
    find_stars : Run `find_stars` then use the output in the input to
        `plot_stars` as a diagnostic tool.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.1
    
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    (fig, ax) = plt.subplots(1, 1)
    ax.imshow(image, interpolation=interpolation, **kwargs)
    for (idx, x_pix, y_pix) in stars[['x_pix', 'y_pix']].itertuples():
        circle = plt.Circle((x_pix, y_pix), radius=radius,
                            color='yellow', linewidth=1, fill=False)
        ax.add_patch(circle)
        ax.annotate(str(idx), xy=(x_pix, y_pix), xycoords='data',
                    xytext=(0, 0), textcoords='offset points',
                    color='yellow', fontsize=12, rotation=0)
    plt.show()


def is_odd(num):
    """Determine if a number is equivalent to an odd integer.

    Parameters
    ----------
    num : float or int

    Returns
    -------
    tf_odd : bool

    See Also
    --------
    `center_stars` : `center_stars` calls `is_odd` to check that the
        square subimages extracted around each star have an odd number
        pixels on each side.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.0
    Uses `math.fabs`, `math.fmod` rather than abs, % [1]_.
    Allows negative numbers.
    (1 - (1E-13)) evaluates as odd.
    (1 + 0j) raises TypeError.

    References
    ----------
    .. [1] https://docs.python.org/2/library/math.html

    """
    # `==` works for floats and ints.
    tf_odd = (math.fabs(math.fmod(num, 2)) == 1)
    return tf_odd


def get_square_subimage(image, position, width=11):
    """Extract a square subimage centered on a coordinate position.

    If the coordinate position is too close to a image edge, a rectangular subimage is returned.

    Parameters
    ----------
    image : array_like
        2D array of image.
    position : tuple
        (x, y) pixel coordinate position of center of square subimage. Accepts ``float`` or ``int``.
    width : {11}, optional
        `width` x `width` are the dimensions for a square subimage in pixels. Accepts ``float`` or ``int``.
        `width` will be be corrected to be odd and >= 3 so that the subimage is centered on a pixel.

    Returns
    -------
    subimage : numpy.ndarray
        Square subimage with odd number of pixels per side. If the coordinate position is too close to a image edge,
        a rectangular subimage is returned.

    See Also
    --------
    find_stars : Previous step in pipeline. Run `find_stars` then use the output star coordinate positions as the
        input to `get_square_subimage`.
    subtract_subimage_background : Next step in pipeline. Run `get_square_subimage` to extract a subimage around a star
        then use the output subimage as the input to `subtract_subimage_background`.
    center_stars : `center_stars` calls `get_square_subimage` to extract subimages around stars for centroid fitting
        algorithms.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.1
    Uses imageutils.extract_array_2d to extract the subimage [1]_.

    References
    ----------
    .. [1] http://imageutils.readthedocs.org/en/latest/api/imageutils.extract_array_2d.html#imageutils.extract_array_2d

    """
    # Note:
    # - Dimensions of subimage must be odd so that star is centered.
    # - Shape order (width, height) is reverse of position order (x, y).
    # - numpy.ndarrays are ordered by row_idx (y) then col_idx (x). (0,0) is in upper left.
    # - subimage may not be square due to star's proximity to image edge.
    # noinspection PyUnresolvedReferences
    width = np.rint(width)
    if width < 3:
        width = 3
    if not is_odd(width):
        width += 1
    height = width
    subimage = imageutils.extract_array_2d(array_large=image,
                                           shape=(height, width),
                                           position=position)
    (height_actl, width_actl) = subimage.shape
    if (width_actl != width) or (height_actl != height):
        logger.debug(("Star was too close to the edge of the image to extract a square subimage. " +
                     "width={wid}, position={pos}").format(wid=width, pos=position))
    return subimage


# noinspection PyPep8Naming
def subtract_subimage_background(subimage, threshold_sigma=3):
    """Subtract the background intensity from a subimage centered on a source.

    The function estimates the background as the median intensity of pixels
    bordering the subimage (i.e. square aperture photometry). Background sigma
    is also computed from the border pixels. The median + number of selected sigma
    is subtracted from the subimage. Pixels whose original intensity was less
    than the median + sigma are set to 0.

    Parameters
    ----------
    subimage : array_like
        2D array of subimage.
    threshold_sigma : {3}, float or int, optional
        `threshold_sigma` is the number of standard
        deviations above the subimage median for counts per pixel. Pixels with
        fewer counts are set to 0. Uses `sigmaG` [2]_.

    Returns
    -------
    subimage_sub : numpy.ndarray
        Background-subtracted `subimage` as ``numpy.ndarray``.

    See Also
    --------
    get_square_subimage : Previous step in pipeline. Run `get_square_subimage` then use the output subimage as
        the input `subtract_subimage_background`.
    center_stars : Next step in pipeline. `center_stars` calls `subtract_subimage_background` to
        preprocess subimages around stars for centroid fitting algorithms.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.2
    The source must be centered to within ~ +/- 1/4 of the subimage width.
    At least 3 times as many border pixels used in estimating the background
        as compared to the source [1]_.
    `sigmaG` = 0.7413(q75(`subimage`) - q50(`subimage`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
        
    """
    subimage_np = np.array(subimage)
    (height, width) = subimage_np.shape
    if width != height:
        raise IOError(("Subimage must be square.\n" +
                       "  width = {wid}\n" +
                       "  height = {ht}").format(wid=width,
                                                 ht=height))
    # Choose border width such ratio of number of background pixels to source pixels is >= 3.
    border = int(math.ceil(width / 4.0))
    arr_longtop_longbottom = np.append(subimage_np[:border],
                                       subimage_np[-border:])
    arr_shortleft_shortright = np.append(subimage_np[border:-border, :border],
                                         subimage_np[border:-border, -border:])
    arr_background = np.append(arr_longtop_longbottom,
                               arr_shortleft_shortright)
    arr_source = subimage_np[border:-border, border:-border]
    if (arr_background.size / arr_source.size) < 3:
        # Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
        raise AssertionError(("Program error. There must be at least 3 times as many sky pixels\n" +
                              "  as source pixels to accurately estimate the sky background level.\n" +
                              "  arr_background.size = {nb}\n" +
                              "  arr_source.size = {ns}").format(nb=arr_background.size,
                                                                 ns=arr_source.size))
    median = np.median(arr_background)
    sigmaG = astroML_stats.sigmaG(arr_background)
    subimage_sub = subimage_np - (median + threshold_sigma * sigmaG)
    subimage_sub[subimage_sub < 0.0] = 0.0
    return subimage_sub


# noinspection PyUnresolvedReferences
def center_stars(image, stars, box_pix=21, threshold_sigma=3, method='fit_2dgaussian'):
    """Compute centroids of pre-identified stars in an image and return as a dataframe.

    Extract a square subimage around each star. Side-length of the subimage box is `box_pix`.
    With the given method, return a dataframe with sub-pixel coordinates of the centroid and sigma standard deviation.
    Uses a constant `box_pix`, so assumes all stars in the image have the same PSF. This assumption is invalid
    for galaxies.

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
    box_pix : {11}, optional
        `box_pix` x `box_pix` are the dimensions for a square subimage around the source.
        `box_pix` will be corrected to be odd and >= 3 so that the center pixel of the subimage is
        the initial `x_pix`, `y_pix`. Fitting methods converge to within agreement for `box_pix`>=11.
        Typical observed stars fit in `box_pix` = 21.
    threshold_sigma : {3}, optional
        `threshold_sigma` is the number of standard deviations above the subimage median for counts per pixel.
        Accepts ``float`` or ``int``. Pixels with fewer counts are set to 0. Uses `sigmaG` [3]_.
    method : {fit_2dgaussian, fit_bivariate_normal}, optional
        The method by which to compute the centroids and sigma.
        `fit_2dgaussian` : Method is from photutils [1]_ and astropy [2]_. Return the centroid coordinates and
            standard devaition sigma from fitting a 2D Gaussian to the intensity distribution. `fit_2dgaussian`
            executes quickly, agrees with `fit_bivariate_normal`, and converges within agreement
            by `box_pix` = 11. See example below.
        `fit_bivariate_normal` : Model the photon counts within each pixel of the subimage as from a uniform
            distribution [3]_. Return the centroid coordinates and standard deviation sigma from fitting
            a bivariate normal (Gaussian) distribution to the modeled the photon count distribution [4]_.
            `fit_bivariate_sigma` is statistically robust and converges by `box_pix`= 11, but it executes slowly.
            See example below.
        
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : (same as input `idx`).
        Columns:
            `x_pix` : Sub-pixel x-coordinate (pixels) of centroid.
            `y_pix` : Sub-pixel y-coordinate (pixels) of centroid.
            `sigma_pix` : Sub-pixel standard deviation (pixels) of a 2D Gaussian fit to the star.

    See Also
    --------
    find_stars : Previous step in pipeline. Run `find_stars` then use the output of `find_stars`
        in the input to `center_stars`.
            
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 5.0
    Example: Fitting methods `fit_2dgaussian` and `fit_bivariate_normal` were tested on a bright star with peak
        18000 ADU above background, FHWM ~3.8 pix, initial `sigma_pix` = 1, `box_pix` = 3 to 33. 2014-08-11, STH.
        For `fit_2dgaussian`:
        - For varying subimages, position converges to within 0.01 pix of final solution at 11x11 subimage.
        - For varying subimages, sigma converges to within 0.05 pix of final solution at 11x11 subimage.
        - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
        - For 11x11 subimage, method takes ~25 ms. Method scales \propto box_pix.
        For `fit_bivariate_normal`:
        - For varying subimages, position converges to within 0.02 pix of final solution at 11x11 subimage.
        - For varying subimages, sigma converges to within 0.1 pix of final solution at 11x11 subimage.
        - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
        - For 11x11 subimage, method takes ~450 ms. Method scales \propto box_pix**2.
            
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/photutils/morphology.html#centroiding-an-object
    .. [2] http://astropy.readthedocs.org/en/latest/api/astropy.modeling.functional_models.Gaussian2D.html
    .. [3] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.3.1., "The Uniform Distribution"
    .. [4] http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html
    
    """
    # Check input.
    valid_methods = ['fit_2dgaussian', 'fit_bivariate_normal']
    if method not in valid_methods:
        raise IOError(("Invalid method: {meth}\n" +
                       "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    # Make square subimages and compute centroids and sigma by chosen method.
    # Each star or extended source may have a different sigma. Store results in a dataframe.
    stars_init = stars.copy()
    stars_finl = stars.copy()
    stars_finl[['x_pix', 'y_pix', 'sigma_pix']] = np.NaN
    width = int(math.ceil(box_pix))
    for (idx, x_init, y_init, sigma_init) in stars_init[['x_pix', 'y_pix', 'sigma_pix']].itertuples():
        subimage = get_square_subimage(image=image, position=(x_init, y_init), width=width)
        # If the star was too close to the image edge to extract the square subimage, skip the star.
        # Otherwise, compute the initial position for the star relative to the subimage.
        # The initial position relative to the subimage is an integer pixel.
        (height_actl, width_actl) = subimage.shape
        if (width_actl != width) or (height_actl != width):
            # noinspection PyShadowingBuiltins
            tmp_vars = collections.OrderedDict(idx=idx, x_init=x_init, y_init=y_init,
                                               sigma_init=sigma_init, box_pix=box_pix,
                                               width=width, width_actl=width_actl, height_actl=height_actl)
            logger.debug(("Star was too close to the edge of the image to extract a square subimage. Skipping star. " +
                         "Program variables: {tmp_vars}").format(tmp_vars=tmp_vars))
            continue
        x_init_sub = (width_actl - 1) / 2
        y_init_sub = (height_actl - 1) / 2
        # Compute the centroid position and standard deviation sigma for the star relative to the subimage.
        # using the selected method. Subtract background to fit counts only belonging to the source.
        subimage = subtract_subimage_background(subimage, threshold_sigma)
        if method == 'fit_2dgaussian':
            # Test results: 2014-08-11, STH
            # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
            # - For varying subimages, position converges to within 0.01 pix of final solution at 11x11 subimage.
            # - For varying subimages, sigma converges to within 0.05 pix of final solution at 11x11 subimage.
            # - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
            # - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
            # - For 11x11 subimage, method takes ~25 ms. Method scales \propto box_pix.
            # Method description:
            # - See photutils [1]_ and astropy [2]_.
            # - To calculate the standard deviation for the 2D Gaussian:
            # zvec = xvec + yvec
            # xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
            # ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
            # Notation: x = |xvec|, y = |yvec|, z = |zvec|
            # ==> Var(z) = Var(x + y)
            #            = Var(x) + Var(y) + 2*Cov(x, y)
            #            = Var(x) + Var(y) since Cov(x, y) = 0 due to orthogonality.
            # ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
            fit = morphology.fit_2dgaussian(subimage)
            (x_finl_sub, y_finl_sub) = (fit.x_mean, fit.y_mean)
            sigma_finl_sub = math.sqrt(fit.x_stddev ** 2.0 + fit.y_stddev ** 2.0)
        elif method == 'fit_bivariate_normal':
            # Test results: 2014-08-11, STH
            # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
            # - For varying subimages, position converges to within 0.02 pix of final solution at 11x11 subimage.
            # - For varying subimages, sigma converges to within 0.1 pix of final solution at 11x11 subimage.
            # - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
            # - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
            # - For 11x11 subimage, method takes ~450 ms. Method scales \propto box_pix**2.
            # Method description:
            # - Model the photons hitting the pixels of the subimage and
            # robustly fit a bivariate normal distribution.
            # - Conservatively assume that photons hit each pixel, even those of the star,
            # with a uniform distribution. See [3]_, [4]_.
            # - Seed the random number generator only once per call to this method for reproducibility.
            # - To calculate the standard deviation for the 2D Gaussian:
            # zvec = xvec + yvec
            # xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
            # ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
            # Notation: x = |xvec|, y = |yvec|, z = |zvec|
            #   ==> Var(z) = Var(x + y)
            #              = Var(x) + Var(y) + 2*Cov(x, y)
            #              = Var(x) + Var(y)
            #                since Cov(x, y) = 0 due to orthogonality.
            #   ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
            x_dist = []
            y_dist = []
            (height_actl, width_actl) = subimage.shape
            np.random.seed(0)
            for y_idx in xrange(height_actl):
                for x_idx in xrange(width_actl):
                    pixel_counts = np.rint(subimage[y_idx, x_idx])
                    x_dist_pix = scipy.stats.uniform(x_idx - 0.5, 1)
                    x_dist.extend(x_dist_pix.rvs(pixel_counts))
                    y_dist_pix = scipy.stats.uniform(y_idx - 0.5, 1)
                    y_dist.extend(y_dist_pix.rvs(pixel_counts))
            (mu, sigma1, sigma2, alpha) = astroML_stats.fit_bivariate_normal(x_dist, y_dist, robust=True)
            (x_finl_sub, y_finl_sub) = mu
            sigma_finl_sub = math.sqrt(sigma1 ** 2.0 + sigma2 ** 2.0)
        # # NOTE: 2014-08-10, STH
        # # The following methods have been commented out because they do not provide an estimate for the star's
        # # standard deviation as a 2D Gaussian.
        # # elif method == 'centroid_com':
        # # `centroid_com` : Method is from photutils [1]_. Return the centroid from computing the image moments.
        # # Method is very fast but only accurate between 7 <= `box_pix` <= 11 given `sigma`=1 due to
        # # sensitivity to outliers.
        # # Test results: 2014-08-09, STH
        # # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
        # #   seeing = 1.4 arcsec.
        #     # - For varying subimages, method does not converge to final centroid solution.
        #     # - For 7x7 to 11x11 subimages, centroid solution agrees with centroid_2dg centroid solution within
        #     #   +/- 0.01 pix, but then diverges from solution with larger subimages.
        #     #   Method is susceptible to outliers.
        #     # - For 7x7 subimages, method takes ~3 ms per subimage. Method is invariant to box_pix and always
        #     #   takes ~3 ms.
        #     # (x_finl_sub, y_finl_sub) = morphology.centroid_com(subimage)
        #     # elif method == 'fit_max_phot_flux':
        #     # `fit_max_phot_flux` : Method is from Mike Montgomery, UT Austin, 2014. Return the centroid from
        #     # computing the centroid that yields the largest photometric flux. Method is fast, but,
        #     # as of 2014-08-08 (STH), implementation is inaccurate by ~0.1 pix (given `sigma`=1, `box_pix`=7),
        #     # and method is possibly sensitive to outliers.
        #     # Test results: 2014-08-09, STH
        #     # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
        #     #   seeing = 1.4 arcsec.
        #     # - For varying subimages, method converges to within +/- 0.0001 pix of final centroid solution at
        #     #   7x7 subimage, however final centroid solution disagrees with other methods' centroid solutions.
        #     # - For 7x7 subimage, centroid solution disagrees with centroid_2dg centroid solution for 7x7 subimage
        #     #   by ~0.1 pix. Method may be susceptible to outliers.
        #     # - For 7x7 subimage, method takes ~130 ms. Method scales \propto box_pix.
        #     # TODO: Test different minimization methods
        #     def obj_func(subimage, position, radius):
        #         """Objective function to minimize: -1*photometric flux from star.
        #         Assumed to follow a 2D Gaussian point-spread function.
        #
        #         Parameters
        #         ----------
        #         subimage : array_like
        #             2D subimage of image. Used only by `obj_func`.
        #         position : list or array of a tuple
        #             Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``,
        #                 i.e. [x_pix, y_pix] [1]_, [2]_.
        #             Used by both `obj_func` and `jac_func`.
        #         radius : float
        #             The radius of the aperture [1]_. Used only by `obj_func`.
        #
        #         Returns
        #         -------
        #         flux_neg : float
        #             Negative flux computed by photutils [2]_.
        #
        #         References
        #         ----------
        #         .. [1] http://photutils.readthedocs.org/en/latest/api/
        #                photutils.CircularAperture.html#photutils.CircularAperture
        #         .. [2] http://photutils.readthedocs.org/en/latest/api/
        #                photutils.aperture_photometry.html#photutils.aperture_photometry
        #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
        #
        #         """
        #         aperture = ('circular', radius)
        #         (flux_table, aux_dict) = photutils.aperture_photometry(subimage, position, aperture)
        #         flux_neg = -1. * flux_table['aperture_sum'].data
        #         return flux_neg
        #
        #     def jac_func(subimage, position, radius, eps=0.005):
        #         """Jacobian of the objective function for fixed radius.
        #         Assumed to follow a 2D Gaussian point-spread function.
        #
        #         Parameters
        #         ----------
        #         subimage : array_like
        #             2D subimage of image. Used only by `obj_func`
        #         position : list or array of a tuple
        #             Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``,
        #                 i.e. [x_pix, y_pix] [1]_, [2]_.
        #             Used by both `obj_func` and `jac_func`.
        #         radius : float
        #             The radius of the aperture [1]_. Used only by `obj_func`.
        #         eps : float
        #             Epsilon value for computing the change in the gradient. Used only by `jac_func`.
        #
        #         Returns
        #         -------
        #         jac : numpy.ndarray
        #             Jacobian of obj_func as ``numpy.ndarray`` [dx, dy].
        #
        #         References
        #         ----------
        #         .. [1] http://photutils.readthedocs.org/en/latest/api/
        #                photutils.CircularAperture.html#photutils.CircularAperture
        #         .. [2] http://photutils.readthedocs.org/en/latest/api/
        #                photutils.aperture_photometry.html#photutils.aperture_photometry
        #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
        #
        #         """
        #         try:
        #             [x_pix, y_pix] = position
        #         except ValueError:
        #             raise ValueError(("'position' must have the format [x_pix, y_pix]\n"+
        #                               "  position = {pos}").format(pos=position))
        #         jac = np.zeros(len(position))
        #         fxp1 = obj_func(subimage, (x_pix + eps, y_pix), radius)
        #         fxm1 = obj_func(subimage, (x_pix - eps, y_pix), radius)
        #         fyp1 = obj_func(subimage, (x_pix, y_pix + eps), radius)
        #         fym1 = obj_func(subimage, (x_pix, y_pix - eps), radius)
        #         jac[0] = (fxp1-fxm1)/(2.*eps)
        #         jac[1] = (fyp1-fym1)/(2.*eps)
        #         return jac
        #
        #     position = [x_init_sub, y_init_sub]
        #     radius = sigma_to_fwhm(sigma_init)
        #     res = scipy.optimize.minimize(fun=(lambda pos: obj_func(subimage, pos, radius)),
        #                                   x0=position,
        #                                   method='L-BFGS-B',
        #                                   jac=(lambda pos: jac_func(subimage, pos, radius)),
        #                                   bounds=((0, width), (0, height)))
        #     (x_finl_sub, y_finl_sub) = res.x
        else:
            raise AssertionError("Program error. Method not accounted for:\n{meth}".format(meth=method))
        # Compute the centroid coordinates relative to the entire image.
        # Return the dataframe with centroid coordinates and sigma.
        (x_offset, y_offset) = (x_finl_sub - x_init_sub,
                                y_finl_sub - y_init_sub)
        (x_finl, y_finl) = (x_init + x_offset,
                            y_init + y_offset)
        sigma_finl = sigma_finl_sub
        stars_finl.loc[idx, ['x_pix', 'y_pix', 'sigma_pix']] = (x_finl, y_finl, sigma_finl)
    return stars_finl


def drop_duplicate_stars(stars):
    """
    Stars within 1 sigma of each other are assumed to be the same star.
    :type stars: object
    :param stars:
    :return stars:
    """
    # Remove all NaN values and sort `stars` by `sigma_pix` so that sources with larger sigma contain the
    # duplicate sources with smaller sigma. `stars` is updated at the end of each iteration.
    # noinspection PyUnresolvedReferences
    stars.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    # noinspection PyTypeChecker
    if len(stars) > 1:
        for (idx, row) in stars.sort(columns=['sigma_pix']).iterrows():
            sum_sqr_diffs = \
                np.sum(
                    np.power(
                        np.subtract(
                            stars[['x_pix', 'y_pix']].drop(idx, inplace=False),
                            row.loc[['x_pix', 'y_pix']]),
                        2.0),
                    axis=1)
            minssd = sum_sqr_diffs.min()
            idx_minssd = sum_sqr_diffs.idxmin()
            if (minssd < row.loc['sigma_pix']) and (minssd < stars.loc[idx_minssd, 'sigma_pix']):
                if row.loc['sigma_pix'] >= stars.loc[idx_minssd, 'sigma_pix']:
                    raise AssertionError(("Program error. Indices of degenerate stars were not dropped.\n" +
                                          "row:\n{row}\nstars:\n{stars}").format(row=row, stars=stars))
                logger.debug("Dropping duplicate star: {row}".format(row=row))
                stars.drop(idx, inplace=True)
    else:
        # noinspection PyTypeChecker
        logger.debug("No duplicate stars to drop. num_stars = {num}".format(num=len(stars)))
    return stars


def translate_images_1to2(image1, image2):
    """
    Determine image translation from phase correlation.
    Adapted from http://www.lfd.uci.edu/~gohlke/code/imreg.py.html
    """
    # TODO: complete docstring
    # Check input.
    if image1.shape != image2.shape:
        raise IOError(("Images must have the same shape:\n" +
                       "image1.shape = {s1}\n" +
                       "image2.shape = {s2}").format(s1=image1.shape, s2=image2.shape))
    if image1.ndim != 2:
        raise IOError(("Images must be 2D:\n" +
                       "image1.ndim = {n1}\n" +
                       "image2.ndim = {n2}").format(n1=image1.ndim, n2=image2.ndim))
    # Compute the maximum phase correlation for to determine the image translation.
    # Translation: delta = final - initial = image2 - image1
    # The first estimate for image translation is to an integer pixel.
    # Note: numpy is row-major: (y_pix, x_pix)
    shape = image1.shape
    # TODO: may fail for image dimensions not divisible by 2. test.
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    ir = abs(np.fft.ifft2((f1.conjugate() * f2) / (abs(f1) * abs(f2))))
    (dy_int, dx_int) = np.unravel_index(int(np.argmax(ir)), shape)
    # Use center_stars to get the subpixel estimate for the translation. Sub-pixel precision for the image translation
    # allows more precise star identification between images. (A 2D Gaussian fit is not correct.
    # Do not use 'sigma_pix' as uncertainty.)
    # Tile the phase correlation image when estimating subpixel translation since the coordinate for maximum phase
    # correlation is usually near the domain edge. (Tile the image, don't mirror, since the phase correlation is
    # continuous across image boundaries.)
    tiled = np.tile(ir, (3, 3))
    (tiled_offset_y, tiled_offset_x) = shape
    # Returned order should be (x, y) for to match rest of utils convention.
    (tiled_dx_int, tiled_dy_int) = np.add((tiled_offset_x, tiled_offset_y), (dx_int, dy_int))
    translation = center_stars(image=tiled,
                               stars=pd.DataFrame([[tiled_dx_int, tiled_dy_int, 1.0]],
                                                  columns=['x_pix', 'y_pix', 'sigma_pix']))
    if len(translation) != 1:
        raise AssertionError(("Program error. `translation` dataframe should have only one element,\n" +
                              "the maximum phase correlation. translation:\n" +
                              "{df}").format(df=translation))
    (dx_pix, dy_pix) = np.subtract(translation.loc[0, ['x_pix', 'y_pix']], (tiled_offset_x, tiled_offset_y))
    # Because phase correlation image is continuous across boundaries, restrict all coordinates to be relative to
    # first quandrant (containing (1,1)).
    if dy_pix > shape[0] / 2.0:
        dy_pix -= shape[0]
    if dx_pix > shape[1] / 2.0:
        dx_pix -= shape[1]
    logger.debug(("Image translation: image1_coords - image2_coords = (dx_pix, dy_pix)" +
                  " = {tup}").format(tup=(dx_pix, dy_pix)))
    # noinspection PyRedundantParentheses
    return (dx_pix, dy_pix)


# noinspection PyUnresolvedReferences
def gaussian_weights(width=11, sigma=3):
    """
    Weight pixels depending on distance to center pixel.
    Sigma is standard deviation of Gaussian weighting function
    Useful for localized image matching.
    http://scikit-image.org/docs/dev/auto_examples/plot_matching.html
    http://en.wikipedia.org/wiki/Gaussian_blur
    """
    left = int(math.floor(width / 2.0))
    right = int(math.ceil(width / 2.0))
    bottom = int(math.floor(width / 2.0))
    top = int(math.ceil(width / 2.0))
    (y_pix, x_pix) = np.mgrid[-left:right, -bottom:top]
    weights = np.zeros(y_pix.shape, dtype=np.double)
    weights[:] = (1.0 / (2.0 * np.pi * sigma ** 2.0)) * np.exp(
        -0.5 * ((x_pix ** 2.0 / sigma ** 2.0) + (y_pix ** 2.0 / sigma ** 2.0)))
    return weights


# noinspection PyUnresolvedReferences
def _plot_matches(image1, image2, stars1, stars2):
    """
    Visualize image matching.
    http://scikit-image.org/docs/dev/auto_examples/plot_matching.html
    """
    # Check input. Reindex so pandas dataframe indices match numpy ndarray indices.
    if (stars1['verif1to2'] != stars2['verif2to1']).any():
        raise IOError(("Dataframe indices are not verified as matching by 'verif1to2', 'verif2to1' columns:\n" +
                       "stars1: {stars1}\n" +
                       "stars2: {stars2}\n").format(stars1=stars1, stars2=stars2))
    (fig, axes) = plt.subplots(nrows=2, ncols=1)
    stars1.reset_index(drop=True, inplace=True)
    stars2.reset_index(drop=True, inplace=True)
    keypoints1 = stars1[['y_pix', 'x_pix']].values
    keypoints2 = stars2[['y_pix', 'x_pix']].values
    # noinspection PyPep8
    matches = stars1[stars1['verif1to2'] == True].index.tolist()
    feature.plot_matches(ax=axes[0], image1=image1, image2=image2, keypoints1=keypoints1, keypoints2=keypoints2,
                         matches=np.column_stack((matches, matches)), keypoints_color='yellow',
                         matches_color='green')
    axes[0].axis('off')
    axes[0].set_title('Verified matches (left: image1, right: image2)')
    # noinspection PyPep8
    not_matches = stars1[stars1['verif1to2'] == False].index.tolist()
    feature.plot_matches(ax=axes[1], image1=image1, image2=image2, keypoints1=keypoints1, keypoints2=keypoints2,
                         matches=np.column_stack((not_matches, not_matches)), keypoints_color='yellow',
                         matches_color='red')
    axes[1].axis('off')
    axes[1].set_title('Unverified matches (left: image1, right: image2)')
    plt.show()
    return None


# noinspection PyUnresolvedReferences
def match_stars(image1, image2, stars1, stars2, test=False):
    """
    Match stars within two images.
    http://scikit-image.org/docs/dev/auto_examples/plot_matching.html
    """
    # Check input.
    stars1.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    stars2.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    num_stars1 = len(stars1)
    num_stars2 = len(stars2)
    if num_stars1 < 1:
        raise IOError(("stars1 must have at least one star.\n" +
                       "stars1 = {stars1}").format(stars1=stars1))
    if num_stars1 != num_stars2:
        logger.debug(("`image1` and `image2` have different numbers of stars. There may be clouds. " +
                      "num_stars1: {n1} num_stars2: {n2}").format(n1=num_stars1, n2=num_stars2))
    # Create heirarchical dataframe for tracking star matches. Match from star1 positions to star2 positions.
    # `stars` dataframe has the same number of stars as `stars1`: num_stars = num_stars1
    # Sort columns to permit heirarchical slicing.
    df_stars1 = stars1.copy()
    df_stars1['verif1to2'] = np.NaN
    df_tform1to2 = (stars1.copy())[['x_pix', 'y_pix']]
    df_tform1to2[:] = np.NaN
    df_stars2 = stars1.copy()
    df_stars2[:] = np.NaN
    df_stars2['idx2'] = np.NaN
    df_stars2['verif2to1'] = np.NaN
    df_stars2['minssd'] = np.NaN
    df_dict = {'stars1': df_stars1,
               'tform1to2': df_tform1to2,
               'stars2': df_stars2}
    stars = pd.concat(df_dict, axis=1)
    stars.sort_index(axis=1, inplace=True)
    # If any stars exist in stars2, match them...:
    if num_stars2 > 0:
        # Compute image transformation using only translation and transform coordinates of stars from image1 to image2.
        # TODO: allow user to give custom translation
        translation = translate_images_1to2(image1=image1, image2=image2)
        tform = skimage.transform.SimilarityTransform(translation=translation)
        stars.loc[:, ('tform1to2', ['x_pix', 'y_pix'])] = tform(stars.loc[:, ('stars1', ['x_pix', 'y_pix'])].values)
        pars = collections.OrderedDict(translation=tform.translation, rotation=tform.rotation, scale=tform.scale,
                                       params=tform.params)
        logger.debug("Transform parameters: {pars}".format(pars=pars))
        # Use least sum of squares to match stars. Verify that matched stars are within 1 sigma
        # of the centroid of stars2 and are matched 1-to-1.
        # TODO: Allow users to define stars by hand instead of by `find_stars`
        # TODO: Can't individually set pandas.DataFrame elements to True. Report bug?
        stars1_verified = pd.DataFrame(columns=stars1.columns)
        stars1_unverified = stars1.copy()
        stars2_verified = pd.DataFrame(columns=stars2.columns)
        stars2_unverified = stars2.copy()
        for (idx, row) in stars.iterrows():
            sum_sqr_diffs = \
                np.sum(
                    np.power(
                        np.subtract(
                            stars2[['x_pix', 'y_pix']],
                            row.loc['tform1to2', ['x_pix', 'y_pix']]),
                        2.0),
                    axis=1)
            minssd = sum_sqr_diffs.min()
            idx2_minssd = sum_sqr_diffs.idxmin()
            # Faint stars undersample the PSF given a noisy background and are calculated to have smaller sigma than
            # the actual sigma of the PSF. Thus, accept found stars up to 3 sigma away from the predicted coordinates
            # as being the matching star.
            if minssd < 3.0 * stars2.loc[idx2_minssd, 'sigma_pix']:
                row.loc['stars2'].update(stars2.loc[idx2_minssd])
                row.loc['stars2', 'idx2'] = idx2_minssd
                row.loc['stars2', 'minssd'] = minssd
                idx1 = idx
                row1 = stars1.loc[idx1]
                if idx1 not in stars1_verified.index:
                    stars1_verified.loc[idx1] = row1
                else:
                    raise AssertionError(("Program error. Star from stars1 already verified:\n" +
                                          "{row}").format(row=row))
                stars1_unverified.drop(idx1, inplace=True)
                row.loc['stars1', 'verif1to2'] = 1
                idx2 = row.loc['stars2', 'idx2'].astype(int)
                row2 = stars2.loc[idx2]
                if idx2 not in stars2_verified.index:
                    stars2_verified.loc[idx2] = row2
                else:
                    raise AssertionError(("Program error. Star from stars2 already verified:\n" +
                                          "{row}").format(row=row))
                stars2_unverified.drop(idx2, inplace=True)
                row.loc['stars2', 'verif2to1'] = 1
            else:
                row.loc['stars2'].update(row.loc['tform1to2'])
                row.loc['stars1', 'verif1to2'] = 0
                row.loc['stars2', 'verif2to1'] = 0
                logger.debug("Star not verified: {row}".format(row=row))
            # Save results and verify found matches.
            stars.loc[idx].update(row)
        # Check that all stars have been accounted for. Stars without matches have NaNs in 'star1' or 'star2'.
        # Sort columns to permit heirarchical slicing.
        if (len(stars1_verified) != len(stars1)) or (len(stars1_unverified) != 0):
            logger.debug(("Not all stars in stars1 were verified as matching stars in stars2." +
                          " stars1_unverified: {s1u}").format(s1u=stars1_unverified))
        if (len(stars2_verified) != len(stars2)) or (len(stars2_unverified) != 0):
            logger.debug(("Not all stars in stars2 were verified as matching stars in stars1." +
                          " stars2_unverified: {s2u}").format(s2u=stars2_unverified))
            df_dict = {'stars1': stars['stars1'],
                       'tform1to2': stars['tform1to2'],
                       'stars2': (stars['stars2']).append(stars2_unverified, ignore_index=True)}
            stars = pd.concat(df_dict, axis=1)
    # ...otherwise there are no stars in stars2.
    else:
        logger.debug("No stars in stars2. Assuming stars2 (x, y) are same as stars1 (x, y).")
        stars.loc[:, ('tform1to2', ['x_pix', 'y_pix'])] = stars.loc[:, ('stars1', ['x_pix', 'y_pix'])].values
        stars.loc[:, ('stars2', ['x_pix', 'y_pix'])] = stars.loc[:, ('tform1to2', ['x_pix', 'y_pix'])].values
        stars.loc[:, ('stars1', 'verif1to2')] = 0
        stars.loc[:, ('stars2', 'verif2to1')] = 0
    stars.sort_index(axis=1, inplace=True)
    stars[('stars1', 'verif1to2')] = (stars[('stars1', 'verif1to2')] == 1)
    stars[('stars2', 'verif2to1')] = (stars[('stars2', 'verif2to1')] == 1)
    if test:
        _plot_matches(image1=image1, image2=image2, stars1=stars['stars1'], stars2=stars['stars2'])
    # Report results.
    df_dict = {'stars1': stars['stars1'],
               'stars2': stars['stars2'].drop(['idx2', 'minssd'], axis=1)}
    matched_stars = pd.concat(df_dict, axis=1)
    return matched_stars
