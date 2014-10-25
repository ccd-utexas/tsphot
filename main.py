#!/usr/bin/env python
"""Main module for time-series photometry pipeline.

See Also
--------
read_spe : Module for reading SPE files.
utils : Utilities for time-series photometry.

Notes
-----
noinspection : Comments are created by PyCharm to flag permitted code inspection violations.
docstrings : This module's documentation follows the `numpy` doc example [1]_.
logger : stdout and stderr are managed by a logger [3]_.
STAGE : Major pipeline stages are flagged by the logger as `STAGE`. The associated value for each stage is separated
    by underscores.
TODO : Flag all to-do items with 'TODO:' in the code body (not the docstring) so that they are flagged when using
    an IDE.
'See Also' : Methods describe their relationships to each other within their docstrings under the 'See Also' section.
    All methods should be connected to at least one other method within this module [2]_.

References
----------
.. [1] https://github.com/numpy/numpy/blob/master/doc/example.py
.. [2] http://en.wikipedia.org/wiki/Pipeline_(software)
.. [3] https://docs.python.org/2/library/logging.html

"""

# Forwards compatibility imports.
from __future__ import division, absolute_import, print_function

# Standard library imports.
import os
import time
import math
import json
import pickle
import logging
import argparse
import collections

# External package imports. Grouped procedurally then categorically.
import astropy
import ccdproc

# Internal package imports.
import utils


# noinspection PyShadowingNames
def main(fconfig, rereduce=False, verbose=False):
    """Time-series photometry pipeline.

    Parameters
    ----------
    fconfig : string
        Path to input configuration file as .json.
    rereduce : {False}, bool, optional
        Re-reduce all files. Overwrite previously reduced files. If false, use previously reduced files.
    verbose : {False}, bool, optional
        Print startup 'INFO:' messages to stdout.

    Returns
    -------
    None

    Notes
    -----
    Call as top-level script. Example usage:
        $ python main.py --fconfig path/to/config.json -v

    """
    # TODO: write out fits files
    # Read configuration file.
    # Use binary read-write for cross-platform compatibility. Use Python-style indents in the JSON file.
    if verbose:
        print("INFO: Reading configuration file {fpath}".format(fpath=fconfig))
    with open(fconfig, 'rb') as fp:
        config_settings = json.load(fp, object_pairs_hook=collections.OrderedDict)
    if verbose:
        print("INFO: Configuration file settings: {settings}".format(settings=config_settings))
    # Check configuration file.
    if verbose:
        print("INFO: Checking configuration settings.")
    utils.check_reduce_config(dobj=config_settings)
    ################################################################################
    # Create logger.
    # Note: For root-level logger, use `getLogger()`, not `getLogger(__name__)`
    # http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
    # TODO: make stdout from logger look like output to file.
    if verbose:
        print("INFO: stdout now controlled by logger.")
    logger = logging.getLogger()
    logger.setLevel(level=getattr(logging, config_settings['logging']['level'].upper()))
    fmt = '"%(asctime)s","%(name)s","%(levelname)s","%(message)s"'
    formatter = logging.Formatter(fmt=fmt)
    formatter.converter = time.gmtime
    flog = config_settings['logging']['filename']
    if flog is not None:
        fhandler = logging.FileHandler(filename=flog, mode='ab')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    logger.info("STAGE: BEGIN_LOG")
    logger.info("Log format: {fmt}".format(fmt=fmt.replace('\"', '\'')))
    logger.info("Log date format: default ISO 8601, UTC")
    logger.info("Configuration file settings: {settings}".format(settings=config_settings))
    ################################################################################
    # Create master calibration (calib.) frames.
    # TODO: parallelize
    logger.info("STAGE: MASTER_CALIBRATIONS")
    calib_fpath = config_settings['calib']
    master_fpath = config_settings['master']
    master_ccddata = {}
    for imtype in calib_fpath:
        cfpath = calib_fpath[imtype]
        mfpath = master_fpath[imtype]
        # noinspection PyUnusedLocal
        do_master = None
        # If not rereducing.
        if not rereduce:
            # If master calib. frame files already exists, read them.
            if (mfpath is not None) and os.path.isfile(mfpath):
                logger.info("Reading master {imtype} from: {fpath}".format(imtype=imtype, fpath=mfpath))
                with open(mfpath, 'rb') as fp:
                    master_ccddata[imtype] = pickle.load(fp)
                do_master = False
            # Otherwise if calib. frames exist, create master calib. frames.
            elif (cfpath is not None) and os.path.isfile(cfpath):
                do_master = True
            # Otherwise set the master calib. frame as None.
            else:
                master_ccddata[imtype] = None
                do_master = False
        # Otherwise rereducing.
        else:
            # If calib. frames exist, create master calib. frames.
            if (cfpath is not None) and os.path.isfile(cfpath):
                do_master = True
            # Otherwise set the master calib. frame as None.
            else:
                master_ccddata[imtype] = None
                do_master = False
        # Create master calib. frame if needed.
        # If master calib. frame file is specified, save master calib. frame to file.
        if do_master:
            if (cfpath is None) or (not os.path.isfile(cfpath)):
                raise AssertionError(("Program error. Calibration frames must exist to make master " +
                                      "{imtype}: {fpath}").format(imtype=imtype, fpath=cfpath))
            logger.info("Creating master {imtype} from: {fpath}".format(imtype=imtype, fpath=cfpath))
            dobj = utils.spe_to_dict(fpath=cfpath)
            master_ccddata[imtype] = utils.create_master_calib(dobj=dobj)
            if mfpath is not None:
                logger.info("Writing master {imtype} to: {fpath}".format(imtype=imtype, fpath=mfpath))
                with open(mfpath, 'wb') as fp:
                    pickle.dump(master_ccddata[imtype], fp)
        if do_master is None:
            raise AssertionError("Program error. Not all cases for do_master are accounted for.")
    ################################################################################
    # Reduce object data and clean cosmic rays.
    logger.info("STAGE: REDUCE_DATA_AND_CLEAN_COSMIC_RAYS")
    rawfpath = config_settings['object']['raw']
    redfpath = config_settings['object']['reduced']
    # noinspection PyUnusedLocal
    do_reduction = None
    # If not rereducing.
    if not rereduce:
        # If reduced and cleaned object file already exists, read it.
        if (redfpath is not None) and os.path.isfile(redfpath):
            logger.info("Reading reduced object data from: {fpath}".format(fpath=redfpath))
            with open(redfpath, 'rb') as fp:
                object_ccddata = pickle.load(fp)
            do_reduction = False
        # Otherwise reduced and cleaned object frames do not yet exist and need to be created.
        else:
            do_reduction = True
    # Otherwise rereducing.
    else:
        do_reduction = True
    # If reduced and cleaned object frames need to be created:
    # Reduce and clean object frames using all available master calib. frames.
    # If reduced object file is specified, save reduced and cleaned object frames to file.
    # `reduce_ccddata` handles cases where master calibration frames do not exist.
    # TODO: use hdf5 instead of spe
    if do_reduction:
        logger.info("Reading raw object data from: {fpath}".format(fpath=rawfpath))
        object_ccddata = utils.spe_to_dict(rawfpath)
        dark_exptime = None
        if master_ccddata['dark'] is not None:
            dark_spe_footer_xml = master_ccddata['dark'].meta['footer_xml']
            dark_exptime = utils.get_exptime_prog(spe_footer_xml=dark_spe_footer_xml) * astropy.units.second
        flat_exptime = None
        if master_ccddata['flat'] is not None:
            flat_spe_footer_xml = master_ccddata['flat'].meta['footer_xml']
            flat_exptime = utils.get_exptime_prog(spe_footer_xml=flat_spe_footer_xml) * astropy.units.second
        object_spe_footer_xml = object_ccddata['footer_xml']
        object_exptime = utils.get_exptime_prog(spe_footer_xml=object_spe_footer_xml) * astropy.units.second
        exp_times = dict(dobj_exptime=object_exptime, dark_exptime=dark_exptime, flat_exptime=flat_exptime)
        logger.info("Exposure times: {exp_times}".format(exp_times=exp_times))
        logger.info("Reducing data.")
        object_ccddata = utils.reduce_ccddata(dobj=object_ccddata, dobj_exptime=object_exptime,
                                              bias=master_ccddata['bias'],
                                              dark=master_ccddata['dark'], dark_exptime=dark_exptime,
                                              flat=master_ccddata['flat'], flat_exptime=flat_exptime)
        # TODO: calculate gain and readnoise. correct for gain.
        # TODO: Make a closure/class to track progress.
        # TODO: for online analysis, skip cleaning cosmic rays
        logger.info("Cleaning cosmic rays.")
        print_progress = utils.define_progress(dobj=object_ccddata)
        sorted_image_keys = sorted([key for key in object_ccddata.keys() if isinstance(object_ccddata[key],
                                                                                       ccdproc.CCDData)])
        for key in sorted_image_keys:
            # TODO: give dict with readnoise, gain
            # TODO: save ray_mask in ccd_data
            (object_ccddata[key].data, ray_mask) = utils.remove_cosmic_rays(object_ccddata[key].data)
            print_progress(key=key)
        if redfpath is not None:
            logger.info("Writing reduced object data to: {fpath}".format(fpath=redfpath))
            with open(redfpath, 'wb') as fp:
                pickle.dump(object_ccddata, fp)
    if do_reduction is None:
        raise AssertionError("Program error. Not all cases for do_reduction are accounted for.")
    ################################################################################
    # Create timeseries.
    # TODO: check programmed/actual exposure times since pulses could have been missed
    # TODO: check default experiments with footer metadata to confirm correct experiment settings for calib. frames
    # TODO: finish importing code from ipython notebook
    # logger.info("STAGE: CALCULATE_TIMESERIES")
    # logger.info("Getting timestamps and calculating timeseries.")
    # timestamps, timeseries = utils.timestamps_timeseries(dobj=object_ccddata)
    # logger.info("Making lightcurve.")
    # lightcurve = utils.make_lightcurve(timestamps=timestamps, timeseries=timeseries, target_index=target_index)

    # Clean up.
    if flog is not None:
        # noinspection PyUnboundLocalVariable
        logger.removeHandler(fhandler)
    return None


if __name__ == '__main__':
    defaults = {'fconfig': 'reduce_config.json'}
    parser = argparse.ArgumentParser(description="Read configuration file and reduce data.")
    parser.add_argument('--fconfig',
                        default=defaults['fconfig'],
                        help=(("Input JSON configuration file for data reduction.\n" +
                               "Default: {dflt}").format(dflt=defaults['fconfig'])))
    parser.add_argument('--rereduce',
                        action='store_true',
                        help=("Re-reduce all files. Overwrite previously reduced files.\n" +
                              "If option omitted, use previously reduced files."))
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="Print startup 'INFO:' messages to stdout.")
    args = parser.parse_args()
    if args.verbose:
        print("INFO: Arguments: {args}".format(args=args))
    if not os.path.isfile(args.fconfig):
        raise IOError("Configuration file does not exist: {fpath}".format(fpath=args.fconfig))
    (fbase, ext) = os.path.splitext(args.fconfig)
    if ext != '.json':
        raise IOError("Configuration file extension is not '.json': {fpath}".format(fpath=args.fconfig))
    main(fconfig=args.fconfig, rereduce=args.rereduce, verbose=args.verbose)
