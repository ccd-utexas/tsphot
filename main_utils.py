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
TODO : Flag all to-do items with 'TODO:' in the code body (not the docstring) so that they are flagged when using
    an IDE.
'See Also' : Methods describe their relationships to each other within their docstrings under the 'See Also' section.
    All methods should be connected to at least one other method within this module [2]_.

References
----------
.. [1] https://github.com/numpy/numpy/blob/master/doc/example.py
.. [2] http://en.wikipedia.org/wiki/Pipeline_(software)

"""
# TODO: Rename as main.py when 'calibrate' branch is merged with master.

# Forwards compatibility imports.
from __future__ import division, absolute_import, print_function

# Standard library imports.
import os
import json
import pickle
import argparse

# External package imports. Grouped procedurally then categorically.

# Internal package imports.
import utils


def main(fconfig, rereduce=False, verbose=0):
    """Time-series photometry pipeline.

    Parameters
    ----------
     fconfig : string
        Path to input configuration file as .json.
    rereduce : {False}, bool, optional
        Re-reduce all files. Overwrite previously reduced files. If false, use previously reduced files.
    verbose : {0}, int
        Print 'INFO:' messages to stdout with increasing verbosity.

    Returns
    -------
    None

    Notes
    -----
    Call as top-level script. Example usage:
        $ python main_utils.py --fconfig path/to/config.json -v

    """
    # Read configuration file.
    # Use binary read-write for cross-platform compatibility. Use Python-style indents in the JSON file.
    if verbose >= 1:
        print("INFO: Reading config file {fconfig}".format(fconfig=fconfig))
    with open(fconfig, 'rb') as fp:
        # noinspection PyUnusedLocal
        config_settings = json.load(fp)
    if verbose >= 2:
        print("DEBUG: Config file contents:")
        print(config_settings)
    # Create master calibration frames.
    # TODO: parallelize
    if verbose >= 1:
        print("INFO: Creating master calibration files.")
    master_ccddata = {}
    for calib in sorted(config_settings['calib']):
        fpath = config_settings['calib'][calib]
        dobj = utils.spe_to_dict(fpath=fpath)
        master_ccddata[calib] = utils.create_master_calib(dobj=dobj)
    # TEST:
    print(master_ccddata.keys())
    # TODO: resume here
    return None


if __name__ == '__main__':
    # TODO: use logging
    defaults = {'fconfig': 'config.json'}
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Read configuration file and reduce data.")
    parser.add_argument('--fconfig',
                        default=defaults['fconfig'],
                        help=(("Input configuration file as .json.\n" +
                               "Default: {dflt}").format(dflt=defaults['fconfig'])))
    parser.add_argument('--rereduce',
                        action='store_true',
                        help=("Re-reduce all files. Overwrite previously reduced files.\n" +
                              "If option omitted, use previously reduced files."))
    parser.add_argument('--verbose', '-v',
                        action='count',
                        help="Print 'INFO:' messages to stdout.")
    args = parser.parse_args()
    if not os.path.isfile(args.fconfig):
        raise IOError("Configuration file does not exist: {fconfig}".format(fconfig=args.fconfig))
    (fconfig_base, ext) = os.path.splitext(args.fconfig)
    if ext != '.json':
        raise IOError("Configuration file extension is not '.json': {fconfig}".format(fconfig=args.fconfig))
    if args.verbose:
        print("INFO: Arguments:")
        for arg in args.__dict__:
            print('', arg, args.__dict__[arg])
    main(sorted(**args.__dict__))
