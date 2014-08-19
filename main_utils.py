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
import argparse
import json

# External package imports. Grouped procedurally then categorically.

# Internal package imports.
import utils


# noinspection PyShadowingNames
def main(args):
    """Time-series photometry pipeline.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Class with arguments as attributes.

    Returns
    -------
    None

    See Also
    --------

    Notes
    -----


    """
    # TODO: reduce then create lightcurve
    # Use binary read-write for cross-platform compatibility. Use Python-style indents in the JSON file.
    with open(args.fconfig, 'rb') as fp:
        # noinspection PyUnusedLocal
        args_fconfig = utils.dict_to_class(json.load(fp))
    # TODO: resume here
    return None


if __name__ == '__main__':
    # TODO: use logging
    # noinspection PyDictCreation
    defaults = {}
    defaults['fconfig'] = "config.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Read configuration file and reduce data.")
    parser.add_argument("--fconfig",
                        default=defaults['fconfig'],
                        help=(("Input configuration file as .json.\n" +
                               "Default: {default}").format(default=defaults['fconfig'])))
    parser.add_argument("--verbose", "-v",
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
            print("", arg, args.__dict__[arg])
    main(args)
