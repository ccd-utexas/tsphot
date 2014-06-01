#!/usr/bin/env python
"""
Run online analysis for a .spe file.
"""

from __future__ import print_function
import os
import time
import argparse
import new_spe_process
import new_lc_online2

def main(args):
    """
    Call modules to do aperture photmetry then plot results.
    """
    if args.focus:
        # TODO: call modules for focusing
        pass
    else:
        cwd  = os.getcwd()
        view_msg = ("INFO: To view {fname}, open Chrome to:\n"
                    +"  file:///path/to/file.pdf\n"
                    +"  If using default --flc_pdf option, open Chrome to:\n"
                    +"  file://"+os.path.join(cwd, args.flc_pdf)).format(fname=args.flc_pdf)
        stop_msg = ("INFO: To stop program, hit Ctrl-C\n"
                    +"  If in IPython Notebook, click 'Interrupt Kernel'.")
        sleep_time = args.sleep # seconds
        sleep_msg = ("INFO: Sleeping for {num} seconds.").format(num=sleep_time)
        while True:
            # Import main from spe_process, lc_online2 because not modularized.
            # TODO: break spe_process into more functions/class
            try:
                new_spe_process.main(args)
                new_lc_online2.main(args)
            # IndexError or ValueError can be raised by lc_online2 due to namespace conflicts with spe_process.
            # TODO: Resolve namespace issues by sharing state info within modules using classes.
            except IndexError:
                new_spe_process.main(args)
                new_lc_online2.main(args)
            except ValueError:
                new_spe_process.main(args)
                new_lc_online2.main(args)
            if args.verbose:
                print(view_msg)
                print(stop_msg)
                print(sleep_msg)
            time.sleep(sleep_time)
    return None

if __name__ == '__main__':
    # TODO: Orangize arguments better. 2014-05-31 STH
    defaults = {}
    defaults['fcoords'] = "phot_coords"
    defaults['flc']     = "lightcurve.app"
    defaults['flc_pdf'] = "lc.pdf"
    defaults['fap_pdf'] = "aperture.pdf"
    defaults['sleep']   = 60
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Continuously read .spe file and do aperture photometry."
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
                               +" with columns of star intensities by aperture size.\n"
                               +"Default: {fname}").format(fname=defaults['flc'])))
    parser.add_argument("--flc_pdf",
                        default=defaults['flc_pdf'],
                        help=(("Output .pdf file with plots of the lightcurve.\n"
                              +"Default: {fname}").format(fname=defaults['flc_pdf'])))
    parser.add_argument("--fap_pdf",
                        default=defaults['fap_pdf'],
                        help=(("Output .pdf file with plot of scatter vs aperture size.\n"
                               +"Default: {fname}").format(fname=defaults['fap_pdf'])))
    parser.add_argument("--sleep", "-s",
                        default=defaults['sleep'],
                        type=float,
                        help=(("Number of seconds to sleep before reducing new frames.\n"
                               +"Default: {num}").format(num=defaults['sleep'])))
    parser.add_argument("--focus",
                        action='store_true',
                        help=("Report the FWHM of the brightest star for focusing.\n"
                              +"Give 'fpath' arg to temporary focus file."))
    parser.add_argument("--verbose", "-v",
                        action='store_true',
                        help=("Print 'INFO:' messages to stdout."))
    args = parser.parse_args()
    if args.verbose:
        print("INFO: Arguments:")
        for arg in args.__dict__:
            print(' ', arg, args.__dict__[arg])
    if not os.path.isfile(args.fcoords):
        raise IOError(("File does not exist: {fname}").format(fname=args.fcoords))
    main(args)
