#!/usr/bin/env python
"""
Run online analysis.
"""

from __future__ import print_function
import os
import time
import argparse
import read_spe
import spe_process
import lc_online2

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
                    +"  If in IPython Notebook, click \'Interrupt Kernel\'.")
        sleep_time = args.sleep # seconds
        sleep_msg = ("INFO: Sleeping for {num} seconds.").format(num=sleep_time)
        spe = read_spe.File(args.fpath)
        #TODO: make open method in read_spe so that 'with open() as f' works to close file safely.
        is_first_iter = True
        while True:
            # TODO: to run incrementally, reduce duplication between top-level main script
            # and imorted modules.
            # get num frames, get last frame, send to spe_process
            num_frames = spe.get_num_frames()
            if args.frame_end == -1:
                args.frame_end = num_frames - 1
            if not is_first_iter:
                args.frame_start = frame_end_old
                args.frame_end = num_frames - 1
            try:
                spe_process.main(args)
                lc_online2.main(args)
            # IndexError or ValueError can be raised by lc_online2 due to namespace conflicts with spe_process.
            # TODO: Resolve namespace issues by sharing state info within modules using classes.
            except IndexError:
                spe_process.main(args)
                lc_online2.main(args)
            except ValueError:
                spe_process.main(args)
                lc_online2.main(args)
            if args.verbose:
                print(view_msg)
                print(stop_msg)
                print(sleep_msg)
            time.sleep(sleep_time)
            # Save variables from last iteration.
            num_frames_old = num_frames
            frame_start_old = args.frame_start
            frame_end_old = args.frame_end
            is_first_iter = False
    return None

if __name__ == '__main__':
    # TODO: read config file from configparser, STH
    arg_default_map = {}
    arg_default_map['fcoords'] = "phot_coords.txt"
    arg_default_map['flc']     = "lightcurve.txt"
    arg_default_map['frame_start'] = 0
    arg_default_map['frame_end']   = -1
    arg_default_map['flc_pdf'] = "lc.pdf"
    arg_default_map['fap_pdf'] = "aperture.pdf"
    arg_default_map['sleep']   = 1
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=("Continuously read .spe file and do aperture photometry."
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
                        help=(("Output fixed-width-format text file"
                               +" with columns of star intensities by aperture size.\n"
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
    parser.add_argument("--flc_pdf",
                        default=arg_default_map['flc_pdf'],
                        help=(("Output .pdf file with plots of the lightcurve.\n"
                              +"Default: {fname}").format(fname=arg_default_map['flc_pdf'])))
    parser.add_argument("--fap_pdf",
                        default=arg_default_map['fap_pdf'],
                        help=(("Output .pdf file with plot of scatter vs aperture size.\n"
                               +"Default: {fname}").format(fname=arg_default_map['fap_pdf'])))
    parser.add_argument("--sleep", "-s",
                        default=arg_default_map['sleep'],
                        type=float,
                        help=(("Number of seconds to sleep before reducing new frames.\n"
                               +"Default: {num}").format(num=arg_default_map['sleep'])))
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
    if os.path.isfile(args.flc):
        print(("INFO: Overwriting lightcurve file:\n"
               +" {flc}").format(flc=args.flc))
        # TODO: Just rename with timestamp
        os.remove(args.flc)
    main(args)
