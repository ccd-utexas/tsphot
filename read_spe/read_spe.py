#!/usr/bin/env python
"""
Read .SPE file into numpy array.

Adapted from http://wiki.scipy.org/Cookbook/Reading_SPE_files
Offsets and names taken as from SPE 3.0 File Format Specification:
ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf

Note: Use with SPE 3.0. Not backwards compatible with SPE 2.X.
"""
# TODO: make test modules with test_yes/no_footer.spe files

from __future__ import print_function
from __future__ import division
import argparse
import os
import sys
import numpy as np
import pandas as pd
from lxml import objectify, etree
from datetime import datetime

class File(object):
    
    def __init__(self, fname):
        """
        Open file and load metadata from header and footer.
        """
        # For online analysis, read metadata from binary header.
        # For final reductions, read more complete metadata from XML footer.
        # TODO: check if ver 3.0, warn if not
        print("in __init__")
        self._fname = fname
        self._fid = open(fname, 'rb')
        self._load_header_metadata()
        self._load_footer_metadata()
        return None

    def _load_header_metadata(self):
        """
        Load SPE metadata from binary header as a pandas dataframe.
        Use metadata from header for online analysis
        since XML footer does not yet exist while taking data.
        Only the fields required for SPE 3.0 files are loaded. All other fields are numpy NaN.
        See SPE 3.0 File Format Specification:
        ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf
        """
        # file_header_ver and xml_footer_offset are
        # the only required header fields for SPE 3.0.
        # Header information from SPE 3.0 File Specification, Appendix A.
        # Read in CSV of header format without comments.
        ffmt = 'spe_30_header_format.csv'
        ffmt_base, ext = os.path.splitext(ffmt)
        ffmt_nocmts = ffmt_base + '_temp' + ext
        if not os.path.isfile(ffmt):
            raise IOError("SPE 3.0 header format file does not exist: {fname}".format(fname=ffmt))
        if not ext == '.csv':
            raise TypeError("SPE 3.0 header format file is not .csv: {fname}".format(fname=ffmt))
        with open(ffmt) as fcmts:
            # Make a temporary file without comments.
            with open(ffmt_nocmts, 'w') as fnocmts:
                for line in fcmts:
                    if line.startswith('#'):
                        continue
                    else:
                        fnocmts.write(line)
        self.header_metadata = pd.read_csv(ffmt_nocmts, sep=',')
        os.remove(ffmt_nocmts)
        binary_ntypes = {"8s": np.int8,
                         "8u": np.uint8,
                         "16s": np.int16,
                         "16u": np.uint16,
                         "32s": np.int32,
                         "32u": np.uint32,
                         "64s": np.int64,
                         "64u": np.uint64,
                         "32f": np.float32,
                         "64f": np.float64}
        # TODO: Efficiently read values and create column following
        # http://pandas.pydata.org/pandas-docs/version/0.13.1/cookbook.html
        # TODO: use zip and map to map read_at over arguments
        # Index values by offset byte position.
        values_by_offset = {}
        for idx in xrange(len(self.header_metadata)):
            offset = self.header_metadata["Offset"][idx]
            try:
                size = (self.header_metadata["Offset"][idx+1]
                        - self.header_metadata["Offset"][idx]
                        - 1)
            # Key error if at last value in the header
            except KeyError:
                size = 1
            ntype = binary_ntypes[self.header_metadata["Binary"][idx]]
            values_by_offset[offset] = self.read_at(offset, size, ntype)
        # Store only the values for the byte offsets required of SPE 3.0 files.
        # Read only first element of these values since for files written by LightField,
        # other elements and values from offets are 0.
        nan_array = np.empty(len(self.header_metadata))
        nan_array[:] = np.nan
        self.header_metadata["Value"] = pd.DataFrame(nan_array)
        spe_30_required_offsets = [6, 18, 34, 42, 108, 656, 658, 664, 678, 1446, 1992, 2996, 4098]
        for offset in spe_30_required_offsets:
            tf_mask = (self.header_metadata["Offset"] == offset)
            self.header_metadata["Value"].loc[tf_mask] = values_by_offset[offset][0]
        return None

    def _load_footer_metadata(self):
        """
        Load SPE metadata from XML footer as an lxml object.
        Use metadata from footer for final reductions
        since XML footer is more complete.
        """
        tf_mask = (self.header_metadata["Type_Name"] == "XMLOffset")
        offset = self.header_metadata[tf_mask]["Value"].values[0]
        if offset == 0:
            print("INFO: XML footer metadata is empty.", file=sys.stderr)
        else:
            self._fid.seek(offset)
            # All XML footer metadata is contained within one line.
            self.footer_metadata = objectify.fromstring(self._fid.read())
        return None

    def read_at(self, offset, size, ntype):
        """
        Seek to offset byte position then read size number of bytes in ntype format from file.
        """
        self._fid.seek(offset)
        return np.fromfile(self._fid, ntype, int(size))

    def get_frames(self, frame_list=None):
        """
        Yield a frame and per-frame metadata from the file.
        Return a frame and per-frame metadata from the file.
        Frame is returned as a numpy 2D array.
        Time stamp metadata is returned as Python datetime object.
        frame_list argument is python indexed: 0 is first frame.
        """
        # self.num_frames
        # self.last_frame_num
        pass
        
                
    def get_frame(self, frame_num=0):
        """
        Return a frame and per-frame metadata from the file.
        Frame is returned as a numpy 2D array.
        Time stamp metadata is returned as Python datetime object.
        frame_num argument is python indexed: 0 is first frame.
        """
        # See SPE 3.0 File Format Specification:
        # ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf

        # TODO: Create frame class. Object has own frame metadata.
        # TODO: Catch if using ROIs. Currently only supports one ROI.
        # TODO: separate into two internal functions
        # TODO: allow lists
        
        # If XML footer metadata exists (i.e. for final reductions).
        if hasattr(self, 'footer_metadata'):
            # TODO: complete as below
            pass
        
        # Else use binary header metadata (i.e. for online analysis).
        # else:
        # Get offset byte position of start of all data.
        tf_mask = (self.header_metadata["Type_Name"] == "lastvalue")
        start_offset = self.header_metadata[tf_mask]["Offset"].values[0] + 2
        # Get number of pixels per frame.
        tf_mask = (self.header_metadata["Type_Name"] == "xdim")
        xdim = self.header_metadata[tf_mask]["Value"].values[0]
        tf_mask = (self.header_metadata["Type_Name"] == "ydim")
        ydim = self.header_metadata[tf_mask]["Value"].values[0]
        pixels_per_frame = xdim * ydim
        # Get pixel datatype, bit-depth. Assume metadata datatype, bit-depth.
        # Datatypes 6, 2, 1, 5 are for only SPE 2.X, not SPE 3.0.
        # Assumed metadata datatype 64-bit signed integer
        # is from XML footer metadata using previous experiments with LightField.
        tf_mask = (self.header_metadata["Type_Name"] == "datatype")
        pixel_datatype = self.header_metadata[tf_mask]["Value"].values[0]
        ntypes_by_datatype = {6: np.uint8,
                              3: np.uint16,
                              2: np.int16,
                              8: np.uint32,
                              1: np.int32,
                              0: np.float32,
                              5: np.float64}
        pixel_ntype = ntypes_by_datatype[pixel_datatype]
        metadata_ntype = np.int64
        bitdepth_by_ntype = {np.int8: 8,
                             np.uint8: 8,
                             np.int16: 16,
                             np.uint16: 16,
                             np.int32: 32,
                             np.uint32: 32,
                             np.int64: 64,
                             np.uint64: 64,
                             np.float32: 32,
                             np.float64: 64}
        bits_per_pixel = bitdepth_by_ntype[pixel_ntype]
        bits_per_metadata = bitdepth_by_ntype[metadata_ntype]
        # Infer frame size, stride. Infer per-frame metadata size.
        # From SPE 3.0 File Format Specification, Ch 1 (with clarifications):
        # bytes_per_frame = pixels_per_frame * bits_per_pixel / (8 bits per byte)
        # bytes_per_metadata = 8 bytes per metadata
        #   metadata includes time stamps, frame tracking number, etc with 8 bytes each.
        # bytes_per_stride = bytes_per_frame + bytes_per_metadata
        num_metadata = 3
        bits_per_byte = 8
        bytes_per_frame = pixels_per_frame * (bits_per_pixel / bits_per_byte)
        bytes_per_metadata = bits_per_metadata / bits_per_byte
        bytes_per_stride = bytes_per_frame + (num_metadata * bytes_per_metadata)
        # Infer the number of frames that have been taken using the file size in bytes.
        # NumFrames from the binary header metadata is the 
        # number of frames typed into LightField that will potentially be taken,
        # not the number of frames that have already been taken and are in the file being read.
        # In case the file is currently being written to by LightField
        # when the file is being read by Python, count only an integer number of frames.
        # Allow negative indexes using mod.
        self._fid.seek(0, 2)
        eof_offset = self._fid.tell()
        num_frames = (eof_offset - start_offset) // bytes_per_stride
        frame_num = frame_num % num_frames
        # Infer frame offset. Infer per-frame metadata offsets.
        # Assuming metadata: time_stamp_exposure_started, time_stamp_exposure_ended, frame_tracking_number
        # TODO: need flags from user if per-frame meta data. print warning if not available.
        # TODO: make num_metadata an arg
        frame_offset = start_offset + (frame_num * bytes_per_stride)
        metadata_offset = frame_offset + bytes_per_frame
        # Read frame, metadata. Format metadata timestamps to be absolute time, UTC.
        # Time_stamps from the ProEM's internal timer-counter card are in 1E6 ticks per second.
        # Ticks per second from XML footer metadata using previous LightField experiments.
        # 0 ticks is when "Acquire" was first clicked on LightField.
        # Assume "Acquire" was clicked when the .SPE file was created.
        # File creation time is in seconds since epoch, Jan 1 1970 UTC.
        # Note: Only relevant for online analysis. Not accurate for reductions.
        # TODO: pop metadata off (default) input list to read.
        frame = self.read_at(frame_offset, pixels_per_frame, pixel_ntype)
        frame = frame.reshape((ydim, xdim))
        file_ctime = os.path.getctime(self._fname)
        ticks_per_second = 1000000
        metadata_tsexpstart_offset = metadata_offset
        metadata_tsexpend_offset = metadata_tsexpstart_offset + bytes_per_metadata
        metadata_ftracknum_offset = metadata_tsexpend_offset + bytes_per_metadata
        metadata = {}
        metadata_tsexpstart = self.read_at(metadata_tsexpstart_offset, 1, metadata_ntype)[0] / ticks_per_second
        metadata_tsexpend = self.read_at(metadata_tsexpend_offset, 1, metadata_ntype)[0] / ticks_per_second
        metadata_ftracknum = self.read_at(metadata_ftracknum_offset, 1, metadata_ntype)[0]
        metadata["time_stamp_exposure_started"] = datetime.utcfromtimestamp(file_ctime + metadata_tsexpstart)
        metadata["time_stamp_exposure_ended"] = datetime.utcfromtimestamp(file_ctime + metadata_tsexpend)
        metadata["frame_tracking_number"] = metadata_ftracknum
        return (frame, metadata)

    def close(self):
        """
        Close file.
        """
        self._fid.close()
        return None

def main(fname, frame_num):
    """
    Read a numbered frame from the SPE file.
    Show a plot and print the metadata.
    """
    fid = File(fname)
    (frame, metadata) = fid.get_frame(frame_num)
    fid.close()
    return (frame, metadata)
            
if __name__ == "__main__":
    
    # TODO: have defaults for metadata
    fname_default = "test_yes_footer.spe"
    frame_num_default = -1

    parser = argparse.ArgumentParser(description="Read a SPE file and return ndarray frame and dict metadata variables.")
    parser.add_argument("--fname",
                        default=fname_default,
                        help=("Path to SPE file. "
                              +"Default: {default}".format(default=fname_default)))
    parser.add_argument("--frame_num",
                        default=frame_num_default,
                        help=("Frame number to read in. First frame is 0. Last frame is -1. "
                              +"Default: {default}".format(default=frame_num_default)))
    args = parser.parse_args()
    print(args)

    (frame, metadata) = main(fname=args.fname, frame_num=args.frame_num)
