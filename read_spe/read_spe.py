#!/usr/bin/env python
"""
Read .SPE file into numpy array.

Adapted from http://wiki.scipy.org/Cookbook/Reading_SPE_files
Offsets and names taken as from .SPE file specification, see
ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
SPE%203.0%20File%20Format%20Specification.pdf

Note: Use with SPE 3.0. Not backwards compatible with SPE 2.X.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
from lxml import objectify, etree

class File(object):
    
    def __init__(self, fname):
        """
        Open file and load metadata from header and footer.
        """
        # For online analysis, read metadata from binary header.
        # For final reductions, read more complete metadata from XML footer.
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
        ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf
        """
        # file_header_ver and xml_footer_offset are
        # the only required header fields for SPE 3.0.
        # Header information from SPE 3.0 File Specification, Appendix A.
        # Read in CSV of header format without comments.
        format_file = 'spe_30_header_format.csv'
        format_file_base, ext = os.path.splitext(format_file)
        format_file_nocmts = format_file_base + '_temp' + ext
        if not os.path.isfile(format_file):
            raise IOError("SPE 3.0 header format file does not exist: "+format_file)
        if not ext == '.csv':
            raise TypeError("SPE 3.0 header format file is not .csv: "+format_file)
        with open(format_file, 'r') as f_cmts:
            # Make a temporary file without comments.
            with open(format_file_nocmts, 'w') as f_nocmts:
                for line in f_cmts:
                    if line.startswith('#'):
                        continue
                    else:
                        f_nocmts.write(line)
        self.header_metadata = pd.read_csv(format_file_nocmts, sep=',')
        os.remove(format_file_nocmts)
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
            pos = self.header_metadata["Offset"][idx]
            try:
                size = (self.header_metadata["Offset"][idx+1]
                        - self.header_metadata["Offset"][idx]
                        - 1)
            # Key error if at last value in the header
            except KeyError:
                size = 1
            ntype = binary_ntypes[self.header_metadata["Binary"][idx]]
            values_by_offset[pos] = self.read_at(pos, size, ntype)
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
        pos = self.header_metadata[tf_mask]["Value"].values[0]
        if pos == 0:
            print("INFO: XML footer metadata is empty.", file=sys.stderr)
        else:
            self._fid.seek(pos)
            # All XML footer metadata is contained within one line.
            self.footer_metadata = objectify.fromstring(self._fid.read())
        return None

    def read_at(self, pos, size, ntype):
        """
        Seek to position then read size number of bytes in ntype format from file.
        """
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, int(size))
            
    def load_frame(self, frame_num=0):
        """
        Load a frame from the file.
        frame_num is python indexed: 0 is first frame.
        Uses header metadata.
        """
        # Allow negative indexes
        # TODO: allow lists
        tf_mask = (self.header_metadata["Type_Name"] == "NumFrames")
        numframes = self.header_metadata[tf_mask]["Value"].values[0]
        frame_num = frame_num % numframes
        # Get byte position of start of frame data
        tf_mask = (self.header_metadata["Type_Name"] == "lastvalue")
        start = self.header_metadata[tf_mask]["Offset"].values[0] + 2
        # Get size
        tf_mask = (self.header_metadata["Type_Name"] == "xdim")
        xdim = self.header_metadata[tf_mask]["Value"].values[0]
        tf_mask = (self.header_metadata["Type_Name"] == "ydim")
        ydim = self.header_metadata[tf_mask]["Value"].values[0]
        size = xdim * ydim
        # Compute read position
        # TODO: infer stride
        pos = start + (frame_num * size)
        print(pos)
        # Get datatype
        # datatypes 6, 2, 1, 5 are for only SPE 2.X, not SPE 3.0.
        # From SPE 3.0 File Format Specification, Chapter 1.
        tf_mask = (self.header_metadata["Type_Name"] == "datatype")
        datatype = self.header_metadata[tf_mask]["Value"].values[0]
        ntypes_by_datatype = {6: np.uint8,
                              3: np.uint16,
                              2: np.int16,
                              8: np.uint32,
                              1: np.int32,
                              0: np.float32,
                              5: np.float64}
        ntype = ntypes_by_datatype[datatype]
        # Read frame data.
        frame = self.read_at(pos, size, ntype)
        return frame.reshape((ydim, xdim))

    def close(self):
        """
        Close file.
        """
        self._fid.close()
        return None

# if __name__ == "__main__":
#     # TODO: use argparse
#     # TODO: check if ver 3.0, warn if not
#     import sys
#     img = load(sys.argv[-1])
