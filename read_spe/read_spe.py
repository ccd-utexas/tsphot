#!/usr/bin/env python
"""
Read .SPE file into numpy array.

Adapted from http://wiki.scipy.org/Cookbook/Reading_SPE_files
For .SPE file specification, see
ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
SPE%203.0%20File%20Format%20Specification.pdf
"""

import numpy as np

class File(object):

    def __init__(self, fname):
        """
        Open file and load metadata.
        """
        self._fid = open(fname, 'rb')
        self._load_file_header_ver()
        self._load_xml_footer_offset()
        self._load_datatype()
        self._load_xdim()
        self._load_ydim()
        self._load_size()
        self._load_NumFrames()
        return None

    def _load_file_header_ver(self):
        """
        Load SPE version.
        """
        self._file_header_ver = self.read_at(1992, 1, np.float32)[0]
        return None
    
    def get_file_header_ver(self):
        """
        Get SPE version.
        """
        return self._file_header_ver
    
    def _load_xml_footer_offset(self):
        """
        Load offset to the XML footer in bytes.
        """
        self._xml_footer_offset = self.read_at(678, 1, np.uint64)[0]
        return None

    def get_xml_footer_offset(self):
        """
        Get offset to the XML footer in bytes.
        """
        return self._xml_footer_offset
    
    def _load_datatype(self):
        """
        Load binary type of pixel.
        """
        datatypes = {6: np.uint8,
                     3: np.uint16,
                     2: np.int16,
                     8: np.uint32,
                     1: np.int32,
                     0: np.float32,
                     5: np.float64}
        key = self.read_at(108, 1, np.int16)[0]
        self._datatype = datatypes[key]
        return None
        
    def get_datatype(self):
        """
        Get binary type of pixel.
        """
        return self._datatype

    def _load_xdim(self):
        """
        Load width of a frame in pixels.
        """
        self._xdim = np.int64(self.read_at(42, 1, np.uint16)[0])
        return None

    def get_xdim(self):
        """
        Get width of a frame in pixels.
        """
        return self._xdim

    def _load_ydim(self):
        """
        Load height of a frame in pixels.
        """
        self._ydim = np.int64(self.read_at(656, 1, np.uint16)[0])
        return None
    
    def get_ydim(self):
        """
        Get height of a frame in pixels.
        """
        return self._ydim
        
    def read_at(self, pos, size, ntype):
        """
        Seek to position then read from file.
        """
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self):
        """
        Load the first image in the file.
        """
        img = self.read_at(4100, self._xdim * self._ydim, np.uint16)
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        """
        Close file.
        """
        self._fid.close()
        return None

def load(fname):
    fid = File(fname)
    img = fid.load_img()
    fid.close()
    return img

if __name__ == "__main__":
    # TODO: use argparse
    import sys
    img = load(sys.argv[-1])
