#!/usr/bin/env python
"""
Read .SPE file into numpy array.

Adapted from http://wiki.scipy.org/Cookbook/Reading_SPE_files
Offsets and names taken as from .SPE file specification, see
ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
SPE%203.0%20File%20Format%20Specification.pdf

Note: Use with SPE 3.0. Not backwards compatible with SPE 2.X.
"""

import numpy as np
import lxml

class File(object):
    
    def __init__(self, fname):
        """
        Open file and load metadata from header and footer.
        """
        # For online analysis, read metadata from binary header.
        # For final reductions, read more complete metadata from XML footer.
        self._fid = open(fname, 'rb')
        self._load_header_metadata(self)
        self._load_footer_metadata(self)
        return None

    def _load_header_metadata(self):
        """
        Load SPE metadata from binary header as a dict.
        Use metadata from header for online analysis
        since XML footer does not yet exist while taking data.
        """
        # file_header_ver and xml_footer_offset are
        # the only required header fields for SPE 3.0.
        # Header information from SPE 3.0 File Specification, Appendix A.
        # TODO: read data into pandas df
        # read in csv from appendix a
        # 
        
        pass

    def get_header_metadata(self):
        """
        Return SPE metadata from binary header as a dict.
        Use metadata from header for online analysis
        since XML footer does not yet exist while taking data.
        """
        # file_header_ver and xml_footer_offset are
        # the only required header fields for SPE 3.0.
        # TODO: read data into dict
        pass

    def _load_footer_metadata(self):
        """
        Load SPE metadata from XML footer as an lxml object.
        Use metadata from footer for final reductions
        since XML footer is more complete.
        """
        # TODO: read in as object

    def get_footer_metadata(self):
        """
        Return SPE metadata from XML footer as an lxml object.
        Use metadata from footer for final reductions
        since XML footer is more complete.
        """
        # TODO: return object

    # def _load_file_header_ver(self):
    #     """
    #     Load SPE version.
    #     """
    #     self._file_header_ver = self.read_at(1992, 1, np.float32)[0]
    #     return None
    
    # def _load_xml_footer_offset(self):
    #     """
    #     Load offset to the XML footer in bytes.
    #     """
    #     self._xml_footer_offset = self.read_at(678, 1, np.uint64)[0]
    #     return None
    
    # def _load_datatype(self):
    #     """
    #     Load binary type of pixel.
    #     """
    #     # datatypes 6, 2, 1, 5 are for only SPE 2.X, not SPE 3.0.
    #     datatypes = {6: np.uint8,
    #                  3: np.uint16,
    #                  2: np.int16,
    #                  8: np.uint32,
    #                  1: np.int32,
    #                  0: np.float32,
    #                  5: np.float64}
    #     key = self.read_at(108, 1, np.int16)[0]
    #     self._datatype = datatypes[key]
    #     return None
    
    # def _load_xdim(self):
    #     """
    #     Load width of a frame in pixels.
    #     """
    #     self._xdim = np.int64(self.read_at(42, 1, np.uint16)[0])
    #     return None
    
    # def _load_ydim(self):
    #     """
    #     Load height of a frame in pixels.
    #     """
    #     self._ydim = np.int64(self.read_at(656, 1, np.uint16)[0])
    #     return None
    
    # def _load_NumFrames(self):
    #     """
    #     Load number of frames.
    #     """
    #     self._NumFrames = np.int64(self.read_at(1446, 1, np.int32)[0])
    #     return None
    
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
    
    def load_metadata(self):
        """
        Load the XML metadata footer.
        """
        # TODO: use xml parser
        self._fid.seek(self._xml_footer_offset)
        # All metadata contained in one line
        return ET.fromstring(self._fid.read())

    # def tag_uri_and_name(elt):
    #     """
    #     Extract a URI XML namespace for appending to element tags.
    #     """
    #     # from http://stackoverflow.com/questions/1953761/accessing-xmlns-attribute-with-python-elementree
    #     if elt.tag[0] == "{":
    #         uri, ignore, tag = elt.tag[1:].partition("}")
    #     else:
    #         uri = None
    #         tag = elt.tag
    #     return uri, tag

    # def uri_tag(uri, tag):
    #     """
    #     Append element tag with XML namespace URI. 
    #     """
    #     return '{'+uri+'}'+tag

    # def _load_attribs(self):
    #     """
    #     Load file attributes.
    #     """
    #     (uri, ns) = tag_uri_and_name(self._metadata)
    #     dataformat_tag = uri_tag(uri, 'DataFormat')
    #     datablock_tag = uri_tag(uri, 'DataBlock')
    #     for elt in metadata.iter():
    #     if elt.tag == datablock_tag and elt.attrib['type'] == 'Frame':
    #         frame_attrib = elt.attrib
    
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
    # TODO: check if ver 3.0, warn if not
    import sys
    img = load(sys.argv[-1])
