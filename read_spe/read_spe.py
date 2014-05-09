#!/usr/bin/env python
"""
Read .SPE file into numpy array.

Adapted from http://wiki.scipy.org/Cookbook/Reading_SPE_files
For .SPE file specification, see
ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
SPE%203.0%20File%20Format%20Specification.pdf
"""

import numpy as N


class File(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = N.int64(self.read_at(42, 1, N.uint16)[0])
        self._ydim = N.int64(self.read_at(656, 1, N.uint16)[0])

    def get_size(self):
        return (self._xdim, self._ydim)
        
    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return N.fromfile(self._fid, ntype, size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim, N.uint16)
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        self._fid.close()


def load(fname):
    fid = File(fname)
    img = fid.load_img()
    fid.close()
    return img


if __name__ == "__main__":
    import sys
    img = load(sys.argv[-1])

