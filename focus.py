#!/usr/bin/env python
"""
Analyze a frame for focusing.
"""

from __future__ import print_function

def find_stars(ndarr):
    """
    Find stars in a 2D ndarray.
    Arguments:
      ndarr = Image data as 2D numpy array.
    Returns:
      coords = List of x-y coordinate tuples [(x1, y1), (x2, y2), ...]
               Tuples are sorted by brightness.
    """
    # use sci-kit image
    # return whatever sci-kit image gives.
    pass

def compute_fwhm(ndarr, coord_list):
    """
    Compute full-width at half maximum for stars in a 2D ndarray.
    Arguments:
      ndarr = Image data as 2D numpy array.
      coords = List of x-y coordinate tuples [(x1, y1), (x2, y2), ...]
    Returns:
      fwhms = List of FHWM measurements for each star in order.
    """
    # Note: Objects in Python are passed by reference, not by making a copy.
    # Passing ndarr does not affect performance.
    pass

def main():
    """
    Only import this module.
    """
    pass

if __name__ == '__main__':
    main()
