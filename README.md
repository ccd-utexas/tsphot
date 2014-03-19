tsphot
======

Time series photometry using astropy.

Right now, there are 3 files:

1. mark.py 
You use this to “mark” locations of the target and comparison stars. It creates the file phot_coords. 
Example:
	cat phot_coords
	294.022 400.938
	418.234   77.829
	260.217 378.926
These are just (x,y) pairs of the pixel positions of the stars on the first image.
Actually, it’s just as easy to create this file using a text editor and using ds9 to find the pixel positions of the stars

2. fits_process.py
This routine finds the darks and flats and creates master darks and flats. Then, it goes through each data frame sequentially, e.g., starting with A2392.0001.fits. First, it dark- and flat-corrects the frame, then it centers on the stars by looking near their previous positions (if it’s the first frame, it reads their positions from phot_coords). Next, using these positions, it performs aperture photometry on each star for a range of apertures; currently this range is 1 to 18 pixels in steps of 1 pixel. This step includes subtracting sky. It also does PSF fitting using a Gaussian to each star, and derives a separate estimate of the stars’ positions and counts. Thus after processing one file it produces aperture photometry for 18 apertures plus one PSF fit. 

It then does this for the rest of the frames in order. All the results are written to the file “lightcurve.app”. I decided that instead of creating 18 separate light curve files, one for each aperture, I would just put it all in one file (it’s all going to be read a computer, anyway). The aperture is the second column of this file, so if you want, e.g., the light curve for an aperture of 7.0 pixels, then just use the lines with a 7.0 in the second column. 

This script isn’t user friendly yet. For instance, you have to edit the code to specify which darks and flats to use. This is an easy change to make, but hardly critical until we’re sure the algorithms are working well in all cases. I’ll need to clean it up for the FRI kids, though.

3. lc_process.py
This is still under active development. Just last night I added to ability to deal with multiple comparison stars. What it currently does is take all possible combinations of comparison stars, compute the target divided by comparison light curve, and calculate the point to point scatter for this light curve. If there are two comparisons, then there are 3 ways of using them: use only the first, use only the second, or use both. Obviously, is you have 10 comparisons then the number of possibilities will be a lot higher, but that’s why we use computers!
