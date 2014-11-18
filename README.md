# tsphot

Time series photometry using astropy.

Top level script: [main.py](main.py)

## Install requirements

This proto-package requires:
- Install the [Anaconda Python](http://continuum.io/downloads) distribution.
- Install the latest release of [ccdproc](https://github.com/astropy/ccdproc). As of 2014-09-14, v0.2.  
Note for future: When ccdproc is eventually merged with astropy, modify code to use astropy.ccdproc.  
Example on MacOS X to install and test:  
```bash
$ pip install ccdproc
$ ipython
In[1] import ccdproc
```
- Install latest stable, tagged version of [imageutils](https://github.com/astropy/imageutils) by running ```setup.py``` twice (some dependencies aren't built the first time). As of 2014-09-14, no tagged version exists. Install from https://github.com/ccd-utexas/imageutils instead.  
Note for future: When imageutils is eventually merged with astropy, modify code to use astropy.image.  
Example on MacOS X to install and test import:  
```bash
$ git clone https://github.com/ccd-utexas/imageutils
$ cd imageutils
$ python setup.py install
$ # again
$ python setup.py install
$ # move to where there is no 'imageutils' directory in your cwd
$ cd ../..
$ ipython
In[1] import imageutils
```
- Install latest stable, tagged version of [photutils](https://github.com/astropy/photutils).  As of 2014-09-14, no tagged version exists. Install from https://github.com/ccd-utexas/photutils instead.  
Note for future: When photutils is eventually merged with astropy, modify code to use astropy.photometry.  
Example on MacOS X to install and test import:
```bash
$ git clone https://github.com/ccd-utexas/photutils
$ cd photutils
$ python setup.py install
$ # again
$ python setup.py install
$ # move to where there is no 'photutils' directory in your cwd
$ cd ../..
$ ipython
In[1] import photutils
```
- Install latest stable release of [astroML](http://www.astroml.org/). As of 2014-11-18, v0.2.  
Example on MacOS X to install and test import:
```bash
$ pip install astroML
$ ipython
In[1] import astroML
```

## Examples

To display help text:  
```bash
$ python main.py --help
[...displays help text...]
```

See the wiki for additional examples: https://github.com/ccd-utexas/tsphot/wiki

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
