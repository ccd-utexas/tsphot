# tsphot

Time series photometry using astropy.

Top level script: [main.py](main.py)

## Install requirements

This proto-package requires:
- Install the [Anaconda Python](http://continuum.io/downloads) distribution.
- Install the latest release of [ccdproc](https://github.com/astropy/ccdproc).  As of 2014-09-14, v0.2. Example on MacOS X: ```$ pip install ccdproc``` Note for future: When ccdproc is eventually merged with astropy, modify code to use astropy.ccdproc.
- Install latest stable, tagged version of [imageutils](https://github.com/astropy/imageutils).  As of 2014-09-14, no tagged version exists. Install from https://github.com/ccd-utexas/imageutils instead. Note for future: When imageutils is eventually merged with astropy, modify code to use astropy.image. Example on MacOS X: ```$ python setup.py install```
- Install latest stable, tagged version of [photutils](https://github.com/astropy/photutils).  As of 2014-09-14, no tagged version exists. Install from https://github.com/ccd-utexas/photutils instead. Note for future: When photutils is eventually merged with astropy, modify code to use astropy.photometry. Example on MacOS X: ```$ python setup.py install```

## Examples

To display help text:  
```bash
=======
**Note:** As of 2014-06-01, photutils must be installed from https://github.com/ccd-utexas/photutils

## Examples

To display help text: 
```
$ python main.py --help
[...displays help text...]
```

See the wiki for additional examples: https://github.com/ccd-utexas/tsphot/wiki
=======
See the wiki for this page: https://github.com/ccd-utexas/tsphot/wiki

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
