# tsphot

Time series photometry using astropy.

Top level script: [main.py](main.py)

## Install requirements

This proto-package requires:
- Install the [Anaconda Python](http://continuum.io/downloads) distribution.
- Install the latest release of [ccdproc](https://github.com/astropy/ccdproc).  As of 2014-07-30, v0.2. Example on MacOS X: ```$ CC=clang pip install ccdproc``` Note for future: When ccdproc is eventually merged with astropy, modify code to use astropy.ccdproc.
- Install latest stable, tagged version of [photutils](https://github.com/astropy/photutils).  As of 2014-08-08, no tagged version exists. Install from https://github.com/ccd-utexas/photutils instead. Note for future: When photutils is eventually merged with astropy, modify code to use astropy.photometry.
- Install latest stable, tagged version of [imageutils](https://github.com/astropy/imageutils).  As of 2014-08-07, no tagged version exists. Install from https://github.com/ccd-utexas/imageutils instead. Note for future: When imageutils is eventually merged with astropy, modify code to use astropy.imageutils. Example on MacOS X: ```$ CC=clang python setup.py install```

## Examples

To display help text:  
```bash
$ python main.py --help
[...displays help text...]
```

See the wiki for additional examples: https://github.com/ccd-utexas/tsphot/wiki

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
