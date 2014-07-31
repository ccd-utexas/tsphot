# tsphot

Time series photometry using astropy.

Top level script: main.py

## Install requirements

This proto-package requires:
- Install Anaconda Python distribution.
- Install latest stable, tagged version of https://github.com/astropy/ccdproc.  As of 2014-07-30, v0.2. Note for future: When ccdproc is eventually merged with astropy, modify code to use astropy.ccdproc.
- Install latest stable, tagged version of https://github.com/astropy/photutils.  As of 2014-07-15, no tagged version exists. Use https://github.com/ccd-utexas/photutils instead. Note for future: When photutils is eventually merged with astropy, modify code to use astropy.photutils.

## Examples

To display help text:  
```
$ python main.py --help
[...displays help text...]
```

See the wiki for additional examples: https://github.com/ccd-utexas/tsphot/wiki

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
