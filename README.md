# tsphot

Time series photometry using astropy.

Note: As of 2014-05, requires numpy 1.7.1 and old version of photutils. See https://github.com/ccd-utexas/photutils_20140522

Top level script: do_spe_online.py

## Examples

To display help text:  
```
$ python do_spe_online.py --help
[...displays help text...]
```

To do online reduction of an SPE file:
```
$ python do_spe_online.py --fpath /path/to/data.spe --sleep 5 --verbose
```
