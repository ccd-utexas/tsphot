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

## How to contribute

- Set up your GitHub account:
- - Create a github account.
- - Have an existing ccd-utexas member add you to the organization. See the stache entry on the ccd-utexas GitHub organization for details.
- - Install the github application: e.g. https://mac.github.com/
- - Sign in with your github credentials
- - Under "Repositories", open the "ccd-utexas" account.
- - By "ccd-utexas/tsphot" click "Clone to Computer"
- - Click the arrow to open the repository.

- Create and edit your branch:
- - Make the "develop" branch active by double clicking.
- - Click "+" to create your new branch off of "develop". Describe your miniproject in your branch name, then click "Branch".
- - Click "Publish" to let others group members know of your project.
- - Navigate to your local version of the cloned repository and make your code changes. To find the local cloned repository: Under "Repositories", right-click "ccd-utexas/tsphot", click "Show in Finder".
- - Commit your changes to your branch often in case you need to revert back. To commit changes: Click "Changes", type a summary and description of your changes, click "+" to enable syncing, click "Commit & Sync".

- Merge and delete your branch:
When you are finished editing your version of the code and ready to merge your changes back into "develop".
- - make "develop" your active branch.
- - Click "Sync Branch" in case someone else commited to "develop while you were editing your verison.
- - Merge your changes into develop: 
