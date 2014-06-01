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

This step-by-step tutorial walks you through how to contribute to this repository using the GitHub application.

Follow this model for contributing: http://nvie.com/posts/a-successful-git-branching-model/

For additional information on collaborating: https://help.github.com/categories/63/articles

**Warning:** Never delete "develop" or "master" branches.

- Set up your GitHub account:
  - Create a github account.
  - Have an existing ccd-utexas member add you to the organization. See the stache entry on the ccd-utexas GitHub organization for details.
  - Install the github application: e.g. https://mac.github.com/
  - Sign in with your github credentials
  - Under "Repositories", open the "ccd-utexas" account.
  - By "ccd-utexas/tsphot" click "Clone to Computer"
  - Click the arrow to open the repository.

- Create your branch from "develop" and edit:
- - Make the "develop" branch active by double clicking.
- - Click "+" to create your new branch off of "develop". Describe your miniproject in your branch name, then click "Branch".
- - Click "Publish" to let others group members know of your project.
- - Navigate to your local version of the cloned repository and make your code changes. To find the local cloned repository: Under "Repositories", right-click "ccd-utexas/tsphot", click "Show in Finder".
- - Commit your changes to your branch often in case you need to revert back. To commit changes: Click "Changes", type a summary and description of your changes, click "+" to enable syncing, click "Commit & Sync".

- Merge your branch back into "develop" and delete your branch:
Follow this section when you are finished editing your version of the code and are ready to merge your changes back into "develop".
- - Make "develop" your active branch.
- - Click "Sync Branch" in case someone else commited to "develop" while you were editing your verison.
- - Merge your changes into "develop": Click "Branches" > click "Merge View" > drag your branch into the left-hand box (the branch you're merging from), drag develop into the right-hand box (the branch you're merging into). The display will show "your_branch" => "develop" > click "Merge Branches". Make "develop" your active branch and resolve any conflicts in your text editer.

**Note:** Clicking "Merge Branches" does not sync the merged "develop" branch back to GitHub.

- - After resolving any code conflicts, make "develop" your active branch, click "Changes", click "Commit & Sync" (and/or "Sync" if there are unsynced commits).
- - All of your code changes in your branch have now been merged into "develop". Delete your branch now that your miniproject is complete. Make "develop" your active branch. Click the down-arrow on your branch, click "Delete", confirm that you want to delete your branch.

**Warning:** Never delete "develop" or "master" branches.

- Merge develop into master:
To issue a new code release into production. Use semantic versioning from http://semver.org/
- - Create a new release branch from develop, e.g. release_v1.0.0
- - Test the release branch.
- - Issue a pull request to merge from the release branch into master:
- - - Navigate to https://github.com/ccd-utexas/tsphot
- - - Make "master" the active branch.
- - - By "release_v1.0.0", click "Compare and pull request" > click "Edit".
- - - Set:
    base: master
    compare: release_v1.0.0
- - - This will merge release_v1.0.0 into master
- - - Click "Send pull request"
- - - An ccd-utexas owner will review the changes and click "Confirm".
- - Admin: 
- - - Once the pull request has been merged, delete the branch "release_v1.0.0".
- - - Tag the release with the version number at https://github.com/ccd-utexas/tsphot/releases

- Admin: Make a backup of tsphot:
- - Fork tsphot from your own GitHub account and rename to tsphot_BACKUP.
- - Update your backup by clicking "Compare, review, create a pull request" (circular branch symbol)
- - Click "Edit"
- - Set:
    base fork: ccd-utexas/tsphot base: master
    head fork: your_account/tsphot_BACKUP compare: master
- - This will merge your_account/tsphot_BACKUP:master into ccd-utexas/tsphot:master
- - Complete the pull request and merge.
