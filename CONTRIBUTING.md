# How to contribute

## Non-members of ccd-utexas

If you **are not** part of the ccd-utexas organization:

- See https://guides.github.com/ for how to contribute using GitHub.
- Use the fork-pull collaboration model.
- On your local fork, branch from and merge back into "develop", following this model: http://nvie.com/posts/a-successful-git-branching-model/

## Members of ccd-utexas

If you **are** part of the ccd-utexas organization (i.e. you're at UT Austin under Don Winget):

**Warning:** Never delete "develop" or "master" branches.

- See https://guides.github.com/ for how to contribute using GitHub.
- Have an existing ccd-utexas member add your GitHub account to the organization. See the stache entry on the ccd-utexas GitHub organization for details.
- Follow this model for contributing: http://nvie.com/posts/a-successful-git-branching-model/
- Release and tag stable versions from the "master" branch using semantic versioning: http://semver.org/
- For admin: Backup the repo by forking to your own GitHub account.
  - Rename the forked repo with the suffix "_BACKUP_YYYYMMDD".
  - It's easiest to update backups by simply creating new backups with different datestamps and then deleting outdated backup repos.
