[Go to the main page](./README.md)

# Publishing a new version of InDSL
The workflow for publishing the `indsl` library consists of bumping the version (1) and building the documentation (2).
1. Bumping the version:
    1. Delete any local branch called **bump-version**: `git branch -d bump-version`
    2. Run [bump_library.sh](./bump_library.sh)

        This script will create a new commit with the `changelog` and a new `tag` following semantic versioning.
        If you want this to be a prerelease, set an environment variable INPUT_PRERELEASE to `alpha`, `beta` or `rc` on the parameter

    3. Modify the CHANGELOG.md if needed and commit the changes for `CHANGELOG.md`, `pyproject.toml`.
    (Make sure that indsl version is updated in `pyproject.toml`)
    4. Push the branch to github
    5. Push the new tag to github: `git push origin --tags`
    6. Create a PR and ask someone in the Cognite `Charts Backend` to approve and merge.
    7. After merge, create a release in github, selecting the new tag. This will trigger the publish [workflow](./publish.yaml). This workflow will build the package and publish it to the PyPI repository

2. Documentation is rebuilt and published automatically when a GitHub release is created (step 1.7 above). The [Build and deploy docs](./.github/workflows/docs.yaml) workflow rebuilds the Sphinx site and deploys it to https://indsl.docs.cognite.com/ — no manual action required.

3. After the release is done, you can check the new version in the [PyPI repository](https://pypi.org/project/indsl/)

4. For updating translations follow the instructions in [TRANSLATIONS.md](./TRANSLATIONS.md)
