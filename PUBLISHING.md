[Go to the main page](./README.md)

# Publishing a new version of InDSL
Publishing the `indsl` library consists of merging a reviewed version bump PR into `main`. GitHub Actions then publishes the package, creates the release tag and GitHub release, and deploys the documentation.

1. Bump the version:
    1. Delete any local branch called **bump-version**: `git branch -d bump-version`
    2. Run [bump_library.sh](./bump_library.sh)

        This script creates a version bump commit and updates `CHANGELOG.md` following semantic versioning. Commitizen creates a local tag during the bump, but the script deletes that tag immediately because the publish workflow creates the release tag.

        For a prerelease, set `INPUT_PRERELEASE` to `alpha`, `beta`, or `rc`:

        ```
        INPUT_PRERELEASE=rc bash bump_library.sh
        ```

    3. Review and adjust `CHANGELOG.md`, `pyproject.toml`, and `indsl/_version.py` if needed, then commit any edits.
    4. Push the branch to GitHub: `git push origin bump-version`
    5. Create a PR and ask someone in the Cognite `Horizontal Apps` to approve and merge. Approving this PR approves the release.

2. Merging the PR runs the [publish workflow](./.github/workflows/publish.yaml) on `main`. It runs on every push to `main`, but only releases when the version in `pyproject.toml` is not on PyPI yet. When it releases, it:
    1. Builds the package and publishes it to PyPI through the `CD` environment.
    2. Creates the `vX.Y.Z` tag on the published commit and a GitHub release with the notes from `CHANGELOG.md`.
    3. Deploys the documentation to https://indsl.docs.cognite.com/ using the [Build and deploy docs](./.github/workflows/docs.yaml) workflow.

    Check that the run passed in the [Actions tab](https://github.com/cognitedata/indsl/actions/workflows/publish.yaml). A failed run can be re-run; it skips anything that was already published.

    > **Do not push release tags manually.** Release tags are created by the publish workflow only.

3. After the release is done, you can check the new version in the [PyPI repository](https://pypi.org/project/indsl/)

4. For updating translations follow the instructions in [TRANSLATIONS.md](./TRANSLATIONS.md)
