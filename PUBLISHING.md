[Go to the main page](./README.md)

# Publishing a new version of InDSL
The workflow for publishing the `indsl` library consists of merging a reviewed version bump PR, creating a protected release tag from `main`, and letting GitHub Actions publish the package and documentation.

1. Bump the version:
    1. Delete any local branch called **bump-version**: `git branch -d bump-version`
    2. Run [bump_library.sh](./bump_library.sh)

        This script creates a version bump commit and updates `CHANGELOG.md` following semantic versioning. Commitizen creates a local tag during the bump, but the script deletes that tag immediately so it cannot be pushed before the PR is merged.

        For a prerelease, set `INPUT_PRERELEASE` to `alpha`, `beta`, or `rc`:

        ```
        INPUT_PRERELEASE=rc bash bump_library.sh
        ```

    3. Review and adjust `CHANGELOG.md`, `pyproject.toml`, and `indsl/_version.py` if needed, then commit any edits.
    4. Push the branch to GitHub: `git push origin bump-version`
    5. Create a PR and ask someone in the Cognite `Charts Backend` to approve and merge.
    6. After merge, run [publish_library.sh](./publish_library.sh):

        ```
        bash publish_library.sh
        ```

        This script checks out the latest `main`, verifies the version files, creates an annotated tag on the merged `main` commit, pushes that specific tag, and creates the GitHub release from the committed changelog.
        Requires the `gh` CLI to be authenticated.

        > **Do not push tags manually.** A release tag must point to a commit that is already on `main`; otherwise it bypasses GitHub ref protection rules and triggers security alerts.

        Creating the GitHub release triggers the [publish workflow](./.github/workflows/publish.yaml), which validates the release tag, builds the package, and publishes it to PyPI using the existing `PYPI_API_TOKEN` secret.

2. Documentation is rebuilt and published automatically when a GitHub release is created. The [Build and deploy docs](./.github/workflows/docs.yaml) workflow rebuilds the Sphinx site and deploys it to https://indsl.docs.cognite.com/ — no manual action required.

3. After the release is done, you can check the new version in the [PyPI repository](https://pypi.org/project/indsl/)

4. For updating translations follow the instructions in [TRANSLATIONS.md](./TRANSLATIONS.md)
