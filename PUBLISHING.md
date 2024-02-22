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
    6. Create a PR and ask someone in the Cognite `Charts InDSL Review` to approve and merge.
    7. After merge, create a release in github, selecting the new tag. This will trigger the publish [workflow](./publish.yaml). This workflow will build the package and publish it to the PyPI repository

2. Build and update the documentation
    1. Delete any local branch called **build-docs**:
        `git branch -d build-docs`
    2. Run [build-docs.sh](./build_docs.sh). This will automatically create a new local branch called `build-docs`, commit the built documentation and push to github.
    3. Create a PR and ask someone in the Cognite `Charts InDSL Review` to approve and merge. When merged to main, github pages will redeploy the documentation page.

3. Update strings in Locize - Before updating docstrings in Locize check if any existing key should be updated. If yes, then make sure it is done according to the rules:
    1. If the string has been translated (available in more than one (EN) language), create a new key (only in EN) in the suggested format and assign the new text to it. In
       this case the current script should be adjusted.
    2. If the string is not translated (available only in EN), update the text in the existing key. In this case the current script should work fine.

4. Run GitHub Actions script to push merged docstring changes to Locize
    1. Go to the `Actions` tab in the InDSL repository on GitHub
    2. Select the `Push JSON to Locize` workflow under `All workflows` on the left hand side
    3. Run the workflow in the main branch by clicking the `Run workflow` dropdown on the right hand side and selecting the `Run workflow` option
