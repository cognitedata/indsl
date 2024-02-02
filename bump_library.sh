if [ $INPUT_PRERELEASE ]; then INPUT_PRERELEASE="--prerelease $INPUT_PRERELEASE"; else INPUT_PRERELEASE=''; fi

git checkout -b bump-version

# create tag, create commit updating CHANGELOG.md, save changelog changes to body.md
poetry run cz bump --yes --changelog-to-stdout --changelog $INPUT_PRERELEASE > body.md
