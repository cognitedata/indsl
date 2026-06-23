#!/usr/bin/env bash
set -euo pipefail

# Creates the release tag on main and creates the GitHub release after the version bump PR is merged.
# Run this from the repo root after the bump-version PR has been merged into main.
#
# Usage: bash publish_library.sh
# Requires: gh CLI authenticated

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: gh CLI is required."
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "Error: gh CLI is not authenticated."
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: tracked files have uncommitted changes."
    exit 1
fi

git fetch origin main --tags
git checkout main
git pull --ff-only

LOCAL_MAIN=$(git rev-parse HEAD)
REMOTE_MAIN=$(git rev-parse origin/main)
if [ "$LOCAL_MAIN" != "$REMOTE_MAIN" ]; then
    echo "Error: local main does not match origin/main."
    exit 1
fi

VERSION=$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
TAG="v$VERSION"
TITLE="InDSL $VERSION"

PACKAGE_VERSION=$(python -c 'from pathlib import Path; ns = {}; exec(Path("indsl/_version.py").read_text(), ns); print(ns["__version__"])')
if [ "$PACKAGE_VERSION" != "$VERSION" ]; then
    echo "Error: pyproject.toml version ($VERSION) does not match indsl/_version.py ($PACKAGE_VERSION)."
    exit 1
fi

REMOTE_TAG_COMMIT=$(git ls-remote origin "refs/tags/$TAG^{}" | awk '{ print $1 }')
if [ -z "$REMOTE_TAG_COMMIT" ]; then
    REMOTE_TAG_COMMIT=$(git ls-remote origin "refs/tags/$TAG" | awk '{ print $1 }')
fi

PUSH_TAG=true
if [ -n "$REMOTE_TAG_COMMIT" ]; then
    if [ "$REMOTE_TAG_COMMIT" != "$LOCAL_MAIN" ]; then
        echo "Error: remote tag $TAG already exists but does not point to origin/main."
        exit 1
    fi
    PUSH_TAG=false
fi

NOTES_FILE=$(mktemp)
cleanup() {
    rm -f "$NOTES_FILE"
}
trap cleanup EXIT

awk -v tag="$TAG" '
    function is_target_header(line) {
        return line == "## " tag || index(line, "## " tag " ") == 1 || index(line, "## [" tag "]") == 1
    }
    /^## / && in_section && !is_target_header($0) { exit }
    is_target_header($0) { in_section = 1 }
    in_section { print }
' CHANGELOG.md > "$NOTES_FILE"

if [ ! -s "$NOTES_FILE" ]; then
    echo "Error: could not find release notes for $TAG in CHANGELOG.md."
    exit 1
fi

if gh release view "$TAG" >/dev/null 2>&1; then
    echo "Error: GitHub release $TAG already exists."
    exit 1
fi

if [ "$PUSH_TAG" = true ]; then
    git tag --annotate --force "$TAG" --message "$TITLE" HEAD

    echo "Pushing tag $TAG to origin at $(git rev-parse --short HEAD)..."
    git push origin "refs/tags/$TAG:refs/tags/$TAG"
else
    echo "Remote tag $TAG already points to origin/main; creating the missing release."
fi

RELEASE_ARGS=(--title "$TITLE" --notes-file "$NOTES_FILE" --verify-tag)
if [[ "$VERSION" =~ (a|b|rc)[0-9]+ ]]; then
    RELEASE_ARGS+=(--prerelease)
fi

echo "Creating GitHub release $TAG..."
gh release create "$TAG" "${RELEASE_ARGS[@]}"

echo ""
echo "Done. Release $TAG published; PyPI publish workflow triggered."
