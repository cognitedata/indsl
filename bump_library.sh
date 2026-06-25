#!/usr/bin/env bash
set -euo pipefail

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: tracked files have uncommitted changes. Stash or commit before bumping." >&2
    exit 1
fi

git fetch origin main
git checkout main
git pull --ff-only

CZ_ARGS=(--yes --changelog)
if [ -n "${INPUT_PRERELEASE:-}" ]; then
    CZ_ARGS+=(--prerelease "$INPUT_PRERELEASE")
fi

git checkout -b bump-version

# Commitizen creates a local tag as part of the bump. Delete it immediately so
# the only publishable tag is created later from the merged main commit.
uv run cz bump "${CZ_ARGS[@]}"

TAG=$(git describe --tags --exact-match HEAD 2>/dev/null || true)
if [ -n "$TAG" ]; then
    git tag -d "$TAG" >/dev/null
else
    echo "Warning: no tag found at HEAD after cz bump — nothing to delete." >&2
fi

echo ""
echo "Prepared release $TAG on branch bump-version."
echo "The local tag was deleted intentionally."
echo "After the PR is merged, run: bash publish_library.sh"
