#!/usr/bin/env bash
set -euo pipefail

CZ_ARGS=(--yes --changelog)
if [ -n "${INPUT_PRERELEASE:-}" ]; then
    CZ_ARGS+=(--prerelease "$INPUT_PRERELEASE")
fi

git checkout -b bump-version

# Commitizen creates a local tag as part of the bump. Delete it immediately so
# the only publishable tag is created later from the merged main commit.
uv run cz bump "${CZ_ARGS[@]}"

TAG=$(git describe --tags --exact-match HEAD 2>/dev/null)
git tag -d "$TAG" >/dev/null

echo ""
echo "Prepared release $TAG on branch bump-version."
echo "The local tag was deleted intentionally."
echo "After the PR is merged, run: bash publish_library.sh"
