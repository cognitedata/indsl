#!/bin/bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Start from a clean build-docs branch, overwriting any local copy.
git branch -D build-docs 2>/dev/null || true
git checkout -b build-docs

# Regenerate the docs CHANGELOG from the repo CHANGELOG.
printf '# Changelog\n\n%s\n' "$(cat CHANGELOG.md)" > docs-source/source/CHANGELOG.md
git add docs-source/source/CHANGELOG.md
git commit -m "docs: update CHANGELOG.md"

# Build the HTML docs with all extras so gallery examples can import
# optional dependencies (csaps, scikit-image, etc.).
cd docs-source/
make clean
uv sync --group docs --all-extras
uv run make html
cd "$REPO_ROOT"

# Replace the published docs with the freshly built site so removed or
# renamed pages don't linger.
rm -rf docs/*
cp -r docs-source/build/html/. docs/
git add docs/
git commit -m "docs: update compiled doc files"

git push origin build-docs
