#!/bin/bash
git checkout -b build-docs
echo - "# Changelog\n$(cat CHANGELOG.md)" > docs-source/source/CHANGELOG.md
git commit docs-source/source/CHANGELOG.md -m "docs: update CHANGELOG.md"
cd docs-source/
make clean
poetry install --with docs
poetry run make html
cp -r ./build/html/* ../docs
git commit ../docs -m "docs: update compiled doc files"
git push origin build-docs
