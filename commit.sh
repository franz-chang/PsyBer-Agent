#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Error: current directory is not inside a Git repository."
  exit 1
fi
cd "${REPO_ROOT}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == "HEAD" ]]; then
  echo "Error: detached HEAD detected. Please checkout a branch first."
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Error: remote 'origin' is not configured."
  exit 1
fi

if git ls-remote --exit-code --heads origin "${BRANCH}" >/dev/null 2>&1; then
  echo "[sync] Pulling latest changes from origin/${BRANCH} ..."
  git pull --rebase origin "${BRANCH}"
else
  echo "[sync] origin/${BRANCH} does not exist yet. Skipping pull."
fi

echo "[stage] Staging all changes ..."
git add -A

if git diff --cached --quiet; then
  echo "[commit] No changes to commit."
  echo "[push] Ensuring branch is pushed to origin/${BRANCH} ..."
  git push -u origin "${BRANCH}"
  echo "Done."
  exit 0
fi

MESSAGE="${*:-chore: auto commit $(date '+%Y-%m-%d %H:%M:%S')}"
echo "[commit] Creating commit: ${MESSAGE}"
git commit -m "${MESSAGE}"

echo "[push] Pushing to origin/${BRANCH} ..."
git push -u origin "${BRANCH}"

echo "Done."
