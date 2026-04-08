#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/.git" ]]; then
  echo "Error: ${PROJECT_ROOT} is not inside a Git repository."
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
  TEMP_STASH_CREATED=0
  TEMP_STASH_NAME="commit.sh-autostash-$(date '+%Y%m%d_%H%M%S')"

  if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "[sync] Local changes detected. Creating temporary stash (including untracked files) ..."
    git stash push -u -m "${TEMP_STASH_NAME}" >/dev/null
    TEMP_STASH_CREATED=1
  fi

  git pull --rebase origin "${BRANCH}"

  if [[ "${TEMP_STASH_CREATED}" -eq 1 ]]; then
    echo "[sync] Restoring temporary stash ..."
    if ! git stash pop --index; then
      echo "Error: stash restore produced conflicts. Resolve conflicts, then run: git stash list"
      echo "If needed, manually re-apply with: git stash apply \"stash^{/${TEMP_STASH_NAME}}\""
      exit 1
    fi
  fi
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
