#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/.git" ]]; then
  echo "Error: project is not inside a Git repository."
  exit 1
fi

REL_PROJECT_PATH="${PROJECT_DIR#${REPO_ROOT}/}"
if [[ "${REL_PROJECT_PATH}" == "${PROJECT_DIR}" ]]; then
  REL_PROJECT_PATH="."
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

echo "[sync] Fetching origin/${BRANCH} ..."
git fetch origin "${BRANCH}" --quiet || true

if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  read -r BEHIND AHEAD < <(git rev-list --left-right --count "origin/${BRANCH}...${BRANCH}")
  if [[ "${BEHIND}" -gt 0 ]]; then
    echo "Error: local branch is behind origin/${BRANCH} by ${BEHIND} commit(s)."
    echo "Please pull/rebase first, then re-run this script."
    exit 1
  fi
fi

echo "[stage] Staging project path: ${REL_PROJECT_PATH}"
git add -A -- "${REL_PROJECT_PATH}"

if git diff --cached --quiet; then
  echo "[commit] No staged changes under ${REL_PROJECT_PATH}."
  echo "[push] Ensuring upstream is set for ${BRANCH} ..."
  git push -u origin "${BRANCH}"
  echo "Done."
  exit 0
fi

MESSAGE="${*:-feat: update DG-GAA LLM homepage}"
echo "[commit] Creating commit: ${MESSAGE}"
git commit -m "${MESSAGE}"

echo "[push] Pushing to origin/${BRANCH} ..."
git push -u origin "${BRANCH}"

echo "Done."
