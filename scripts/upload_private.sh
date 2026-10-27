#!/usr/bin/env bash
set -euo pipefail
# Run from the delivered repository after installing GitHub CLI and signing in.
# No deletion, visibility change, or force push is performed.
cd "$(dirname "$0")/.."
repo="${1:-Keke946/LLM4Rec}"
gh auth status
if git remote get-url origin >/dev/null 2>&1; then
  echo 'An origin remote already exists. Review it before pushing; no changes made.' >&2
  exit 1
fi
if gh repo view "$repo" >/dev/null 2>&1; then
  echo 'Repository already exists. Review its contents and permissions; no changes made.' >&2
  exit 1
fi
# The API creation fails safely if an inaccessible repository has this name.
gh repo create "$repo" --private --source . --remote origin --push
gh repo view "$repo" --json url,isPrivate,defaultBranchRef
gh run list --repo "$repo" --limit 3
