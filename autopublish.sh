#!/bin/bash
# Auto build dashboard and push to GitHub when thesis files change
cd "$(dirname "$0")"

# Rebuild dashboard
python3 build.py || exit 1

# Check if anything changed
if git diff --quiet && git diff --cached --quiet; then
  exit 0
fi

# Commit and push
git add thesis/ dashboard.html template.html build.py .gitignore .github/
git -c user.name=foldnote -c user.email=foldnote@users.noreply.github.com commit -m "auto: update dashboard $(date +%Y-%m-%d\ %H:%M)"
git push origin main
