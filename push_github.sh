#!/usr/bin/env bash
set -e

# Usage: ./push_github.sh https://github.com/USER/REPO.git

if [ -z "$1" ]; then
  echo "Donne l'URL du repo: ./push_github.sh https://github.com/USER/REPO.git"
  exit 1
fi

git init
git add .
git commit -m "Initial commit - Projet1 Streamlit CyberVision"
git branch -M main
git remote add origin "$1"
git push -u origin main
