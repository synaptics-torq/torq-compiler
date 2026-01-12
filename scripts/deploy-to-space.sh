#!/bin/bash

if [[ -z "$1" ]] ; then
  echo "Usage: $0 <space-name>"
  exit 1
fi

read -p "Warning: This will destroy the history of the space. Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] ; then
    exit 1
fi

if [[ -d .git ]] ; then
    read -p "Warning: .git directory already exists. Delete it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] ; then
        exit 1
    fi
    rm -rf .git
fi

set -x

git init .
git config user.email "torq-ci@synaptics.com"
git config user.name "Torq CI"
git add .
git commit -m "Snapshot"
git push -f git@hf.co:spaces/$1 HEAD:main
