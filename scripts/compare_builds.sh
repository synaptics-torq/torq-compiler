#!/bin/bash

build_and_test_branch() {
    local branch="$1"
    local output_dir="$2"
    shift 2

    echo -e "\033[1mTesting $branch\033[0m"

    # switch to the specified branch and build
    git checkout "$branch"
    ninja -C ../iree-build/ torq

    mkdir -p "$output_dir"

    # run the test provided on the command line
    ( "$@" --debug-ir "$output_dir" 2>&1 | tee "$output_dir/test.log" ) && true
}

# check we received two branch names
if [ "$#" -le 2 ]; then
    echo "Usage: $0 <good branch1> <bad branch2> [test command]"
    exit 1
fi

# stop on error
set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..
cd "$BASE_DIR"

# check that the work tree is clean
if ! git diff --quiet; then
    echo "Error: Working tree is not clean"
    exit 1
fi

GOOD_BRANCH="$1"
BAD_BRANCH="$2"

for branch in "$GOOD_BRANCH" "$BAD_BRANCH"; do
    if ! git rev-parse --verify "$branch" >/dev/null 2>&1; then
        echo "Error: Branch '$branch' does not exist."
        exit 1
    fi
done

shift 2

# create the comparison directory
rm -rf tmp/comparison
mkdir -p tmp/comparison

# find the current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# when finished the script switch back to CURRENT_BRANCH
trap 'git checkout "$CURRENT_BRANCH"' EXIT

build_and_test_branch "$GOOD_BRANCH" tmp/comparison/good "$@"
build_and_test_branch "$BAD_BRANCH" tmp/comparison/bad "$@"

