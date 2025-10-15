#!/usr/bin/env python3

# how to use on an existing branch (pre-rename)
#
#   git checkout ${existing branch}
#   git rebase ${commit where this script was merged}
#   ./scripts/rename.sh
#   git reset origin/main
#   git checkout HEAD -- third_party
#
# git status should now show only the changes that were done originally but with the new names

import os
import re
import shutil
import subprocess

SUBS = [
    ("iree-synaptics-synpu", "iree-synaptics-XXXX"),
    (".amazonaws.com/synaptics-synpu", ".amazonaws.com/synaptics-XXXX"),
    ("@syna-astra-dev/synpu-iree-plugin-maintainers", "@syna-astra-dev/XXXX-iree-plugin-maintainers"),
    ("zSvcsynpu", "zSvcXXXX"),
    ("synpu", "torq"),
    ("SYNPU", "TORQ"),
    ("SyNPU", "Torq"),
    ("Synpu", "Torq"),
    ("SynPU", "Torq"),
    ("zSvcXXXX", "zSvcsynpu"),
    ("iree-synaptics-XXXX", "iree-synaptics-synpu"),
    (".amazonaws.com/synaptics-XXXX", ".amazonaws.com/synaptics-synpu"),
    ("@syna-astra-dev/XXXX-iree-plugin-maintainers", "@syna-astra-dev/synpu-iree-plugin-maintainers")
]

sed = 'gsed' if shutil.which('gsed') is not None else "sed"

def rename(root, file):        

    for key, value in SUBS:
        if key in file:
            new_file = file.replace(key, value)
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            return new_file
    
    if "synpu" in file.lower():
        raise Exception("Unsupported name " + file)
    
    return file


def replace_text(path):

    # do not replace in this file
    if os.path.realpath(path) == os.path.realpath(__file__):
        return

    # do not modify compiled binaries
    if path.endswith(".a") or path.endswith(".so") or path.endswith(".dylib"):
        return

    for key, value in SUBS:
        subprocess.check_call([sed, '-i', f's|{key}|{value}|g', path])


def rename_files(base_dir):

    excludes = re.compile(r"^\.(/.git|/tmp|/.pytest_cache|/.vscode|/third_party/iree|/doc/_build|.*/__pycache__)(/|$)")    

    # we need to rename from bottom up because we may rename directories
    for root, dirs, files in os.walk(base_dir, topdown=False):
        
        if excludes.match(root):
            continue

        for file in files:
            print("f ", os.path.join(root, file))
            new_file = rename(root, file)            
            replace_text(os.path.join(root, new_file))

        for dir in dirs:            
            if excludes.match(os.path.join(root, dir)):
                continue
            print("d ", os.path.join(root, dir))
            rename(root, dir)


def main():
    rename_files(".")
    format_code = 'scripts/format-code'
    if os.path.exists(format_code):
        subprocess.check_call([format_code])

if __name__ == "__main__":
    main()
