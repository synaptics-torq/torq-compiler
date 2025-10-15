#!/usr/bin/env python3

import re
import subprocess
import sys
import argparse
from tempfile import NamedTemporaryFile

parser = argparse.ArgumentParser(description="Create a call graph for CSS tasks from an ELF file.")
parser.add_argument("elf_file", help="Path to the ELF file to analyze.")
parser.add_argument("output_file", help="Path to the output PNG file for the call graph.")

args = parser.parse_args()

lines = subprocess.check_output(["riscv64-unknown-elf-nm", "-S", "-a", args.elf_file]).decode().splitlines()

sizes = {}

sizes_match = re.compile(r'^[0-9a-f]+\s+([0-9a-f]+)\s+[tT]\s+(.*)')

for line in lines:
    m = sizes_match.match(line)

    if m:
        size = int(m.group(1), 16)
        name = m.group(2)
        sizes[name] = size

lines = subprocess.check_output(["riscv64-unknown-elf-objdump", "-d", args.elf_file]).decode().splitlines()

symbol_match = re.compile(r'^[0-9a-f]+ <(.*)>:')

reference_match = re.compile(r'^[0-9a-f]+:.*<([^+]*)(\+0x[a-z0-9]+)?>')

functions = {}

current_function = None
references = set()

for line in lines:

    m =  symbol_match.match(line)

    if m:

        if current_function:
            functions[current_function] = references

        current_function = m.group(1)

        references = set()

        continue

    m = reference_match.match(line)
    
    if m:                
        if current_function == m.group(1):
            continue
    
        references.add(m.group(1)) 


def get_dependencies(function_name):
    dependencies = set()

    to_scan = set(functions.get(function_name, set()))

    while len(to_scan) > 0:        
        ref = to_scan.pop()        
        if ref in dependencies:
            continue

        if ref == "l0":
            continue
            
        dependencies.add(ref)
        to_scan |= set(functions.get(ref, {}))

    return dependencies



def get_total_size(function_name):
    
    size = sizes.get(function_name, 0)

    for dep in get_dependencies(function_name):
        size += sizes.get(dep, 0)

    return size

with NamedTemporaryFile() as tmpfile:

    with open(tmpfile.name, "w") as fp:
        fp.write("digraph G {\n")
        fp.write("rankdir=LR;\n")
        fp.write("node [shape=box];\n")

        for function, refs in functions.items():       
            fp.write(f'"{function}" [label="{function}\\nTot: {get_total_size(function)}\\nSize: {sizes.get(function, 0)}"];\n') 
            for ref in refs:
                if ref in ["__dtcm_size", "__text_end", "__css_regs_start" ,"__itcm_start", "__itcm_size"]:
                    continue
                fp.write(f'"{function}" -> "{ref}";\n')

        fp.write("}\n")

    subprocess.check_call(["dot", "-Tpng", tmpfile.name, "-o", args.output_file])