#!/usr/bin/env python3

import argparse
import logging
from torq.performance import annotate_host_profile_from_files, logger

def main():
    parser = argparse.ArgumentParser(description="Parse an MLIR file.")
    parser.add_argument("mlir_file", type=str, help="Path to the executable-targets phase dump file")
    parser.add_argument("profile_file", type=str, help="Path to the file produced with the --torq_profile_host option of torq-run-module ")
    parser.add_argument("annotated_profile_file", type=str, help="Path to the output (either .csv, .xlsx, or .pb for perfetto trace)")    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:        
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel("DEBUG")


    annotate_host_profile_from_files(
                    args.mlir_file,
                    args.profile_file,
                    [args.annotated_profile_file]
                )

if __name__ == "__main__":
    main()
