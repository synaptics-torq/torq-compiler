#!/usr/bin/env python3

import argparse
import logging
from torq.performance import annotate_host_profile_from_files, logger

def main():
    parser = argparse.ArgumentParser(description="Annotate a Torq runtime profiling log.")
    parser.add_argument("debug_info_dir", type=str, help="Path to the debug-info directory produced with --torq-debug-info")
    parser.add_argument("profile_file", type=str, help="Path to the runtime CSV produced with --torq_profile_host")
    parser.add_argument("annotated_profile_file", type=str, help="Path to the output file (.csv, .xlsx, or .pb for a Perfetto trace)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:        
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel("DEBUG")


    annotate_host_profile_from_files(
                    args.debug_info_dir,
                    args.profile_file,
                    [args.annotated_profile_file]
                )

if __name__ == "__main__":
    main()
