#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from torq.lab.profiling.annotate import annotate_host_profile_from_files, logger

def main():
    parser = argparse.ArgumentParser(description="Annotate a Torq runtime profiling log.")
    parser.add_argument("debug_info_dir", type=str, help="Path to the debug-info directory produced with --torq-debug-info")
    parser.add_argument("profile_file", type=str, help="Path to the runtime CSV produced with --torq_profile_host")
    parser.add_argument("annotated_profile_file", type=str, help="Path to the output file (.csv, .xlsx, or .pb for a Perfetto trace)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--npu-frequency-mhz", type=int, help="Nominal NPU frequency in MHz (used to convert from cycles to seconds compile time profiling data)")
    parser.add_argument("--measured-frequency-mhz", type=int, help="Runtime NPU frequency in MHz (used to normalize runtime profiling data to the nominal chip frequency)")

    args = parser.parse_args()

    if args.debug:        
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel("DEBUG")

    annotate_host_profile_from_files(
                    args.debug_info_dir,
                    args.profile_file,
                    [args.annotated_profile_file],
                    target_frequency_mhz=args.npu_frequency_mhz,
                    measured_frequency_mhz=args.measured_frequency_mhz
                )

if __name__ == "__main__":
    main()
