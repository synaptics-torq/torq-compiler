#!/usr/bin/env python3

import argparse
from torq.performance import annotate_nss_profile_from_files, annotate_host_profile_from_files

def main():
    parser = argparse.ArgumentParser(description="Parse an MLIR file.")
    parser.add_argument("mlir_file", type=str, help="Path to the executable-targets phase dump file")
    parser.add_argument("profile_file", type=str, help="Path to the runtime profile file.")
    parser.add_argument("annotated_profile_file", type=str, help="Path to the output")
    parser.add_argument("--host_profiling", action="store_true", default=False, help="Enable tracing of buffers in the torq runtime")

    args = parser.parse_args()
    if args.host_profiling:
        print("Host profiling enabled")
        annotate_host_profile_from_files(
                        args.mlir_file,
                        args.profile_file,
                        args.annotated_profile_file
                    )
    else:
        annotate_nss_profile_from_files(
                        args.mlir_file,
                        args.profile_file,
                        args.annotated_profile_file
                    )

if __name__ == "__main__":
    main()
