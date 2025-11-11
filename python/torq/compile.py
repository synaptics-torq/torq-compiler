import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from shutil import copy2

try:
    import iree.compiler.tools as iree_c
    import iree.compiler.tflite as iree_tflite_compile
    IREE_C_PYAPI = True
except ImportError:
    IREE_C_PYAPI = False

logger = logging.getLogger("Torq-compile")


def get_iree_version() -> str | None:
    import re

    try:
        out = subprocess.check_output(["iree-compile", "--version"], text=True)
    except FileNotFoundError:
        logger.warning("Failed to check iree-compile version; Ensure 'iree-compile' is installed and accessible from PATH")
        return None
    m = re.search(r"IREE compiler version ([\w.\-+]+)", out)
    if not m:
        return None
    ver_str = m.group(1)
    if ver_str.lower() == "unknown":
        return None
    return ver_str


def export_onnx_to_mlir(
    onnx_model: str | Path,
    mlir_model: str | Path,
    opset: int = 17,
    use_cli: bool = False
):
    if not Path(onnx_model).exists():
        raise FileNotFoundError(f"ONNX model '{onnx_model}' not found")

    if IREE_C_PYAPI and not use_cli:
        try:
            import sys
            subprocess.check_output(
                [
                    sys.executable, "-m", "iree.compiler.tools.import_onnx",
                    str(onnx_model),
                    "-o", str(mlir_model),
                    # TODO: currently unsupported, enable in future IREE versions
                    # "--opset-version", str(opset), 
                    "--data-prop",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None
    else:
        if not IREE_C_PYAPI:
            logger.warning("IREE compile python API not found, will attempt fallback to `iree-import-onnx` CLI")
        try:
            subprocess.check_output(
                [
                    "iree-import-onnx",
                    str(onnx_model),
                    "-o", str(mlir_model),
                    # TODO: currently unsupported, enable in future IREE versions
                    # "--opset-version", str(opset), 
                    "--data-prop",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "iree-import-onnx binary not found in PATH"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None


def export_tflite_to_mlir(
    tflite_model: str | Path,
    mlir_model: str | Path | None = None,
    use_cli: bool = False
):
    if not Path(tflite_model).exists():
        raise FileNotFoundError(f"TFLite model '{tflite_model}' not found")

    if IREE_C_PYAPI and not use_cli:
        iree_tflite_compile.compile_file(
            str(tflite_model),
            import_only=True,
            output_file=str(mlir_model)
        )
    else:
        if not IREE_C_PYAPI:
            logger.warning("IREE compile python API not found, will attempt fallback to `iree-import-tflite` CLI")
        try:
            subprocess.check_output(
                [
                    "iree-import-tflite",
                    str(tflite_model),
                    "-o", str(mlir_model)
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "iree-import-tflite binary not found in PATH"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export TFLite model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None


def compile_mlir_for_vm(
    mlir_model: str | Path,
    output_model: str | Path,
    target: str = "llvm-cpu",
    compiler_args: list[str] | None = None,
    cross_compile: bool = False,
    use_cli: bool = False
):
    compiler_args = compiler_args or []
    if target == "llvm-cpu":
        compiler_args += [
            "--iree-hal-target-backends=llvm-cpu"
        ]
        if cross_compile:
            compiler_args += [
                "--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu",
                "--iree-llvmcpu-target-cpu-features=+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse",
                "--iree-llvmcpu-target-cpu=generic"
            ]
        else:
            compiler_args += [
                "--iree-llvmcpu-target-cpu=host"
            ]

            from packaging import version
            if (iree_version := get_iree_version()) and version.parse(iree_version) >= version.parse("3.7.1"):
                compiler_args.append("--iree-hal-target-device=local")
    elif target == "torq":
        compiler_args += [
            "--iree-hal-target-backends=torq",
        ]
        if cross_compile:
            compiler_args += [
                "--torq-target-host-triple=aarch64-unknown-linux-gnu",
                "--torq-target-host-cpu=generic",
                "--torq-target-host-cpu-features=+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse"
            ]

    if IREE_C_PYAPI and not use_cli:
        compiled_bytes = iree_c.compile_file(
            str(mlir_model),
            target_backends=[target],
            extra_args=compiler_args,
        )
        with open(output_model, "wb") as f:
            f.write(compiled_bytes)
    else:
        if not IREE_C_PYAPI:
            logger.warning("IREE compile python API not found, will attempt fallback to `iree-compile` CLI")
        try:
            compile_cmd = [
                "iree-compile",
                str(mlir_model),
                "-o", str(output_model)
            ] + [str(arg) for arg in compiler_args]
            logger.debug("Full compile command: '%s'", " ".join(compile_cmd))
            subprocess.check_output(
                compile_cmd,
                text=True,
                stderr=subprocess.STDOUT
            )
        except FileNotFoundError:
            raise RuntimeError(
                "iree-compile binary not found in PATH"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to compile MLIR model with: '{' '.join(e.cmd)}:\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def export_iree(
    input_model: str | Path,
    output_dir: str | Path,
    compile_vmfb: bool = True,
    opset: int = 17,
    compiler_args: list[str] | None = None,
    cross_compile: bool = False,
    use_iree_cli: bool = False
):
    input_model = Path(input_model)
    output_dir = Path(output_dir)
    model_name = input_model.stem
    model_type = input_model.suffix
    if model_type not in (".onnx", ".tflite"):
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported extensions are: .onnx, .tflite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        mlir_model = Path(temp_dir) / f"{model_name}.mlir"
        if model_type == ".onnx":
            export_onnx_to_mlir(input_model, mlir_model, opset, use_iree_cli)
        else:
            export_tflite_to_mlir(input_model, mlir_model, use_iree_cli)
        copy2(mlir_model, output_dir / f"{model_name}.mlir")
        if compile_vmfb:
            vmfb_model = output_dir / f"{model_name}.vmfb"
            compile_mlir_for_vm(
                mlir_model,
                vmfb_model,
                compiler_args=compiler_args,
                cross_compile=cross_compile,
                use_cli=use_iree_cli
            )


def add_iree_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("IREE args")
    group.add_argument(
        "--opset",
        type=int,
        default=22,
        help="ONNX opset to use, older models will be updated to this opset (default: %(default)s)"
    )
    group.add_argument(
        "--ic-arg",
        type=str,
        action="append",
        default=[],
        metavar="ARG [=VALUE] | FILE",
        help="IREE compile arg, provide as `--ic-arg arg` or `--ic-arg arg=value` or a flagfile"
    )
    group.add_argument(
        "--cross-compile",
        action="store_true",
        default=False,
        help="Cross compile for aarch64"
    )
    group.add_argument(
        "--use-iree-cli",
        action="store_true",
        default=False,
        help="Enforce using the `iree-compile` binary instead of Python API"
    )


def process_iree_args(args: argparse.Namespace) -> list[str]:

    def _fmt_arg(arg: str) -> str | None:
        arg = arg.strip()
        if not arg:
            return None
        return arg if arg.startswith("--") else ("--" + arg)

    def _process_flagfile(arg_file: Path) -> list[str]:
        args = []
        raw_args = arg_file.read_text().splitlines()
        for raw_arg in raw_args:
            if raw_arg.startswith("#"):
                continue
            if (arg := _fmt_arg(raw_arg)):
                args.append(arg)
        return args

    iree_c_args: list[str] = []
    for raw_arg in args.ic_arg:
        if (arg_file := Path(raw_arg)).exists() and arg_file.is_file():
            iree_c_args.extend(_process_flagfile(arg_file))
        else:
            if (arg := _fmt_arg(raw_arg)):
                iree_c_args.append(arg)
    return iree_c_args


def main():
    import argparse
    import sys
    from .utils.logging import configure_logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=Path,
        metavar=".onnx | .tflite | .mlir | DIR",
        help="Path to MLIR or ONNX/TFLite model, or directory containing models"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        required=True,
        choices=["torq", "llvm-cpu"],
        help="Torq compile target (choices: %(choices)s)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="Output directory (default: <model_dir>/<model_name>)"
    )
    parser.add_argument(
        "-d", "--dump-debug",
        action="store_true",
        default=False,
        help="Dump debug symbols"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Root directory for dumping debug symbols (default: <output_dir>/debug)"
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    add_iree_args(parser)
    args = parser.parse_args()

    configure_logging(args.logging)

    if not args.model.exists():
        print(f"Invalid model file/directory '{args.model}'")
        sys.exit(1)
    if args.model.is_dir():
        model_files: list[Path] = [f for f in args.model.iterdir() if f.is_file() and f.suffix in (".onnx", ".mlir")]
        if not model_files:
            print(f"No models to compile in '{args.model}'")
            sys.exit(1)
    else:
        model_files: list[Path] = [args.model]

    success = 0
    failed = 0
    for model_file in model_files:
        output_dir: Path = args.output_dir or (Path(model_file.parent) / Path(model_file).stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory set to '%s'", str(output_dir))
        debug_dir = None
        if args.dump_debug:
            debug_dir: Path = debug_dir or (output_dir / model_file.stem / "debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Debug directory set to '%s'", str(debug_dir))
        
        iree_compile_args: list[str] = process_iree_args(args)
        if debug_dir:
            iree_compile_args += [
            f"--iree-hal-dump-executable-sources-to={debug_dir}/exec",
            f"--iree-hal-dump-executable-intermediates-to={debug_dir}/exec",
            f"--dump-compilation-phases-to={debug_dir}/compile"
        ]
        logger.debug("Added iree-compile debug args, current args: %s", str(iree_compile_args))

        output_model: Path = Path(output_dir / (model_file.stem + ".vmfb"))
        model_type: str = model_file.suffix.lower()
        if model_type != ".mlir":
            mlir_model: Path = Path(output_dir / (model_file.stem + ".mlir"))
            if model_type == ".onnx":
                logger.info("Exporting ONNX model '%s' to MLIR...", str(model_file))
                export_onnx_to_mlir(model_file, mlir_model, args.opset, args.use_iree_cli)
            elif model_type == ".tflite":
                logger.info("Exporting TFLite model '%s' to MLIR...", str(model_file))
                export_tflite_to_mlir(model_file, mlir_model, args.use_iree_cli)
            else:
                logger.error("Unsupported model type '%s'", model_type)
                failed += 1
                continue
            logger.info("Successfully exported '%s'", str(mlir_model))
        else:
            mlir_model: Path = model_file

        logger.info("Compiling MLIR model '%s' for %s...", str(mlir_model), args.target)
        try:
            compile_mlir_for_vm(
                mlir_model,
                output_model,
                args.target,
                iree_compile_args,
                args.cross_compile,
                args.use_iree_cli
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("Compilation failed for target %s: %s", args.target, str(e))
            if debug_dir:
                logger.info("Debug symbols dumped to '%s'", str(debug_dir))
            else:
                logger.info("Run with '-d' to dump debug symbols")
            failed += 1
        else:
            logger.info("Successfully compiled '%s'", str(output_model))
            success += 1

    print(f"Summary: successfully compiled {success} models, failed to compile {failed} models")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
