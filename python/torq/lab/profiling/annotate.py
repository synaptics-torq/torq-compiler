# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host-profile annotation and Perfetto-trace helpers for torq.lab.

The heavy dependencies (pandas, the perfetto protobuf bindings, XlsxWriter)
ship only with the ``torq-compiler[profile]`` extra, so they are imported
lazily and guarded by ``_require_profile_deps`` -- ``import torq.lab`` and
local compile/run stay usable without the extra installed.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from torq.lab import LabError

logger = logging.getLogger("torq.lab.profiling")

try:  # dependencies come from the torq-compiler [profile] extra
    import pandas as pd
    from torq.lab.profiling import perfetto
    from torq.lab.profiling.debug_info import (
        ActionDebugInfo,
        BaseDispatchDebugInfo,
        CombinedDispatchDebugInfo,
        DebugInfo,
        HalDispatchDebugInfo,
        NssProgramWorkUnitDebugInfo,
        parse_profiling_log,
    )
except ImportError:  # pragma: no cover - exercised only without the extra
    pd = None


def _require_profile_deps():
    """Raise a clear error when the optional profiling dependencies are absent."""
    if pd is None:
        raise LabError(
            "Profiling requires the torq-compiler [profile] extra "
            "(pandas, XlsxWriter, protobuf). Install 'torq-compiler[profile]'."
        )


_DESIRED_COLUMNS = [
    "action_id", "job_id", "operation", "invocation_names", "original_operators", "total_time",
    "slice_used_0_in_program", "slice_used_1_in_program",
    "dma_in_used_in_program", "dma_out_used_in_program", "cdma_used_in_program", "css_used_in_program",
    "timestamp_start", "timestamp_end", "location"
]

_HUMAN_FRIENDLY_COLUMNS = {
    "action_id": "Action ID",
    "job_id": "Job ID",
    "operation": "Action in program",
    "invocation_names": "Slice Invocation Names (if applicable)",
    "total_time": "Total Time [us]",
    "slice_used_0_in_program": "Slice 0 Used in NSS Program",
    "slice_used_1_in_program": "Slice 1 Used in NSS Program",
    "dma_in_used_in_program": "DMA In Used in NSS Program",
    "dma_out_used_in_program": "DMA Out Used in NSS Program",
    "cdma_used_in_program": "CDMA Used in CSS Program",
    "css_used_in_program": "CSS Used in CSS Program",
    "timestamp_start": "Timestamp Start [us]",
    "timestamp_end": "Timestamp End [us]",
    "location": "Location",
    "original_operators": "Original Operators"
}


def _to_tabular_data(data: ActionDebugInfo):

    if isinstance(data.workunit, NssProgramWorkUnitDebugInfo):
        nss_data: NssProgramWorkUnitDebugInfo = data.workunit
        invocation_names = nss_data.related_slice_invocations_names
        job_id = nss_data.job_id
    else:
        nss_data = None
        invocation_names = None
        job_id = None

    return {
        "action_id": data.action_id,
        "job_id": job_id,
        "operation": str(data.operation.name),
        "invocation_names": ",".join(invocation_names) if invocation_names else None,
        "total_time": data.total_time_ns / 1000 if data.total_time_ns is not None else None,
        "slice_used_0_in_program": nss_data.slice_used[0] if nss_data else None,
        "slice_used_1_in_program": nss_data.slice_used[1] if nss_data else None,
        "dma_in_used_in_program": nss_data.dma_in_used if nss_data else None,
        "dma_out_used_in_program": nss_data.dma_out_used if nss_data else None,
        "cdma_used_in_program": nss_data.cdma_used if nss_data else None,
        "css_used_in_program": nss_data.css_used if nss_data else None,
        "timestamp_start": data.start_time_ns / 1000 if data.start_time_ns is not None else None,
        "timestamp_end": data.end_time_ns / 1000 if data.end_time_ns is not None else None,
        "location": data.dispatch.debug_info.pretty_print_location(data.location),
        "original_operators": " ".join(data.dispatch.debug_info.get_original_operators(data.location))
    }


def _build_action_dataframe(debug_info: BaseDispatchDebugInfo) -> "pd.DataFrame":
    """Build a DataFrame from the actions in a dispatch debug info container."""
    rows = [_to_tabular_data(x) for x in debug_info.actions.values()]
    return pd.DataFrame(rows, columns=_DESIRED_COLUMNS)


def _write_perfetto_trace(debug_info: BaseDispatchDebugInfo, perfetto_file: str):
    """
    Writes the annotated profiling data to a Perfetto trace file.
    """

    trace_writer = perfetto.PerfettoTraceWriter(perfetto_file)
    perfetto.log_runtime_profile_data(trace_writer, debug_info)

    # Compute Metrics and Render Overview
    metrics_result = perfetto.compute_runtime_metrics(debug_info)
    perfetto.render_overview_tracks("Host Profile", metrics_result['overall_start'], metrics_result['metrics'], trace_writer)

    trace_writer.close()
    logger.debug(f"Perfetto trace complete: {perfetto_file}")

    return perfetto.get_measurements(metrics_result)


def _write_host_annotated_csv(debug_info: BaseDispatchDebugInfo, output_file: str):
    """
    Writes the annotated profiling data to a CSV file.
    """
    df = _build_action_dataframe(debug_info)
    df.to_csv(output_file, index=False, sep=';')


def _write_host_annotated_xlsx(debug_info: BaseDispatchDebugInfo, output_file: str):
    """
    Writes the annotated profiling data to an Excel file with human-friendly formatting.
    """
    df = _build_action_dataframe(debug_info)

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.rename(columns=_HUMAN_FRIENDLY_COLUMNS, inplace=True)
        sheet_name = "Detailed Performance Data"
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        for worksheet in writer.sheets.values():
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].dropna().astype(str).map(len).max() or 0, len(str(col))) + 2
                worksheet.set_column(idx, idx, max_len)


def _write_dispatch_outputs(dispatch_debug_info, output_files):

    measurements = []

    for output_file in output_files:
        if output_file.endswith(".xlsx"):
            _write_host_annotated_xlsx(dispatch_debug_info, output_file)
        elif output_file.endswith(".csv"):
            _write_host_annotated_csv(dispatch_debug_info, output_file)
        elif output_file.endswith(".pb"):
            dispatch_measurements = _write_perfetto_trace(dispatch_debug_info, output_file)
            measurements.append(dispatch_measurements)
        else:
            raise ValueError(f"Unsupported output file format: {output_file}")

    return measurements


def annotate_host_profile_from_files(debug_info, profile_file, output_files):
    """
    Given a debug-info directory generated with --torq-debug-info and a profiling log produced by the runtime with --torq_profile_host,
    generates annotated profiling data that matches the actions in the profiling log with the corresponding operations in the model,
    and outputs this data to the specified output files (Excel, CSV or Perfetto trace).
    """

    _require_profile_deps()

    logger.debug(f"Annotating host profile from {profile_file}")

    invocation_data = parse_profiling_log(profile_file)
    combined_dispatch_debug_info = CombinedDispatchDebugInfo()
    hal_dispatch_debug_info = HalDispatchDebugInfo()
    has_hal_events = False

    for invocation in invocation_data:
        if invocation.dispatch_name.startswith("__HAL"):
            has_hal_events = True
            hal_dispatch_debug_info.append_runtime_events(invocation.event_data, invocation.dispatch_name)
            continue

        # Use a fresh DebugInfo so each invocation keeps independent timing data.
        dispatch_debug_info = DebugInfo(debug_info).get_dispatch(invocation.dispatch_name)
        dispatch_debug_info.load_compile_time_estimates()
        dispatch_debug_info.load_runtime_events(invocation.event_data)
        combined_dispatch_debug_info.add_dispatch(dispatch_debug_info)

    if has_hal_events:
        combined_dispatch_debug_info.add_dispatch(hal_dispatch_debug_info)


    measurements = _write_dispatch_outputs(combined_dispatch_debug_info, output_files)

    if len(measurements) == 1:
        return measurements[0]
    else:
        print("Warning: Multiple measurement sets found, but only one can be returned.")
        print(measurements)
        return None


# ---------------------------------------------------------------------------
# torq.lab orchestration helpers
#
# Thin wrappers used by ModelPipeline to produce the run/compile profiling
# artifacts under ``<work-dir>/profiles/``. Each requires the [profile] extra.
# ---------------------------------------------------------------------------


def annotate_run_profile(debug_dir, host_profile_csv, out_dir) -> dict:
    """Annotate a runtime host profile against compile-time debug info.

    Writes ``annotated_profile.xlsx`` and ``trace.pb`` into ``out_dir`` and
    returns a dict with their paths.
    """
    _require_profile_deps()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = out_dir / "annotated_profile.xlsx"
    trace = out_dir / "trace.pb"
    annotate_host_profile_from_files(
        str(debug_dir), str(host_profile_csv), [str(annotated), str(trace)]
    )
    return {"annotated": annotated, "trace": trace}


def write_compile_trace(debug_dir, out_dir) -> list:
    """Generate compile-time Perfetto trace(s) into ``out_dir``.

    ``convert_to_perfetto`` writes one ``<dispatch>.pb`` per dispatch into a
    directory; they are moved into ``out_dir`` with a ``_compile`` suffix so the
    HTML report tags them as compile-time traces. Returns the produced paths.
    """
    _require_profile_deps()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "_compile_trace"
    perfetto.convert_to_perfetto(str(debug_dir), str(staging))
    produced = []
    for pb in sorted(staging.glob("*.pb")):
        dest = out_dir / f"{pb.stem}_compile.pb"
        shutil.move(str(pb), str(dest))
        produced.append(dest)
    shutil.rmtree(staging, ignore_errors=True)
    return produced


def write_perfetto_report(pb_files, out_html) -> Path:
    """Render a self-contained HTML viewer for the given Perfetto ``.pb`` files."""
    _require_profile_deps()
    from torq.lab.profiling.report import generate_html

    out_html = Path(out_html)
    html = generate_html([Path(p) for p in pb_files])
    out_html.write_text(html, encoding="utf-8")
    return out_html
