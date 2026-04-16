#!/usr/bin/env python3
"""
Generate an HTML viewer for Perfetto trace files (.pb) in the current directory.
Each model name is displayed with a clickable link that opens the trace in Perfetto UI.
"""

import re
import base64
import math
import html as html_module
from pathlib import Path
import json


def extract_model_name(filename):
    """Extract the model name from the filename pattern: test_keras_model[MODEL_NAME].pb"""
    match = re.search(r'\[(.*?)\]', filename)
    if match:
        return match.group(1)
    return filename.replace('.pb', '')


def extract_model_type(model_name, filename=''):
    """Extract the model type from the test function name pattern: test_<type>_"""
    filename_lower = filename.lower()
    
    # Extract type from test_<type>_ pattern
    match = re.search(r'test_([^_]+)_', filename_lower)
    if match:
        return match.group(1)
    
    return 'other'


def parse_time_to_ns(time_str):
    """Parse time string (e.g., '10.351ms') to nanoseconds as integer."""
    if not time_str:
        return None
    
    match = re.search(r'([0-9.]+)\s*([a-zµμ]+)', str(time_str), re.IGNORECASE)
    if not match:
        return None
    
    value = float(match.group(1))
    unit = match.group(2).lower()
    
    # Convert to nanoseconds
    if unit == 'ms':
        return int(value * 1_000_000)
    elif unit in ['µs', 'μs', 'us']:
        return int(value * 1_000)
    elif unit == 's':
        return int(value * 1_000_000_000)
    elif unit == 'ns':
        return int(value)
    
    return None


def extract_perfetto_summary(pb_file_path):
    """Extract summary information from Perfetto .pb file by reading readable text sections"""
    try:
        with open(pb_file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        summary = {
            'total_duration': None,
            'dma_time': None,
            'dma_percent': None,
            'dma_only_time': None,
            'dma_only_percent': None,
            'dma_total_time': None,
            'dma_total_percent': None,
            'cdma_time': None,
            'cdma_percent': None,
            'dma_in_time': None,
            'dma_in_percent': None,
            'dma_out_time': None,
            'dma_out_percent': None,
            'compute_time': None,
            'compute_percent': None,
            'compute_only_time': None,
            'compute_only_percent': None,
            'slice_time': None,
            'slice_percent': None,
            'slice_0_time': None,
            'slice_0_percent': None,
            'slice_1_time': None,
            'slice_1_percent': None,
            'css_time': None,
            'css_percent': None,
            'overlap_time': None,
            'overlap_percent': None,
            'idle_time': None,
            'idle_percent': None,
            'available': False,
            'engine_compilation_time': None
        }
        # Note: time values stored as formatted strings (e.g., "47.545ms"), percent as strings (e.g., "99.94")
        
        # Extract formatted strings directly from .pb file (no parsing/reformatting needed)
        
        # Extract OVERALL duration - maps to "12 OVERALL"
        # Match OVERALL: followed by time value (protobuf has binary data, so don't require newline before)
        overall_match = re.search(r'OVERALL:\s*([0-9.]+[a-zµμ]+)', content, re.IGNORECASE)
        if overall_match:
            summary['total_duration'] = overall_match.group(1)  # Store formatted string directly (e.g., "47.347ms")
            summary['available'] = True
        
        # Extract DMA+CDMA combined total - maps to "00 OVERVIEW DMA COMBINED"
        # Exact match: "DMA+CDMA union:"
        dma_combined_match = re.search(r'DMA\+CDMA\s+union:\s*([0-9.]+[a-zµμ]+)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if dma_combined_match:
            summary['dma_time'] = dma_combined_match.group(1)  # e.g., "47.545ms"
            summary['dma_percent'] = dma_combined_match.group(2)  # e.g., "99.94"
        
        # Extract DMA ONLY (exclusive) - maps to "08 OVERVIEW DMA ONLY"
        # Exact match: "DMA/CDMA ONLY (no compute):"
        dma_only_match = re.search(r'DMA/CDMA\s+ONLY\s*\(no\s+compute\):\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if dma_only_match:
            summary['dma_only_time'] = dma_only_match.group(1)
            summary['dma_only_percent'] = dma_only_match.group(2)
        
        # Extract plain DMA total - maps to "02 OVERVIEW DMA"
        # Must NOT match "DMA+CDMA union" or "DMA/CDMA ONLY"
        # Negative lookbehind ensures we don't match "DMA+CDMA" or within "DMA/CDMA ONLY"
        plain_dma_match = re.search(r'(?<![\+A-Z/])DMA\s+total:\s*([0-9.]+[a-zµμ]+)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if plain_dma_match:
            summary['dma_total_time'] = plain_dma_match.group(1)
            summary['dma_total_percent'] = plain_dma_match.group(2)
        
        # Extract CDMA total - maps to "03 OVERVIEW CDMA"  
        # Must NOT match "DMA+CDMA union"
        cdma_only_match = re.search(r'(?<!\+)CDMA\s+total:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if cdma_only_match:
            summary['cdma_time'] = cdma_only_match.group(1)
            summary['cdma_percent'] = cdma_only_match.group(2)
        
        # Extract DMA In statistics (from individual track data, not overview)
        dma_in_match = re.search(r'DMA[\s_-]*In[^|]*\|\s*dur=([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if dma_in_match:
            summary['dma_in_time'] = dma_in_match.group(1)
            summary['dma_in_percent'] = dma_in_match.group(2)
        
        # Extract DMA Out statistics (from individual track data, not overview)
        dma_out_match = re.search(r'DMA[\s_-]*Out[^|]*\|\s*dur=([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if dma_out_match:
            summary['dma_out_time'] = dma_out_match.group(1)
            summary['dma_out_percent'] = dma_out_match.group(2)
        
        # Extract SLICE+CSS combined total - maps to "01 OVERVIEW COMPUTE COMBINED"
        # Exact match: "SLICE+CSS union:"
        compute_combined_match = re.search(r'SLICE\+CSS\s+union:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if compute_combined_match:
            summary['compute_time'] = compute_combined_match.group(1)
            summary['compute_percent'] = compute_combined_match.group(2)

        # Extract COMPUTE ONLY (exclusive) - maps to "09 OVERVIEW COMPUTE ONLY"
        # Exact match: "COMPUTE ONLY (no DMA/CDMA):"
        compute_only_match = re.search(r'COMPUTE\s+ONLY\s*\(no\s+DMA/CDMA\):\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if compute_only_match:
            summary['compute_only_time'] = compute_only_match.group(1)
            summary['compute_only_percent'] = compute_only_match.group(2)
        
        # Extract SLICE total - maps to "04 OVERVIEW SLICE" (NOT SLICE 0 or SLICE 1)
        # Match "SLICE 0 + 1 union:"
        slice_match = re.search(r'SLICE\s+0\s+\+\s+1\s+union:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if slice_match:
            summary['slice_time'] = slice_match.group(1)
            summary['slice_percent'] = slice_match.group(2)
            
        # Extract SLICE 0 total - maps to "05 OVERVIEW SLICE 0"
        slice_0_match = re.search(r'SLICE\s+0\s+total:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if slice_0_match:
            summary['slice_0_time'] = slice_0_match.group(1)
            summary['slice_0_percent'] = slice_0_match.group(2)

        # Extract SLICE 1 total - maps to "06 OVERVIEW SLICE 1"
        slice_1_match = re.search(r'SLICE\s+1\s+total:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if slice_1_match:
            summary['slice_1_time'] = slice_1_match.group(1)
            summary['slice_1_percent'] = slice_1_match.group(2)
        
        # Extract CSS total - maps to "07 OVERVIEW CSS"
        # Must NOT match "SLICE+CSS union"
        css_match = re.search(r'(?<!\+)CSS\s+total:\s*([0-9.]+(?:\s*[a-zµμ]+)?)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if css_match:
            summary['css_time'] = css_match.group(1)
            summary['css_percent'] = css_match.group(2)
        
        # Extract DMA+COMPUTE overlap - maps to "10 OVERVIEW DMA COMPUTE OVERLAP"
        # Exact match: "DMA/CDMA<->COMPUTE overlap:"
        overlap_match = re.search(r'DMA/CDMA<->COMPUTE\s+overlap:\s*([0-9.]+[a-zµμ]+)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if overlap_match:
            summary['overlap_time'] = overlap_match.group(1)
            summary['overlap_percent'] = overlap_match.group(2)
        
        # Extract IDLE time - maps to "11 OVERVIEW IDLE"
        # Match "IDLE:" followed by time value and percentage (protobuf has binary data)
        idle_match = re.search(r'IDLE:\s*([0-9.]+[a-zµμ]+)\s*\(([0-9.]+)%\)', content, re.IGNORECASE)
        if idle_match:
            summary['idle_time'] = idle_match.group(1)
            summary['idle_percent'] = idle_match.group(2)
        
        return summary
    except Exception as e:
        print(f"  ⚠ Warning: Could not extract summary from {pb_file_path}: {e}")
        return {'available': False}


def get_pb_files():
    """Get all .pb files in the current directory"""
    current_dir = Path('.')
    pb_files = sorted(current_dir.glob('*.pb'))
    return pb_files

def generate_html(pb_files, db_summaries=None, reference_keys=None, reference_summaries=None, test_names=None, test_run_ids=None, test_statuses=None, base_url=None, current_session_id=None, available_sessions=None, default_comparison_session_id=None, non_profiled_tests=None):
    """
    Generate HTML content with all trace files
    
    Args:
        pb_files: List of Path objects pointing to .pb files
        db_summaries: Optional dict mapping pb file path to summary dict from database
                     If provided, skips parsing .pb files for metrics
        test_names: Optional dict mapping pb file path to test case display name
        test_run_ids: Optional dict mapping pb file path to test_run.id for download URLs
        test_statuses: Optional dict mapping pb file path to test status dict with 'outcome' and 'outcome_value'
        base_url: Optional base URL for the server (e.g., 'https://server.hf.space')
                 If provided, download URLs will be absolute
        current_session_id: Optional current session ID for comparison feature
        available_sessions: Optional list of available sessions for comparison dropdown
        default_comparison_session_id: Optional session ID to load by default for comparison
        non_profiled_tests: Optional list of dicts with keys: name, outcome, outcome_class, batch_name
                           These are tests without .pb profiling data that should appear in the same list
    """

    # layersByModel = { 'model': { 'layer': { 'engine1': total_duration, ... , 'engine_n': total_duration } } }
    layersByModel = {}
    models = []
    engines = []
        
    # Collect all model types for filter buttons
    model_types = set()
    
    all_items = list(pb_files)
    if reference_keys:
        all_items.extend(reference_keys)

    all_db_summaries = dict(db_summaries or {})
    if reference_summaries:
        all_db_summaries.update(reference_summaries)


    trace_items = []
    for i, item in enumerate(all_items):
        item_str = str(item)
        
        # Use test name from database if available, otherwise extract from filename
        if test_names and item_str in test_names:
            model_name = test_names[item_str]
        else:
            model_name = extract_model_name(item.name)
        model_type = extract_model_type(model_name, item.name)

        isAlternativeEngine = "alternative_engine" in item_str

        # Get model, layer and engine from the information inside of the brackets e.g. [mbv2_layer_CONV_2D_2-sim-default]
        match = re.search(r"\[(.*?)\]", model_name)
        content = match.group(1) if match else None
        
        test_file = model_name.split("::")[0].replace("tests/", "").removesuffix(".py")

        if content:
            if "_full_model" in content:
                keyword = "_full_model"
            elif '_layer_' in content:
                keyword = '_layer_'
            elif "mlir" in content:
                keyword = ".mlir"
            else:
                keyword = "-"

            content = content.split(keyword, 1)

            if keyword == '-':
                engine = content[1]
            else:
                engine = "-".join(content[1].split('-')[1:])
            
            model = content[0]
            layer = keyword[1:] + content[1].split('-')[0]

            if "ops" in test_file:
                layer = model
            
            if engine not in engines:
                engines.append(engine)

            if test_file not in layersByModel and test_file not in models:
                models.append(test_file)
                layersByModel[test_file] = {}
            
            if layer not in layersByModel[test_file]:
                layersByModel[test_file][layer] = {}

            layersByModel[test_file][layer][engine] = all_db_summaries[item_str]["total_duration"]
        
        model_types.add(model_type)
        file_path = item.name

        download_url = None
        encoded_data = None
        
        if item.suffix == ".pb":
            # Get download URL if test_run_id is available, otherwise use base64 (fallback)
            if test_run_ids and item_str in test_run_ids:
                test_run_id = test_run_ids[item_str]
                # Use absolute URL if base_url provided (for viewing HTML outside server)
                if base_url:
                    download_url = f"{base_url}/download-trace/{test_run_id}/"
                else:
                    download_url = f"/download-trace/{test_run_id}/"
                    # Don't embed data when we have a download URL
            else:
                # Fallback: embed as base64 (for standalone HTML files)
                with open(item, 'rb') as f:
                    file_data = f.read()
                    encoded_data = base64.b64encode(file_data).decode('utf-8')
        
        # Get summary - use database if available, otherwise parse .pb file        
        if all_db_summaries and item_str in all_db_summaries:
            summary = all_db_summaries[item_str]
            # Ensure all expected keys exist with None defaults, then override with DB values
            full_summary = {
                'total_duration': None,
                'dma_time': None,
                'dma_percent': None,
                'dma_only_time': None,
                'dma_only_percent': None,
                'dma_total_time': None,
                'dma_total_percent': None,
                'cdma_time': None,
                'cdma_percent': None,
                'dma_in_time': None,
                'dma_in_percent': None,
                'dma_out_time': None,
                'dma_out_percent': None,
                'compute_time': None,
                'compute_percent': None,
                'compute_only_time': None,
                'compute_only_percent': None,
                'slice_time': None,
                'slice_percent': None,
                'slice_0_time': None,
                'slice_0_percent': None,
                'slice_1_time': None,
                'slice_1_percent': None,
                'css_time': None,
                'css_percent': None,
                'overlap_time': None,
                'overlap_percent': None,
                'idle_time': None,
                'idle_percent': None,
                'available': True,
                'engine_compilation_time' : None
            }

            # Set this values to '-' just to have a uniform render
            if isAlternativeEngine:
                full_summary['dma_time'] = '-'
                full_summary['compute_time'] = '-'
                full_summary['overlap_time'] = '-'
                full_summary['idle_time'] = '-'
                        
            # Update with values from database
            full_summary.update(summary)
            summary = full_summary
        else:
            summary = extract_perfetto_summary(item)
        
        # Build overview HTML
        overview_html = ''
        if summary['available']:
            metrics_html = ''
            
            # 1. TOTAL DURATION
            if summary['total_duration']:
                metrics_html += f'''
                    <div class="metric-card" data-metric="total_duration" data-time-ns="{parse_time_to_ns(summary['total_duration'])}">
                        <div class="metric-label">Total Duration</div>
                        <div style="display:flex; align-items:center; justify-content:space-between;">
                            <div class="metric-value" style="margin-bottom:0;">{summary['total_duration']}</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                    </div>'''
            
            # 2. OVERLAP (DMA/CDMA VS COMPUTE)
            if summary['overlap_time'] is not None and summary['overlap_percent'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="overlap_time" data-time-ns="{parse_time_to_ns(summary['overlap_time'])}">
                        <div class="metric-label">Overlap (DMA/CDMA vs Compute)</div>
                        <div class="metric-value">{summary['overlap_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['overlap_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {min(float(summary['overlap_percent']), 100)}%"></div>
                        </div>
                    </div>'''
            
            # 3. SLICE (0+1) UNION
            if summary['slice_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="slice_time" data-time-ns="{parse_time_to_ns(summary['slice_time'])}">
                        <div class="metric-label">SLICE (0+1) Union</div>
                        <div class="metric-value">{summary['slice_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['slice_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 4. SLICE 0
            if summary['slice_0_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="slice_0_time" data-time-ns="{parse_time_to_ns(summary['slice_0_time'])}">
                        <div class="metric-label">SLICE 0 Time</div>
                        <div class="metric-value">{summary['slice_0_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['slice_0_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_0_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 5. SLICE 1
            if summary['slice_1_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="slice_1_time" data-time-ns="{parse_time_to_ns(summary['slice_1_time'])}">
                        <div class="metric-label">SLICE 1 Time</div>
                        <div class="metric-value">{summary['slice_1_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['slice_1_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_1_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 6. SLICE+CSS UNION
            if summary['compute_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="compute_time" data-time-ns="{parse_time_to_ns(summary['compute_time'])}">
                        <div class="metric-label">SLICE+CSS Union</div>
                        <div class="metric-value">{summary['compute_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['compute_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['compute_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 7. CSS TIME
            if summary['css_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="css_time" data-time-ns="{parse_time_to_ns(summary['css_time'])}">
                        <div class="metric-label">CSS Time</div>
                        <div class="metric-value">{summary['css_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['css_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['css_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 8. COMPUTE ONLY (NO DMA/CDMA)
            if summary['compute_only_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="compute_only_time" data-time-ns="{parse_time_to_ns(summary['compute_only_time'])}">
                        <div class="metric-label">Compute Only (No DMA/CDMA)</div>
                        <div class="metric-value">{summary['compute_only_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['compute_only_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['compute_only_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 9. DMA+CDMA UNION
            if summary['dma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="dma_time" data-time-ns="{parse_time_to_ns(summary['dma_time'])}">
                        <div class="metric-label">DMA+CDMA Union</div>
                        <div class="metric-value">{summary['dma_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['dma_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 10. DMA ONLY (NO COMPUTE)
            if summary['dma_only_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="dma_only_time" data-time-ns="{parse_time_to_ns(summary['dma_only_time'])}">
                        <div class="metric-label">DMA Only (No Compute)</div>
                        <div class="metric-value">{summary['dma_only_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['dma_only_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_only_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 11. DMA TOTAL
            if summary['dma_total_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="dma_total_time" data-time-ns="{parse_time_to_ns(summary['dma_total_time'])}">
                        <div class="metric-label">DMA Total</div>
                        <div class="metric-value">{summary['dma_total_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['dma_total_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_total_percent']}%"></div>
                        </div>
                    </div>'''

            # 12. CDMA TOTAL
            if summary['cdma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="cdma_time" data-time-ns="{parse_time_to_ns(summary['cdma_time'])}">
                        <div class="metric-label">CDMA Total</div>
                        <div class="metric-value">{summary['cdma_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['cdma_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['cdma_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 13. IDLE TIME
            if summary['idle_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="idle_time" data-time-ns="{parse_time_to_ns(summary['idle_time'])}">
                        <div class="metric-label">Idle Time</div>
                        <div class="metric-value">{summary['idle_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['idle_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill idle" style="width: {summary['idle_percent']}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_in_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="dma_in_time" data-time-ns="{parse_time_to_ns(summary['dma_in_time'])}">
                        <div class="metric-label">DMA In Time</div>
                        <div class="metric-value">{summary['dma_in_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['dma_in_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_in_percent']}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_out_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card" data-metric="dma_out_time" data-time-ns="{parse_time_to_ns(summary['dma_out_time'])}">
                        <div class="metric-label">DMA Out Time</div>
                        <div class="metric-value">{summary['dma_out_time']}</div>
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <div class="metric-percent" style="margin-bottom:0;">{summary['dma_out_percent']}%</div>
                            <span class="metric-compare" style="display:none;"></span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_out_percent']}%"></div>
                        </div>
                    </div>'''
            
            if download_url:
                # Use download link (for Django web app)
                overview_html = f'''
                <div class="overview-section">
                    <div class="metrics-grid">
                        {metrics_html}
                    </div>
                    <div class="overview-actions">
                        <div id="currentSessionActions" style="text-align: center;">
                            {'<div style="font-size: 12px; font-weight: 600; color: #4a5568; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Current Session #' + str(current_session_id) + '</div>' if current_session_id else ''}
                            <div style="display: flex; gap: 8px; justify-content: center;">
                                <button class="open-btn" style="min-width: 140px;" onclick="event.stopPropagation(); openTraceFromUrl('{download_url}', '{file_path}')">Open in Perfetto</button>
                                <a href="{download_url}" class="open-btn" style="min-width: 140px; background-color: #6c757d;" onclick="event.stopPropagation();">Download Trace File</a>
                            </div>
                        </div>
                    </div>
                </div>'''
            else:
                # Use embedded base64 (for standalone HTML files)
                overview_html = f'''
                <div class="overview-section">
                    <div class="metrics-grid">
                        {metrics_html}
                    </div>
                    <div class="overview-actions">
                        <button class="open-btn" onclick="event.stopPropagation(); openTrace('{encoded_data}', '{file_path}')">Open in Perfetto</button>
                    </div>
                </div>'''
        else:
            if download_url:
                # Use download link (for Django web app)
                overview_html = f'''
                <div class="overview-section">
                    <div class="overview-unavailable">Overview data not available for this trace</div>
                    <div class="overview-actions">
                        <button class="open-btn" onclick="event.stopPropagation(); openTraceFromUrl('{download_url}', '{file_path}')">Open in Perfetto</button>
                        <a href="{download_url}" class="open-btn" style="background-color: #6c757d;" onclick="event.stopPropagation();">Download Trace File</a>
                    </div>
                </div>'''
            else:
                # Use embedded base64 (for standalone HTML files)
                overview_html = '''
                <div class="overview-section">
                    <div class="overview-unavailable">Overview data not available for this trace</div>
                    <div class="overview-actions">
                        <button class="open-btn" onclick="event.stopPropagation(); openTrace('{encoded_data}', '{file_path}')">Open in Perfetto</button>
                    </div>
                </div>'''
        
        # Build collapsed summary tiles with data attributes for comparison
        collapsed_summary = ''
        if summary['available']:
            summary_tiles = []
            if summary['total_duration']:
                time_ns = parse_time_to_ns(summary['total_duration'])
                summary_tiles.append(f"<div class='summary-tile' data-metric='total_duration' data-time-ns='{time_ns}'><div style='display:flex; flex-direction:column; gap:1px;'><span class='tile-label'>Duration</span><span class='tile-compare' style='display:none;'></span></div><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['total_duration']}</span></div></div>")
            
            if summary['dma_time'] is not None:
                time_ns = parse_time_to_ns(summary['dma_time'])
                summary_tiles.append(f"<div class='summary-tile' data-metric='dma_time' data-time-ns='{time_ns}'><div style='display:flex; flex-direction:column; gap:1px;'><span class='tile-label'>DMA+CDMA</span><span class='tile-compare' style='display:none;'></span></div><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['dma_time']}</span><span class='tile-percent'>{summary['dma_percent']}%</span></div></div>")
            
            if summary['compute_time'] is not None:
                time_ns = parse_time_to_ns(summary['compute_time'])
                summary_tiles.append(f"<div class='summary-tile' data-metric='compute_time' data-time-ns='{time_ns}'><div style='display:flex; flex-direction:column; gap:1px;'><span class='tile-label'>SLICE+CSS</span><span class='tile-compare' style='display:none;'></span></div><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['compute_time']}</span><span class='tile-percent'>{summary['compute_percent']}%</span></div></div>")
            
            # Add overlap from .pb file
            if summary['overlap_time'] is not None:
                time_ns = parse_time_to_ns(summary['overlap_time'])
                summary_tiles.append(f"<div class='summary-tile' data-metric='overlap_time' data-time-ns='{time_ns}'><div style='display:flex; flex-direction:column; gap:1px;'><span class='tile-label'>Overlap</span><span class='tile-compare' style='display:none;'></span></div><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['overlap_time']}</span><span class='tile-percent'>{summary['overlap_percent']}%</span></div></div>")
            
            if summary['idle_time'] is not None:
                time_ns = parse_time_to_ns(summary['idle_time'])
                summary_tiles.append(f"<div class='summary-tile' data-metric='idle_time' data-time-ns='{time_ns}'><div style='display:flex; flex-direction:column; gap:1px;'><span class='tile-label'>Idle</span><span class='tile-compare' style='display:none;'></span></div><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['idle_time']}</span><span class='tile-percent'>{summary['idle_percent']}%</span></div></div>")
            collapsed_summary = ''.join(summary_tiles)
        
        # Create type badge
        type_display = model_type.upper() if model_type != 'other' else 'Other'
        type_badge = f'<span class="type-badge type-badge-{model_type}">{type_display}</span>'
        
        # Create status badge if available
        status_badge = ''
        if test_statuses and item_str in test_statuses:
            status_info = test_statuses[item_str]
            outcome = status_info['outcome']
            outcome_value = status_info['outcome_value']
            
            # Map outcome values: 1=Pass, 2=Fail, 3=Skip, 4=Error
            if outcome_value == 1:
                badge_class = 'status-pass'
            elif outcome_value == 2:
                badge_class = 'status-fail'
            elif outcome_value == 3:
                badge_class = 'status-skip'
            elif outcome_value == 4:
                badge_class = 'status-error'
            else:
                badge_class = 'status-unknown'
            
            status_badge = f'<span class="status-badge {badge_class}">{outcome}</span>'

            # Add failure type badge for failed/error tests
            failure_type = status_info.get('failure_type')
            if failure_type and outcome_value in (2, 4):
                status_badge += f' <span class="status-badge status-failure-type">{html_module.escape(failure_type)}</span>'
        
        # Only include data-encoded if we have it (backward compatibility for standalone HTML)
        data_encoded_attr = f'data-encoded="{encoded_data}"' if encoded_data else ''
        
        # Create status filter data attribute
        status_filter_value = 'unknown'
        failure_type_attr = ''
        if test_statuses and item_str in test_statuses:
            outcome_val = test_statuses[item_str]['outcome_value']
            if outcome_val == 1: status_filter_value = 'pass'
            elif outcome_val == 2: status_filter_value = 'fail'
            elif outcome_val == 3: status_filter_value = 'skip'
            elif outcome_val == 4: status_filter_value = 'error'
            elif outcome_val == 5: status_filter_value = 'xfail'
            elif outcome_val == 6: status_filter_value = 'nxpass'
            ft = test_statuses[item_str].get('failure_type') or ''
            if ft:
                failure_type_attr = f'data-failure-type="{html_module.escape(ft)}"'
                
        engine_html_class = ''
        engine_html_id = ''

        if isAlternativeEngine:
            engine_html_class = 'engine'
            engine_html_id = engine
        
        trace_item = f'''        
        <div class="trace-item {engine_html_class}" id="{engine_html_id}" onclick="toggleExpand(this)" data-filename="{file_path}" {data_encoded_attr} data-type="{model_type}" data-status="{status_filter_value}" {failure_type_attr} data-original-index="{i}" data-test-name="{model_name}">
            <div class="trace-header">
                <div class="trace-name-wrapper">
                    <span class="expand-icon">▶</span>
                    <div class="trace-name">{model_name.replace('*','')}</div>
                </div>
                <div class="trace-badges-and-summary">
                    <div class="trace-badges">
                        {type_badge}
                        {status_badge}
                    </div>

                    <div class="trace-summary-container">
                        <div class="trace-summary">
                            {collapsed_summary}
                        </div>
                    </div>
                </div>
            </div>
            {overview_html}'''

        # Add failure log container for failed profiled tests (loaded on-demand via URL)
        if test_statuses and item_str in test_statuses:
            failure_log_url = test_statuses[item_str].get('failure_log_url')
            if failure_log_url:
                trace_item += f'''
            <div class="failure-log" style="display:none;" data-failure-log-url="{html_module.escape(failure_log_url)}">
                <div class="failure-log-loading">Loading failure log...</div>
            </div>'''

        trace_item += '\n        </div>'
        trace_items.append(trace_item)
    
    
    # Add non-profiled tests to the same list
    if non_profiled_tests:
        for j, test in enumerate(non_profiled_tests):
            idx = len(pb_files) + j
            test_name = f"{test['module']}::{test['name']}[{test['parameters']}]"
            model_type = extract_model_type(test_name, '')
            model_types.add(model_type)
            type_display = model_type.upper() if model_type != 'other' else 'Other'
            type_badge = f'<span class="type-badge type-badge-{model_type}">{type_display}</span>'

            outcome_value = test['outcome_class']
            outcome_display = test['outcome']
            if outcome_value == 1:
                badge_class = 'status-pass'
                status_filter_value = 'pass'
            elif outcome_value == 2:
                badge_class = 'status-fail'
                status_filter_value = 'fail'
            elif outcome_value == 3:
                badge_class = 'status-skip'
                status_filter_value = 'skip'
            elif outcome_value == 4:
                badge_class = 'status-error'
                status_filter_value = 'error'
            elif outcome_value == 5:
                badge_class = 'status-xfail'
                status_filter_value = 'xfail'
            elif outcome_value == 6:
                badge_class = 'status-nxpass'
                status_filter_value = 'nxpass'
            else:
                badge_class = 'status-unknown'
                status_filter_value = 'unknown'
            status_badge = f'<span class="status-badge {badge_class}">{outcome_display}</span>'

            # Add failure type badge for failed/error non-profiled tests
            failure_type = test.get('failure_type')
            if failure_type and outcome_value in (2, 4):
                status_badge += f' <span class="status-badge status-failure-type">{html_module.escape(failure_type)}</span>'

            # Get failure log URL before building the trace item
            failure_log_url = test.get('failure_log_url')

            # Add onclick for failed tests with failure logs
            onclick_attr = 'onclick="toggleExpand(this)"' if failure_log_url else ''
            expand_icon = '<span class="expand-icon">▶</span>' if failure_log_url else ''

            ft_attr = f'data-failure-type="{html_module.escape(failure_type)}"' if failure_type else ''

            trace_item = f'''        <div class="trace-item" {onclick_attr} data-type="{model_type}" data-status="{status_filter_value}" {ft_attr} data-original-index="{idx}" data-test-name="{test_name}">
            <div class="trace-header">
                <div class="trace-name-wrapper">
                    {expand_icon}
                    <div class="trace-name">{test_name}</div>
                </div>
                <div class="trace-badges-and-summary">
                    <div class="trace-badges">
                        {type_badge}
                        {status_badge}
                    </div>
                    <div class="trace-summary">
                    </div>
                </div>
            </div>'''

            # Add failure log container for failed non-profiled tests (loaded on-demand)
            if failure_log_url:
                trace_item += f'''
            <div class="failure-log" style="display:none;" data-failure-log-url="{html_module.escape(failure_log_url)}">
                <div class="failure-log-loading">Loading failure log...</div>
            </div>'''

            trace_item += '\n        </div>'
            trace_items.append(trace_item)

    traces_html = '\n\n'.join(trace_items)

    # Generate filter buttons
    filter_buttons = []
    filter_buttons.append('<button class="filter-btn active" onclick="filterByType(\'all\')" data-type="all">All</button>')
    for model_type in sorted(model_types):
        display_type = model_type.upper() if model_type != 'other' else 'Other'
        filter_buttons.append(f'<button class="filter-btn" onclick="filterByType(\'{model_type}\')" data-type="{model_type}">{display_type}</button>')
    
    # Create status filter buttons separately
    status_filter_buttons = []
    status_filter_buttons.append('<button class="filter-btn active" onclick="filterByStatus(\'all\')" data-status-filter="all">All</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'pass\')" data-status-filter="pass">Pass</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'fail\')" data-status-filter="fail">Fail</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'error\')" data-status-filter="error">Error</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'skip\')" data-status-filter="skip">Skip</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'xfail\')" data-status-filter="xfail">XFail</button>')
    status_filter_buttons.append('<button class="filter-btn" onclick="filterByStatus(\'nxpass\')" data-status-filter="nxpass">NXPass</button>')
    
    # Comparison specific filters (initially hidden)
    status_filter_buttons.append('<button id="btn-pass-fail" class="filter-btn comparison-filter" onclick="filterByStatus(\'pass_fail\')" data-status-filter="pass_fail" style="display:none; border-color: #f56565; color: #c53030;">Pass → Fail</button>')
    status_filter_buttons.append('<button id="btn-fail-pass" class="filter-btn comparison-filter" onclick="filterByStatus(\'fail_pass\')" data-status-filter="fail_pass" style="display:none; border-color: #48bb78; color: #2f855a;">Fail → Pass</button>')
    status_filter_buttons.append('<button id="btn-xfail-pass" class="filter-btn comparison-filter" onclick="filterByStatus(\'xfail_pass\')" data-status-filter="xfail_pass" style="display:none; border-color: #48bb78; color: #2f855a;">XFail → Pass</button>')
    
    filters_html = '\n                '.join(filter_buttons)
    status_filters_html = '\n                '.join(status_filter_buttons)
    
    # Generate comparison dropdown HTML if session info is available
    comparison_html = ''
    comparison_data_json = 'null'
    if current_session_id and available_sessions:
        import json
        comparison_data_json = json.dumps({
            'current_session_id': current_session_id,
            'available_sessions': available_sessions,
            'base_url': base_url or ''
        })
        
        session_options = []
        for session in available_sessions:
            if session['id'] != current_session_id:
                selected = ' selected' if session['id'] == default_comparison_session_id else ''
                session_options.append(f"<option value='{session['id']}'{selected}>Session #{session['id']} - {session['timestamp']} ({session['branch']})</option>")
        
        if session_options:
            comparison_html = f'''
                <div class="filter-section" style="flex: 1 1 auto; margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
                    <div class="filter-label">Compare with Previous Session</div>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <select id="compareSessionSelect" class="compare-select" onchange="loadComparisonData()">
                            <option value="">Select a session to compare...</option>
                            {"".join(session_options)}
                        </select>
                        <button id="clearCompareBtn" class="filter-btn" onclick="clearComparison()" style="display:none;">Clear Comparison</button>
                    </div>
                    <div id="comparisonStatus" style="margin-top: 10px; font-size: 13px; color: #718096;"></div>
                </div>'''
            
        engines_options = []
        for engine in engines:
            engines_options.append(f"<option value='{engine}' > {engine} </option>")

        engines_comparison_html = ''
        if engines:
            # FIXME: 'All engines' option is not working when there are different test and you filter by test_name.
            # When selecting 'All engines' only filtered tests should appear for all engines; instead all tests appear for all engines.
            # engines_options.append(f"<option value='all_engines' > All engines </option>")
            engines_comparison_html = f'''
                <div class="filter-section" style="flex: 1 1 auto; margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
                    <div class="filter-label">Compare with other engines</div>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <select id="compareEngineSelect" class="compare-select" onchange="loadEnginesData()">
                            <option value="">Select an engine</option>
                            {"".join(engines_options)}
                        </select>
                    </div>
                    <div id="comparisonStatus" style="margin-top: 10px; font-size: 13px; color: #718096;"></div>
                </div>'''
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Torq Profiling - Trace Viewer</title>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 95%;
            margin: 0 auto;
            font-size: 16px;
        }}

        .trace-item.engine {{
            display: none;
        }}

        .trace-item.engine.selected {{
            display: block;
        }}
        
        header {{
            background: transparent;
            border-radius: 12px;
            padding: 30px 40px;
            margin-bottom: 30px;
            box-shadow: none;
        }}
        
        h1 {{
            color: white;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .subtitle {{
            color: #718096;
            font-size: 16px;
            margin-top: 8px;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .stat-badge {{
            background: #f7fafc;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            color: #4a5568;
            border: 1px solid #e2e8f0;
        }}
        
        .stat-badge strong {{
            color: #2d3748;
            font-weight: 600;
        }}
        
        .controls {{
            background: white;
            border-radius: 12px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        
        .filter-section {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .filter-label {{
            font-size: 14px;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .filter-buttons {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            background: white;
            color: #4a5568;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .filter-btn:hover {{
            border-color: #667eea;
            color: #667eea;
            transform: translateY(-1px);
        }}
        
        .filter-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }}
        
        .compare-select {{
            flex: 1;
            padding: 10px 16px;
            font-size: 14px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            background: white;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .compare-select:hover {{
            border-color: #667eea;
        }}
        
        .compare-select:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .tile-compare {{
            font-weight: 700;
            font-size: 9px;
            padding: 2px 5px;
            border-radius: 3px;
            display: block;
            white-space: nowrap;
            line-height: 1;
            margin-top: 3px;
        }}
        
        .tile-compare.positive {{
            color: #22543d;
            background-color: #c6f6d5;
        }}
        
        .tile-compare.negative {{
            color: #742a2a;
            background-color: #fed7d7;
        }}
        
        .tile-compare.neutral {{
            color: #4a5568;
            background-color: #e2e8f0;
        }}
        
        .metric-compare {{
            font-weight: 700;
            font-size: 11px;
            padding: 3px 6px;
            border-radius: 4px;
            display: inline-block;
            white-space: nowrap;
        }}
        
        .metric-compare.positive {{
            color: #22543d;
            background-color: #c6f6d5;
        }}
        
        .metric-compare.negative {{
            color: #742a2a;
            background-color: #fed7d7;
        }}
        
        .metric-compare.neutral {{
            color: #4a5568;
            background-color: #e2e8f0;
        }}

        .report-box {{
            padding: 14px 0px 0px 0px;
            display: flex;
            justify-content: flex-end; /* pushes button to the right */
            width: 100%;
        }}
        
        .search-box {{
            position: relative;
            width: 100%;
        }}
        
        .search-input {{
            width: 100%;
            padding: 14px 50px 14px 20px;
            font-size: 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            transition: all 0.3s;
            font-family: inherit;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .search-icon {{
            position: absolute;
            right: 18px;
            top: 50%;
            transform: translateY(-50%);
            color: #a0aec0;
            font-size: 18px;
        }}
        
        .trace-grid {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}

        /* MODAL TO GENERATE REPORT BEGINS */

        .modal {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.45);
            justify-content: center;
            align-items: center;
            padding: 20px;
            z-index: 1000;
        }}

        .modal.show {{
            display: flex;
        }}

        .modal-content {{
            width: min(720px, 100%);
            max-height: 90vh;
            overflow-y: auto;
            position: relative;
        }}

        .close-btn {{
            position: absolute;
            top: 12px;
            right: 14px;
            border: none;
            background: transparent;
            font-size: 28px;
            cursor: pointer;
            line-height: 1;
        }}

        h2 {{
            margin-top: 0;
            margin-bottom: 24px;
        }}

        .form-section {{
            margin-bottom: 24px;
        }}

        .form-section label {{
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .form-section select {{
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            background: white;
            font-size: 14px;
        }}

        .selected-box {{
            margin-top: 12px;
            background: #f7f7f7;
            border-radius: 10px;
            padding: 14px 16px;
        }}

        .selected-box ul {{
            margin: 8px 0 0 18px;
            padding: 0;
        }}

        .selected-box li {{
            margin-bottom: 4px;
        }}

        .modal-actions {{
            display: flex;
            justify-content: flex-end;
            margin-top: 30px;
        }}

        .download-btn {{
            border: none;
            border-radius: 8px;
            padding: 12px 18px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        }}

        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}

        .clear-btn {{
            border: none;
            background: transparent;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.7;
        }}

        .clear-btn:hover {{
            opacity: 1;
            text-decoration: underline;
        }}

        .tag-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}

        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #eef2f7;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 13px;
        }}

        .tag-remove {{
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 14px;
            line-height: 1;
            padding: 0;
        }}

        .empty-selection {{
            color: #666;
            font-size: 13px;
        }}

        .clear-btn {{
            border: none;
            background: transparent;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.75;
        }}

        .clear-btn:hover {{
            opacity: 1;
            text-decoration: underline;
        }}

        .clear-btn:disabled {{
            opacity: 0.35;
            cursor: default;
            text-decoration: none;
        }}

        /* MODAL TO GENERATE REPORT ENDS*/


        .grid-item {{
            background: #f7f7f7;
            border-radius: 10px;
            padding: 20px;
        }}
        
        .trace-item {{
            padding: 0;
            margin-bottom: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #ffffff;
            overflow: hidden;
        }}
        
        .trace-item:hover {{
            border-color: #667eea;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
        }}
        
        .trace-item.expanded {{
            border-color: #667eea;
        }}
        
        .trace-item.expanded .expand-icon {{
            transform: rotate(90deg);
        }}
        
        .trace-item.hidden {{
            display: none;
        }}

        .failure-log {{
            padding: 0 24px 16px;
        }}

        .failure-log-loading {{
            color: #a0aec0;
            font-style: italic;
            padding: 8px 0;
        }}

        .failure-log-pre {{
            background: #1a1a2e;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
            margin: 0;
        }}

        .failure-log-copy-btn {{
            background: #e53e3e;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            margin-bottom: 8px;
        }}

        .trace-header {{
            padding: 20px 24px;
            display: flex;
            flex-wrap: nowrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px 20px;
            min-height: 60px;
        }}
        
        .trace-name-wrapper {{
            flex: 1 1 auto;
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .trace-badges-and-summary {{
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: nowrap;
            flex-shrink: 0;
        }}
        
        .trace-badges {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}

        .trace-summary-container {{
            display: flex;
            flex-direction: column;
            gap: 6px
        }}
        
        .expand-icon {{
            color: #667eea;
            font-size: 12px;
            transition: transform 0.3s ease;
            flex-shrink: 0;
            margin-top: 2px;
        }}
        
        .trace-name {{
            font-weight: 600;
            color: #2d3748;
            font-size: 16px;
            word-wrap: break-word;
            word-break: break-word;
            overflow-wrap: break-word;
            flex: 1 1 auto;
            min-width: 0;
            line-height: 1.4;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 0;
            flex-shrink: 0;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 0;
            flex-shrink: 0;
        }}
        
        .status-pass {{
            background: linear-gradient(135deg, #28a745 0%, #20803a 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
        }}
        
        .status-fail {{
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(220, 53, 69, 0.3);
        }}

        .status-error {{
            background: linear-gradient(135deg, #e07c3e 0%, #c96a2e 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(224, 124, 62, 0.3);
        }}
        
        .status-skip {{
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #000;
            box-shadow: 0 2px 4px rgba(255, 193, 7, 0.3);
        }}
        
        .status-xfail {{
            background: linear-gradient(135deg, #a18cd1 0%, #8e7cc3 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(161, 140, 209, 0.3);
        }}
        
        .status-nxpass {{
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(23, 162, 184, 0.3);
        }}
        
        .status-unknown {{
            background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(108, 117, 125, 0.3);
        }}

        .status-failure-type {{
            background: linear-gradient(135deg, #e07c3e 0%, #c96a2e 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(224, 124, 62, 0.3);
            font-size: 0.7rem;
        }}
        
        .type-badge-mlir {{
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(72, 187, 120, 0.3);
        }}
        
        .type-badge-tosa {{
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(237, 137, 54, 0.3);
        }}
        
        .type-badge-onnx {{
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(66, 153, 225, 0.3);
        }}
        
        .type-badge-keras {{
            background: linear-gradient(135deg, #d53f8c 0%, #b83280 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(213, 63, 140, 0.3);
        }}
        
        .type-badge-tflite {{
            background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(246, 173, 85, 0.3);
        }}
        
        .type-badge-torch {{
            background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(252, 129, 129, 0.3);
        }}
        
        .type-badge-vmfb {{
            background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(159, 122, 234, 0.3);
        }}
        
        .type-badge-other {{
            background: linear-gradient(135deg, #a0aec0 0%, #718096 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(160, 174, 192, 0.3);
        }}
        
        .trace-summary {{
            display: flex;
            gap: 12px;
            flex-wrap: nowrap;
            justify-content: flex-end;
            align-items: center;
            flex: 0 0 auto;
            flex-shrink: 0;
        }}

        .trace-item.expanded .trace-summary {{
            display: none;
        }}

        .summary-tile {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            min-width: 140px;
            max-width: 180px;
            min-height: 45px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            transition: all 0.2s ease;
            flex-shrink: 0;
        }}
        
        .summary-tile:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .tile-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.95;
            font-weight: 600;
            white-space: nowrap;
            line-height: 1.2;
        }}
        
        .tile-value {{
            font-size: 14px;
            font-weight: 700;
            line-height: 1.2;
        }}
        
        .tile-percent {{
            font-size: 11px;
            font-weight: 600;
            opacity: 0.95;
            line-height: 1.2;
        }}
        
        .trace-path {{
            color: #718096;
            font-size: 13px;
            font-family: 'Courier New', monospace;
            word-wrap: break-word;
        }}
        
        .overview-section {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
            background: #f7fafc;
            border-top: 1px solid #e2e8f0;
        }}
        
        .trace-item.expanded .overview-section {{
            max-height: 2000px;
            padding: 24px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        
        .metric-label {{
            font-size: 13px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 4px;
        }}
        
        .metric-percent {{
            font-size: 14px;
            color: #667eea;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .metric-bar {{
            width: 100%;
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .metric-bar-fill {{
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 3px;
            transition: width 0.5s ease;
        }}
        
        .metric-bar-fill.idle {{
            background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%);
        }}
        
        .overview-actions {{
            display: flex;
            justify-content: flex-end;
            gap: 32px;
            align-items: flex-start;
            flex-wrap: wrap;
        }}
        
        .overview-unavailable {{
            text-align: center;
            padding: 40px 20px;
            color: #a0aec0;
            font-style: italic;
            margin-bottom: 16px;
        }}
        
        .open-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            white-space: nowrap;
            text-decoration: none;
            display: inline-block;
            text-align: center;
            min-width: 180px;
        }}
        
        .open-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }}
        
        .open-btn:active {{
            transform: translateY(0);
        }}
        
        .info-banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }}
        
        .info-banner h3 {{
            font-size: 18px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .info-banner p {{
            opacity: 0.95;
            line-height: 1.6;
        }}
        
        .status {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 16px 24px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 500;
            display: none;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 400px;
        }}
        
        .status.success {{
            background: #48bb78;
            color: white;
        }}
        
        .status.error {{
            background: #f56565;
            color: white;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #718096;
            display: none;
        }}
        
        .no-results.show {{
            display: block;
        }}
        
        .no-results-icon {{
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.5;
        }}
        
        footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: white;
            opacity: 0.9;
        }}
        
        /* Responsive design for different screen sizes */
        @media (max-width: 1366px) {{
            .summary-tile {{
                min-width: 100px;
                max-width: 150px;
                padding: 6px 10px;
                gap: 6px;
                height: 40px;
            }}
            
            .tile-label {{
                font-size: 9px;
            }}
            
            .tile-value {{
                font-size: 12px;
            }}
            
            .tile-percent {{
                font-size: 11px;
            }}
            
            .trace-name {{
                font-size: 14px;
            }}
        }}
        
        @media (max-width: 1024px) {{
            .summary-tile {{
                min-width: 90px;
                max-width: 130px;
                padding: 5px 8px;
                gap: 5px;
                height: 38px;
            }}
            
            .tile-label {{
                font-size: 8px;
            }}
            
            .tile-value {{
                font-size: 11px;
            }}
            
            .tile-percent {{
                font-size: 10px;
            }}
            
            .trace-name {{
                font-size: 13px;
            }}
            
            .trace-header {{
                padding: 16px 20px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .trace-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }}
            
            .trace-info {{
                max-width: 100%;
                width: 100%;
            }}
            
            .trace-summary {{
                width: 100%;
                justify-content: flex-start;
                min-width: 100%;
            }}
            
            .summary-tile {{
                min-width: 100%;
                flex: 1 1 auto;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .open-btn {{
                width: 100%;
            }}
        }}
    </style>

</head>
<body>
    <div class="container">
        
        <div class="controls">
            <div style="display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 20px;">
                <div class="filter-section" style="flex: 0 1 auto; margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
                    <div class="filter-label">Filter by Type</div>
                    <div class="filter-buttons">
                        {filters_html}
                    </div>
                </div>
                <div class="filter-section" style="flex: 0 1 auto; margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
                    <div class="filter-label">Filter by Status</div>
                    <div class="filter-buttons">
                        {status_filters_html}
                    </div>
                </div>
                {comparison_html}
                <div class="filter-section" style="flex: 0 1 auto; margin-bottom: 0; padding-bottom: 0; border-bottom: none;">
                    <div class="filter-label">Sort</div>
                    <select id="sortSelect" class="compare-select" onchange="sortTraces()">
                        <option value="original">Original Order</option>
                        <option value="duration_desc">Duration (High → Low)</option>
                        <option value="duration_asc">Duration (Low → High)</option>
                        <option value="change_desc" disabled>Change % (Best → Worst)</option>
                        <option value="change_asc" disabled>Change % (Worst → Best)</option>
                    </select>
                </div>
                {engines_comparison_html}
            </div>
            <div class="search-box">
                <input 
                    type="text" 
                    id="searchInput" 
                    class="search-input" 
                    placeholder="Search models by name or file path..."
                    onkeyup="filterTraces()"
                >
                <span class="search-icon">🔍</span>
            </div>

            <div class="report-box">
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="generateReport()" data-type="all" id = "openModalBtn">Generate Report</button>
                </div>
            </div>

        </div>
        
        <div class="trace-grid">
            {traces_html}
            
            <div class="no-results" id="noResults">
                <div class="no-results-icon">🔍</div>
                <h3>No matches found</h3>
                <p>Try adjusting your search terms</p>
            </div>
        </div>

        <div id="reportModal" class="modal">
            <div class="modal-content trace-grid">
                <button class="close-btn" id="closeModalBtn">&times;</button>

                <h2>Generate Compilation Report</h2>

                <div class="form-section">
                    <div class="section-header">
                        <label for="engineSelect">Available engines</label>
                        <button type="button" class="clear-btn" id="clearEnginesBtn">Clear</button>
                    </div>

                    <select id="engineSelect"></select>

                    <div class="selected-box">
                        <strong>Selected engines:</strong>
                        <div id="selectedEnginesList" class="tag-container"></div>
                    </div>
                </div>

                <div class="form-section">
                    <label for="modelSelect">Choose a model</label>
                    <select id="modelSelect"></select>
                </div>

                <div class="modal-actions">
                    <button id="downloadReportBtn" class="download-btn">Download report</button>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Torq Profiling System</p>
        </footer>
    </div>

    <div id="status" class="status"></div>

    <script>

        let engineSelect;
        let modelSelect;

        let selectedEnginesList;
        let downloadReportBtn;
        let clearEnginesBtn;
        let clearLayersBtn;

        let engines = [];
        let models = [];
        let layersByModel = {{}};

        let selectedEngines = [];
        let currentModel = "";

        let reportModalInitialized = false;

        function toggleExpand(element) {{
            // Close all other expanded items
            const allItems = document.querySelectorAll('.trace-item');
            allItems.forEach(item => {{
                if (item !== element && item.classList.contains('expanded')) {{
                    item.classList.remove('expanded');
                    const failLog = item.querySelector('.failure-log');
                    if (failLog) failLog.style.display = 'none';
                }}
            }});
            
            // Toggle this item
            element.classList.toggle('expanded');
            const failLog = element.querySelector('.failure-log');
            if (failLog) {{
                const isExpanding = element.classList.contains('expanded');
                failLog.style.display = isExpanding ? 'block' : 'none';

                // Lazy-load failure log content from server on first expand
                if (isExpanding && failLog.dataset.failureLogUrl && !failLog.dataset.loaded) {{
                    failLog.dataset.loaded = 'true';
                    fetch(failLog.dataset.failureLogUrl)
                        .then(resp => {{
                            if (!resp.ok) throw new Error('Failed to load');
                            return resp.text();
                        }})
                        .then(text => {{
                            failLog.innerHTML = '<button onclick="copyFailureLog(this, event)" class="failure-log-copy-btn">Copy</button>' +
                                '<pre class="failure-log-pre">' + text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</pre>';
                        }})
                        .catch(() => {{
                            failLog.innerHTML = '<pre class="failure-log-pre" style="color:#999;">Could not load failure log.</pre>';
                        }});
                }}
            }}
        }}

        function copyFailureLog(btn, event) {{
            event.stopPropagation();
            const pre = btn.parentElement.querySelector('pre');
            navigator.clipboard.writeText(pre.textContent).then(() => {{
                btn.textContent = 'Copied!';
                btn.style.background = '#38a169';
                setTimeout(() => {{ btn.textContent = 'Copy'; btn.style.background = '#e53e3e'; }}, 2000);
            }}).catch(() => {{
                // Fallback for non-HTTPS contexts
                const range = document.createRange();
                range.selectNodeContents(pre);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
                document.execCommand('copy');
                sel.removeAllRanges();
                btn.textContent = 'Copied!';
                btn.style.background = '#38a169';
                setTimeout(() => {{ btn.textContent = 'Copy'; btn.style.background = '#e53e3e'; }}, 2000);
            }});
        }}

        function generateReport() {{
            engineSelect = document.getElementById("engineSelect");
            modelSelect = document.getElementById("modelSelect");
            layerSelect = document.getElementById("layerSelect");

            selectedEnginesList = document.getElementById("selectedEnginesList");
            downloadReportBtn = document.getElementById("downloadReportBtn");
            clearEnginesBtn = document.getElementById("clearEnginesBtn");
            clearLayersBtn = document.getElementById("clearLayersBtn");

            engines = {json.dumps(engines)};
            models = {json.dumps(models)};

            layersByModel = {json.dumps(layersByModel)}

            selectedEngines = [];
            currentModel = "";

            openReportModal();
            populateEngines(engines);
            populateModels(models);

            renderSelectedEngines(selectedEngines);
            updateClearButtons();
        }}

        function openReportModal() {{
            const modal = document.getElementById("reportModal");
            const openBtn = document.getElementById("openModalBtn");
            const closeBtn = document.getElementById("closeModalBtn");

            if (!reportModalInitialized) {{
                if (openBtn) {{
                    openBtn.addEventListener("click", () => {{
                        modal.classList.add("show");
                    }});
                }}

                if (closeBtn) {{
                    closeBtn.addEventListener("click", () => {{
                        modal.classList.remove("show");
                    }});
                }}

                if (modal) {{
                    modal.addEventListener("click", (e) => {{
                        if (e.target === modal) {{
                            modal.classList.remove("show");
                        }}
                    }});
                }}

                reportModalInitialized = true;
            }}

            if (modal) {{
                modal.classList.add("show");
            }}
        }}

        function populateEngines(engines) {{
            engineSelect.innerHTML = "";

            if (engines.length === 0) {{
                const option = document.createElement("option");
                option.textContent = "no engines available";
                option.disabled = true;
                option.selected = true;
                engineSelect.appendChild(option);
                return;
            }}

            const placeholder = document.createElement("option");
            placeholder.textContent = "Select engines";
            placeholder.disabled = true;
            placeholder.selected = true;
            placeholder.value = "";
            engineSelect.appendChild(placeholder);

            if (engines.length > 1) {{
                const allOption = document.createElement("option");
                allOption.value = "__all__";
                allOption.textContent = "All engines";
                engineSelect.appendChild(allOption);
            }}

            engines.forEach((engine) => {{
                const option = document.createElement("option");
                option.value = engine;
                option.textContent = engine;
                engineSelect.appendChild(option);
            }});
        }}

        function populateModels(models) {{
            modelSelect.innerHTML = "";

            if (models.length === 0) {{
                const option = document.createElement("option");
                option.textContent = "no models available";
                option.disabled = true;
                option.selected = true;
                modelSelect.appendChild(option);
                return;
            }}

            const placeholder = document.createElement("option");
            placeholder.textContent = "Choose a model";
            placeholder.disabled = true;
            placeholder.selected = true;
            placeholder.value = "";
            modelSelect.appendChild(placeholder);

            models.forEach((model) => {{
                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            }});
        }}

        function createTag(text, onRemove) {{
            const tag = document.createElement("span");
            tag.className = "tag";

            const label = document.createElement("span");
            label.textContent = text;

            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "tag-remove";
            removeBtn.setAttribute("aria-label", `Remove ${{text}}`);
            removeBtn.textContent = "x";
            removeBtn.addEventListener("click", onRemove);

            tag.appendChild(label);
            tag.appendChild(removeBtn);

            return tag;
        }}

        function renderSelectedEngines() {{
            selectedEnginesList.innerHTML = "";

            if (selectedEngines.length === 0) {{
                const empty = document.createElement("span");
                empty.className = "empty-selection";
                empty.textContent = "No engines selected";
                selectedEnginesList.appendChild(empty);
                updateClearButtons();
                return;
            }}

            selectedEngines.forEach((engine) => {{
                const tag = createTag(engine, () => {{
                    selectedEngines = selectedEngines.filter((item) => item !== engine);
                    renderSelectedEngines();
                }});

                selectedEnginesList.appendChild(tag);
            }});

            updateClearButtons();
        }}

        function updateClearButtons() {{
            if (clearEnginesBtn) {{
                clearEnginesBtn.disabled = selectedEngines.length === 0;
            }}
        }}

        document.addEventListener("DOMContentLoaded", () => {{
            engineSelect = document.getElementById("engineSelect");
            modelSelect = document.getElementById("modelSelect");

            selectedEnginesList = document.getElementById("selectedEnginesList");
            downloadReportBtn = document.getElementById("downloadReportBtn");
            clearEnginesBtn = document.getElementById("clearEnginesBtn");

            if (engineSelect) {{
                engineSelect.addEventListener("change", (e) => {{
                    const value = e.target.value;

                    if (!value) {{
                        return;
                    }}

                    if (value === "__all__") {{
                        selectedEngines = [...engines];
                    }} else if (!selectedEngines.includes(value)) {{
                        selectedEngines.push(value);
                    }}

                    renderSelectedEngines(selectedEngines);
                    engineSelect.selectedIndex = 0;
                }});
            }}

            if (modelSelect) {{
                modelSelect.addEventListener("change", (e) => {{
                    currentModel = e.target.value;
                }});
            }}

            if (clearEnginesBtn) {{
                clearEnginesBtn.addEventListener("click", () => {{
                    selectedEngines = [];
                    renderSelectedEngines(selectedEngines);
                    if (engineSelect) {{
                        engineSelect.selectedIndex = 0;
                    }}
                }});
            }}

            if (downloadReportBtn) {{
                downloadReportBtn.addEventListener("click", () => {{
                    const reportModelsAndEngines = {{
                        engines: selectedEngines,
                        model: currentModel,
                    }};

                    generateCsv(reportModelsAndEngines);
                    console.log("Downloading report with:", reportModelsAndEngines);
                }});
            }}

            updateClearButtons();
        }});

        function generateCsv(reportModelsAndEngines) {{
            const allData = { json.dumps(layersByModel) };
            const model = reportModelsAndEngines.model;
            const selectedEngines = reportModelsAndEngines.engines;

            const modelData = allData[model];
            

            if (!modelData) {{
                alert("No data found for model: " + model);
                return;
            }}

            const sortedEntries = Object.entries(modelData).sort(([a], [b]) => a.localeCompare(b));

            const rows = [];

            // Header row
            rows.push(["LAYER", ...selectedEngines]);
            rows.push([]);

            // One row per layer
            for (const [layer, layerData] of sortedEntries) {{
                const row = [layer];

                for (const engine of selectedEngines) {{
                    const value = layerData[engine] ?? "";
                    row.push(value);
                }}

                rows.push(row);
            }}

            const csv = rows
                .map(row => row.map(escapeCsvValue).join(","))
                .join("\\n");

            downloadCsv(csv, `${{model}}_report.csv`);
        }}

        function escapeCsvValue(value) {{
            if (value === null || value === undefined) {{
                return "";
            }}

            const str = String(value);
            return `"${{str.replace(/"/g, '""')}}"`;
        }}

        function downloadCsv(csvContent, filename) {{
            const blob = new Blob([csvContent], {{ type: "text/csv;charset=utf-8;" }});
            const url = URL.createObjectURL(blob);

            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();

            URL.revokeObjectURL(url);
        }}
        
        function openTrace(base64Data, fileName) {{
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'none';
            showStatus(`Loading ${{fileName}}...`, 'success');

            try {{
                // Decode base64 to ArrayBuffer
                const binaryString = atob(base64Data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                const traceData = bytes.buffer;

                // Open Perfetto UI
                const perfettoWindow = window.open('https://ui.perfetto.dev', '_blank');
                
                if (!perfettoWindow) {{
                    showStatus('Please allow pop-ups for this site', 'error');
                    return;
                }}

                // Implement PING/PONG handshake protocol
                let pongReceived = false;
                let pingInterval = null;
                
                // Listen for PONG response from Perfetto UI
                const messageHandler = (evt) => {{
                    if (evt.origin !== 'https://ui.perfetto.dev') {{
                        return;
                    }}
                    
                    // Check if we received a PONG response
                    if (evt.data === 'PONG') {{
                        pongReceived = true;
                        console.log('Received PONG from Perfetto UI');
                        
                        // Stop sending PING messages
                        if (pingInterval) {{
                            clearInterval(pingInterval);
                            pingInterval = null;
                        }}
                        
                        // Now send the trace data
                        try {{
                            perfettoWindow.postMessage({{
                                perfetto: {{
                                    buffer: traceData,
                                    title: fileName,
                                    fileName: fileName
                                }}
                            }}, 'https://ui.perfetto.dev');
                            
                            showStatus(`✓ Trace loaded successfully: ${{fileName}}`, 'success');
                            console.log('Trace data sent to Perfetto UI');
                            
                            // Clean up the message listener after sending
                            window.removeEventListener('message', messageHandler);
                        }} catch (e) {{
                            console.error('Failed to send trace data:', e);
                            showStatus('Error sending trace data. Please try again.', 'error');
                        }}
                    }}
                }};
                
                // Register message listener
                window.addEventListener('message', messageHandler);
                
                // Keep sending PING until we get PONG
                let pingCount = 0;
                const maxPings = 50; // Try for 10 seconds (50 * 200ms)
                
                pingInterval = setInterval(() => {{
                    if (pongReceived) {{
                        clearInterval(pingInterval);
                        return;
                    }}
                    
                    pingCount++;
                    console.log(`Sending PING (attempt ${{pingCount}}/${{maxPings}})`);
                    
                    try {{
                        perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev');
                    }} catch (e) {{
                        console.warn('Failed to send PING:', e);
                    }}
                    
                    if (pingCount >= maxPings) {{
                        clearInterval(pingInterval);
                        window.removeEventListener('message', messageHandler);
                        showStatus('Timeout: Perfetto UI did not respond. Please try again.', 'error');
                        console.error('PONG timeout - Perfetto UI did not respond');
                    }}
                }}, 200); // Send PING every 200ms
                
            }} catch (error) {{
                console.error('Error loading trace:', error);
                showStatus('Error loading trace. Please try again.', 'error');
            }}
        }}
        
        function openTraceFromUrl(downloadUrl, fileName) {{
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'none';
            showStatus(`Loading ${{fileName}}...`, 'success');

            try {{
                // Fetch the trace file from the server
                fetch(downloadUrl)
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error(`HTTP error! status: ${{response.status}}`);
                        }}
                        return response.arrayBuffer();
                    }})
                    .then(traceData => {{
                        // Open Perfetto UI
                        const perfettoWindow = window.open('https://ui.perfetto.dev', '_blank');
                        
                        if (!perfettoWindow) {{
                            showStatus('Please allow pop-ups for this site', 'error');
                            return;
                        }}

                        // Implement PING/PONG handshake protocol
                        let pongReceived = false;
                        let pingInterval = null;
                        
                        // Listen for PONG response from Perfetto UI
                        const messageHandler = (evt) => {{
                            if (evt.origin !== 'https://ui.perfetto.dev') {{
                                return;
                            }}
                            
                            // Check if we received a PONG response
                            if (evt.data === 'PONG') {{
                                pongReceived = true;
                                console.log('Received PONG from Perfetto UI');
                                
                                // Stop sending PING messages
                                if (pingInterval) {{
                                    clearInterval(pingInterval);
                                    pingInterval = null;
                                }}
                                
                                // Now send the trace data
                                try {{
                                    perfettoWindow.postMessage({{
                                        perfetto: {{
                                            buffer: traceData,
                                            title: fileName,
                                            fileName: fileName,
                                            url: downloadUrl
                                        }}
                                    }}, 'https://ui.perfetto.dev');
                                    
                                    showStatus(`✓ Trace loaded successfully: ${{fileName}}`, 'success');
                                    console.log('Trace data sent to Perfetto UI');
                                    
                                    // Clean up the message listener after sending
                                    window.removeEventListener('message', messageHandler);
                                }} catch (e) {{
                                    console.error('Failed to send trace data:', e);
                                    showStatus('Error sending trace data. Please try again.', 'error');
                                }}
                            }}
                        }};
                        
                        // Register message listener
                        window.addEventListener('message', messageHandler);
                        
                        // Keep sending PING until we get PONG
                        let pingCount = 0;
                        const maxPings = 100; // Try for 20 seconds (100 * 200ms)
                        
                        pingInterval = setInterval(() => {{
                            if (pongReceived) {{
                                clearInterval(pingInterval);
                                return;
                            }}
                            
                            pingCount++;
                            console.log(`Sending PING (attempt ${{pingCount}}/${{maxPings}})`);
                            
                            try {{
                                perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev');
                            }} catch (e) {{
                                console.warn('Failed to send PING:', e);
                            }}
                            
                            if (pingCount >= maxPings) {{
                                clearInterval(pingInterval);
                                window.removeEventListener('message', messageHandler);
                                showStatus('Timeout: Perfetto UI did not respond. Please try again.', 'error');
                                console.error('PONG timeout - Perfetto UI did not respond');
                            }}
                        }}, 200); // Send PING every 200ms
                    }})
                    .catch(error => {{
                        console.error('Error fetching trace:', error);
                        showStatus('Error loading trace from server. Please try downloading manually.', 'error');
                    }});
            }} catch (error) {{
                console.error('Error loading trace:', error);
                showStatus('Error loading trace. Please try again.', 'error');
            }}
        }}
        
        let currentTypeFilter = 'all';
        let currentStatusFilter = 'all';
        
        function filterByType(type) {{
            currentTypeFilter = type;
            
            // Update active button
            const buttons = document.querySelectorAll('.filter-btn[data-type]');
            buttons.forEach(btn => {{
                if (btn.getAttribute('data-type') === type) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
            
            // Apply filters
            filterTraces();
        }}
        
        function filterByStatus(status) {{
            currentStatusFilter = status;
            
            // Update active button
            const buttons = document.querySelectorAll('.filter-btn[data-status-filter]');
            buttons.forEach(btn => {{
                if (btn.getAttribute('data-status-filter') === status) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
            
            // Apply filters
            filterTraces();
        }}
        
        function sortTraces() {{
            filterTraces();
        }}

        function filterTraces() {{
            const searchInput = document.getElementById('searchInput').value.toLowerCase();
            const traceItems = document.querySelectorAll('.trace-item');
            const noResults = document.getElementById('noResults');
            const traceGrid = document.querySelector('.trace-grid');
            let visibleCount = 0;
            
            // Split search input into words for flexible matching
            const searchWords = searchInput.trim().split(/\\s+/).filter(word => word.length > 0);
            
            // Create array of items with their match scores
            const itemsWithScores = Array.from(traceItems).map(item => {{
                const name = item.querySelector('.trace-name').textContent.toLowerCase();
                const itemType = item.getAttribute('data-type');
                const itemStatus = item.getAttribute('data-status');
                const originalIndex = parseInt(item.getAttribute('data-original-index'));
                
                // Calculate match score (number of search words that match)
                let matchScore = 0;
                if (searchWords.length > 0) {{
                    matchScore = searchWords.filter(word => name.includes(word)).length;
                }}
                
                const matchesSearch = searchWords.length === 0 || matchScore > 0;
                const matchesType = currentTypeFilter === 'all' || itemType === currentTypeFilter;
                
                let matchesStatus = false;
                if (currentStatusFilter === 'all') {{
                    matchesStatus = true;
                }} else if (currentStatusFilter === 'pass_fail' || currentStatusFilter === 'fail_pass' || currentStatusFilter === 'xfail_pass') {{
                     const compStatus = item.getAttribute('data-comparison-status');
                     matchesStatus = (compStatus === currentStatusFilter);
                }} else {{
                     matchesStatus = (itemStatus === currentStatusFilter);
                }}
                
                const isVisible = matchesSearch && matchesType && matchesStatus;
                
                return {{
                    element: item,
                    score: matchScore,
                    visible: isVisible,
                    originalIndex: originalIndex
                }};
            }});
            
            if (searchWords.length === 0) {{
                 const sortSelect = document.getElementById('sortSelect');
                 const sortBy = sortSelect ? sortSelect.value : 'original';
                 
                 itemsWithScores.sort((a, b) => {{
                     if (sortBy === 'original') {{
                         return a.originalIndex - b.originalIndex;
                     }} 
                     
                     if (sortBy.includes('duration')) {{
                         const getDuration = (el) => {{
                             const t = el.querySelector('.summary-tile[data-time-ns]');
                             return t ? parseFloat(t.getAttribute('data-time-ns')) : 0;
                         }};
                         const vA = getDuration(a.element);
                         const vB = getDuration(b.element);
                         return sortBy === 'duration_desc' ? vB - vA : vA - vB;
                     }}
                     
                     if (sortBy.includes('change')) {{
                         const getChange = (el) => {{
                             const val = el.getAttribute('data-duration-change');
                             const num = parseFloat(val);
                             // Treat missing or 0 change as lowest priority for sorting
                             if (val === null || val === '' || isNaN(num)) return Infinity;
                             return num;
                         }};
                         const vA = getChange(a.element);
                         const vB = getChange(b.element);
                         
                         // Handle valid vs invalid comparison (put invalid at bottom)
                         if (vA === Infinity && vB === Infinity) return 0;
                         if (vA === Infinity) return 1;
                         if (vB === Infinity) return -1;
                         
                         // For change_desc (Best→Worst): sort ascending (negative first = best)
                         // For change_asc (Worst→Best): sort descending (positive first = worst)
                         return sortBy === 'change_desc' ? vA - vB : vB - vA;
                     }}
                     return a.originalIndex - b.originalIndex;
                 }});
            }} else {{
                itemsWithScores.sort((a, b) => {{
                    if (a.visible && b.visible) {{
                        return b.score - a.score;  // Higher score first
                    }}
                    if (a.visible) return -1;
                    if (b.visible) return 1;
                    return 0;
                }});
            }}
            
            // Reorder DOM elements and update visibility
            const noResultsElement = document.getElementById('noResults');
            itemsWithScores.forEach(item => {{
                if (item.visible) {{
                    item.element.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    item.element.classList.add('hidden');
                    item.element.classList.remove('expanded');
                }}
                // Move element to maintain sorted order
                traceGrid.insertBefore(item.element, noResultsElement);
            }});
            
            if (visibleCount === 0) {{
                noResults.classList.add('show');
            }} else {{
                noResults.classList.remove('show');
            }}
        }}

        function showStatus(message, type) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${{type}}`;
            statusDiv.style.display = 'block';
            
            if (type === 'success') {{
                setTimeout(() => {{
                    statusDiv.style.display = 'none';
                }}, 5000);
            }}
        }}
        
        // Comparison functionality
        const comparisonConfig = {comparison_data_json};
        let comparisonMetrics = null;

        function loadEnginesData() {{
            const selectedSessionId = document.getElementById('compareEngineSelect').value;

            // reset
            document.body.classList.remove("show-engines");
            document.querySelectorAll(".trace-item.engine.selected")
                .forEach(el => el.classList.remove("selected"));

            if (selectedSessionId === "all_engines") {{
                document.body.classList.add("show-engines");
                return;
            }}

            const el = document.getElementById(selectedSessionId);
            if (el) {{
                el.classList.add("selected");
            }}
        }}
        
        async function loadComparisonData() {{
            const selectElement = document.getElementById('compareSessionSelect');
            const selectedSessionId = selectElement.value;
            const statusDiv = document.getElementById('comparisonStatus');
            const clearBtn = document.getElementById('clearCompareBtn');
            
            if (!selectedSessionId) {{
                clearComparison();
                return;
            }}
            
            statusDiv.textContent = 'Loading comparison data...';
            statusDiv.style.color = '#667eea';
            
            try {{
                const response = await fetch(`${{comparisonConfig.base_url}}/api/session-metrics/${{selectedSessionId}}/`);
                if (!response.ok) {{
                    throw new Error('Failed to load session data');
                }}
                
                const data = await response.json();
                comparisonMetrics = data.metrics;
                
                applyComparison();
                
                statusDiv.textContent = `Comparing with Session #${{selectedSessionId}}`;
                statusDiv.style.color = '#48bb78';
                clearBtn.style.display = 'inline-block';
            }} catch (error) {{
                console.error('Error loading comparison data:', error);
                statusDiv.textContent = 'Error loading comparison data';
                statusDiv.style.color = '#f56565';
                comparisonMetrics = null;
            }}
        }}
        
        function applyComparison() {{
            if (!comparisonMetrics) return;
            
            // Show comparison filters
            document.querySelectorAll('.comparison-filter').forEach(el => el.style.display = 'inline-block');
            
            // Enable sort options
            const sortSelect = document.getElementById('sortSelect');
            if (sortSelect) {{
                sortSelect.querySelector('option[value="change_desc"]').disabled = false;
                sortSelect.querySelector('option[value="change_asc"]').disabled = false;
            }}
            
            const traceItems = document.querySelectorAll('.trace-item');
            
            traceItems.forEach(item => {{
                const testName = item.getAttribute('data-test-name');
                const summaryTiles = item.querySelectorAll('.summary-tile[data-metric]');
                const metricCards = item.querySelectorAll('.metric-card[data-metric]');
                const overviewSection = item.querySelector('.overview-section');
                
                if (!comparisonMetrics[testName]) {{
                    // Test not found in comparison session
                    summaryTiles.forEach(tile => {{
                        const compareSpan = tile.querySelector('.tile-compare');
                        if (compareSpan) {{
                            compareSpan.style.display = 'inline-block';
                            compareSpan.textContent = 'New';
                            compareSpan.className = 'tile-compare neutral';
                        }}
                    }});
                    metricCards.forEach(card => {{
                        const compareSpan = card.querySelector('.metric-compare');
                        if (compareSpan) {{
                            compareSpan.style.display = 'inline-block';
                            compareSpan.textContent = 'New test';
                            compareSpan.className = 'metric-compare neutral';
                        }}
                    }});
                    item.setAttribute('data-comparison-status', 'new');
                    item.setAttribute('data-duration-change', '0');
                    return;
                }}
                
                const comparisonData = comparisonMetrics[testName];
                
                // Determine Status Change
                const currentStatusRaw = item.getAttribute('data-status') || 'unknown';
                const prevStatusRaw = comparisonData.outcome || 'unknown';
                
                const normalize = (s) => {{
                    const sl = s.toLowerCase();
                    if (sl === 'xfail') return 'xfail';
                    if (sl === 'nxpass') return 'nxpass';
                    if (sl.includes('pass')) return 'pass';
                    if (sl.includes('fail')) return 'fail';
                    if (sl.includes('skip')) return 'skip';
                    return 'unknown';
                }};
                
                const currNorm = normalize(currentStatusRaw);
                const prevNorm = normalize(prevStatusRaw);
                
                let compStatus = 'same';
                if (prevNorm === 'pass' && currNorm === 'fail') compStatus = 'pass_fail';
                else if (prevNorm === 'fail' && currNorm === 'pass') compStatus = 'fail_pass';
                else if (prevNorm === 'xfail' && (currNorm === 'pass' || currNorm === 'nxpass')) compStatus = 'xfail_pass';
                else if (prevNorm !== currNorm) compStatus = 'changed';
                
                item.setAttribute('data-comparison-status', compStatus);
                
                // Apply to summary tiles (collapsed view) and track total_duration change for sorting
                let durationChange = 0;
                summaryTiles.forEach(tile => {{
                    const delta = applyComparisonToElement(tile, '.tile-compare', 'tile-compare', comparisonData.metrics);
                    // Store the total_duration change specifically for sorting purposes
                    const metricName = tile.getAttribute('data-metric');
                    if (metricName === 'total_duration' && delta !== null) {{
                        durationChange = delta;
                    }}
                }});
                
                item.setAttribute('data-duration-change', durationChange);
                
                // Apply to metric cards (expanded view)
                metricCards.forEach(card => {{
                    applyComparisonToElement(card, '.metric-compare', 'metric-compare', comparisonData.metrics);
                }});
                
                // Add comparison trace buttons in expanded view
                if (overviewSection && comparisonData.has_trace) {{
                    const actionsDiv = overviewSection.querySelector('.overview-actions');
                    if (actionsDiv) {{
                        // Remove existing comparison buttons if any
                        const existingCompButtons = actionsDiv.querySelectorAll('.comparison-trace-buttons');
                        existingCompButtons.forEach(btn => btn.remove());
                        
                        // Add comparison trace buttons
                        const comparisonButtons = document.createElement('div');
                        comparisonButtons.className = 'comparison-trace-buttons';
                        comparisonButtons.style.cssText = 'text-align: center;';
                        
                        const selectElement = document.getElementById('compareSessionSelect');
                        const selectedSessionId = selectElement ? selectElement.value : '';
                        
                        comparisonButtons.innerHTML = `
                            <div style="font-size: 12px; font-weight: 600; color: #4a5568; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Comparison Session #${{selectedSessionId}}</div>
                            <div style="display: flex; gap: 8px; justify-content: center;">
                                <button class="open-btn" style="min-width: 140px;" onclick="event.stopPropagation(); openTraceFromUrl('${{comparisonConfig.base_url}}/download-trace/${{comparisonData.test_run_id}}/', 'Session ${{selectedSessionId}} - ${{testName}}')">Open in Perfetto</button>
                                <a href="${{comparisonConfig.base_url}}/download-trace/${{comparisonData.test_run_id}}/" class="open-btn" style="min-width: 140px;" onclick="event.stopPropagation();">Download Trace</a>
                            </div>
                        `;
                        
                        actionsDiv.appendChild(comparisonButtons);
                    }}
                }}
            }});
            
            // Re-apply filters to ensure sort and visibility are updated with new data
            filterTraces();
        }}
        
        function applyComparisonToElement(element, compareSelector, compareClass, comparisonData) {{
            const metricName = element.getAttribute('data-metric');
            const compareSpan = element.querySelector(compareSelector);
            
            if (!compareSpan) return null;
            
            // Get current value in nanoseconds from data attribute
            const currentValueNs = parseFloat(element.getAttribute('data-time-ns'));
            if (isNaN(currentValueNs)) return null;
            
            // Get comparison value directly using the same metric name
            const comparisonValueNs = comparisonData[metricName];
            if (!comparisonValueNs || comparisonValueNs === 0) return null;
            
            const delta = ((currentValueNs - comparisonValueNs) / comparisonValueNs) * 100;
            
            compareSpan.style.display = 'inline-block';
            
            if (Math.abs(delta) < 0.01) {{
                compareSpan.textContent = '±0%';
                compareSpan.className = `${{compareClass}} neutral`;
            }} else if (delta < 0) {{
                // Decrease = improvement (green down arrow)
                compareSpan.textContent = `▼ ${{Math.abs(delta).toFixed(2)}}%`;
                compareSpan.className = `${{compareClass}} positive`;
            }} else {{
                // Increase = regression (red up arrow)
                compareSpan.textContent = `▲ ${{delta.toFixed(2)}}%`;
                compareSpan.className = `${{compareClass}} negative`;
            }}
            return delta;
        }}
        
        function clearComparison() {{
            comparisonMetrics = null;
            
            const selectElement = document.getElementById('compareSessionSelect');
            if (selectElement) selectElement.value = '';
            
            const statusDiv = document.getElementById('comparisonStatus');
            if (statusDiv) statusDiv.textContent = '';
            
            const clearBtn = document.getElementById('clearCompareBtn');
            if (clearBtn) clearBtn.style.display = 'none';
            
            // Remove all comparison trace buttons
            document.querySelectorAll('.comparison-trace-buttons').forEach(el => el.remove());
            
            // Hide all comparison indicators in summary tiles
            document.querySelectorAll('.tile-compare').forEach(span => {{
                span.style.display = 'none';
                span.textContent = '';
            }});
            
            // Hide all comparison indicators in metric cards
            document.querySelectorAll('.metric-compare').forEach(span => {{
                span.style.display = 'none';
                span.textContent = '';
            }});
        }}
        
        // Auto-load default comparison session on page load
        document.addEventListener('DOMContentLoaded', function() {{
            const selectElement = document.getElementById('compareSessionSelect');
            if (selectElement && selectElement.value) {{
                // Small delay to ensure page is fully rendered
                setTimeout(() => {{
                    loadComparisonData();
                }}, 100);
            }}
        }});
    </script>
</body>
</html>'''
    
    return html_content


def main():
    print("AWS FPGA Profiling Log Viewer Generator")
    print("=" * 50)
    
    # Get all .pb files
    pb_files = get_pb_files()
    
    if not pb_files:
        print("❌ No .pb files found in the current directory!")
        return
    
    print(f"✓ Found {len(pb_files)} .pb file(s)")

    # Generate HTML
    html_content = generate_html(pb_files)
    
    # Write to file
    output_file = 'perfetto_viewer.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Generated: {output_file}")
    print("\n✓ Done! Just double-click the HTML file to open it.")
    print("  Click any model to expand and view overview, then 'Open in Perfetto' for full analysis!")


if __name__ == '__main__':
    main()
