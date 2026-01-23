#!/usr/bin/env python3
"""
Generate an HTML viewer for Perfetto trace files (.pb) in the current directory.
Each model name is displayed with a clickable link that opens the trace in Perfetto UI.
"""

import re
import base64
import math
from pathlib import Path


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
            'available': False
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


def generate_html(pb_files, db_summaries=None, test_names=None, test_run_ids=None, base_url=None):
    """
    Generate HTML content with all trace files
    
    Args:
        pb_files: List of Path objects pointing to .pb files
        db_summaries: Optional dict mapping pb file path to summary dict from database
                     If provided, skips parsing .pb files for metrics
        test_names: Optional dict mapping pb file path to test case display name
        test_run_ids: Optional dict mapping pb file path to test_run.id for download URLs
        base_url: Optional base URL for the server (e.g., 'https://server.hf.space')
                 If provided, download URLs will be absolute
    """
    
    # Collect all model types for filter buttons
    model_types = set()
    
    trace_items = []
    for i, pb_file in enumerate(pb_files):
        pb_file_str = str(pb_file)
        
        # Use test name from database if available, otherwise extract from filename
        if test_names and pb_file_str in test_names:
            model_name = test_names[pb_file_str]
        else:
            model_name = extract_model_name(pb_file.name)
        
        model_type = extract_model_type(model_name, pb_file.name)
        model_types.add(model_type)
        file_path = pb_file.name
        
        # Get download URL if test_run_id is available, otherwise use base64 (fallback)
        if test_run_ids and pb_file_str in test_run_ids:
            test_run_id = test_run_ids[pb_file_str]
            # Use absolute URL if base_url provided (for viewing HTML outside server)
            if base_url:
                download_url = f"{base_url}/download-trace/{test_run_id}/"
            else:
                download_url = f"/download-trace/{test_run_id}/"
            encoded_data = None  # Don't embed data when we have a download URL
        else:
            # Fallback: embed as base64 (for standalone HTML files)
            with open(pb_file, 'rb') as f:
                file_data = f.read()
                encoded_data = base64.b64encode(file_data).decode('utf-8')
            download_url = None
        
        # Get summary - use database if available, otherwise parse .pb file        
        if db_summaries and pb_file_str in db_summaries:
            summary = db_summaries[pb_file_str]
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
                'available': True
            }
            # Update with values from database
            full_summary.update(summary)
            summary = full_summary
        else:
            summary = extract_perfetto_summary(pb_file)
        
        # Build overview HTML
        overview_html = ''
        if summary['available']:
            metrics_html = ''
            
            # 1. TOTAL DURATION
            if summary['total_duration']:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Total Duration</div>
                        <div class="metric-value">{summary['total_duration']}</div>
                    </div>'''
            
            # 2. OVERLAP (DMA/CDMA VS COMPUTE)
            if summary['overlap_time'] is not None and summary['overlap_percent'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Overlap (DMA/CDMA vs Compute)</div>
                        <div class="metric-value">{summary['overlap_time']}</div>
                        <div class="metric-percent">{summary['overlap_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {min(float(summary['overlap_percent']), 100)}%"></div>
                        </div>
                    </div>'''
            
            # 3. SLICE (0+1) UNION
            if summary['slice_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE (0+1) Union</div>
                        <div class="metric-value">{summary['slice_time']}</div>
                        <div class="metric-percent">{summary['slice_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 4. SLICE 0
            if summary['slice_0_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE 0 Time</div>
                        <div class="metric-value">{summary['slice_0_time']}</div>
                        <div class="metric-percent">{summary['slice_0_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_0_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 5. SLICE 1
            if summary['slice_1_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE 1 Time</div>
                        <div class="metric-value">{summary['slice_1_time']}</div>
                        <div class="metric-percent">{summary['slice_1_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_1_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 6. SLICE+CSS UNION
            if summary['compute_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE+CSS Union</div>
                        <div class="metric-value">{summary['compute_time']}</div>
                        <div class="metric-percent">{summary['compute_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['compute_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 7. CSS TIME
            if summary['css_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">CSS Time</div>
                        <div class="metric-value">{summary['css_time']}</div>
                        <div class="metric-percent">{summary['css_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['css_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 8. COMPUTE ONLY (NO DMA/CDMA)
            if summary['compute_only_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Compute Only (No DMA/CDMA)</div>
                        <div class="metric-value">{summary['compute_only_time']}</div>
                        <div class="metric-percent">{summary['compute_only_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['compute_only_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 9. DMA+CDMA UNION
            if summary['dma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA+CDMA Union</div>
                        <div class="metric-value">{summary['dma_time']}</div>
                        <div class="metric-percent">{summary['dma_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 10. DMA ONLY (NO COMPUTE)
            if summary['dma_only_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Only (No Compute)</div>
                        <div class="metric-value">{summary['dma_only_time']}</div>
                        <div class="metric-percent">{summary['dma_only_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_only_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 11. DMA TOTAL
            if summary['dma_total_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Total</div>
                        <div class="metric-value">{summary['dma_total_time']}</div>
                        <div class="metric-percent">{summary['dma_total_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_total_percent']}%"></div>
                        </div>
                    </div>'''

            # 12. CDMA TOTAL
            if summary['cdma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">CDMA Total</div>
                        <div class="metric-value">{summary['cdma_time']}</div>
                        <div class="metric-percent">{summary['cdma_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['cdma_percent']}%"></div>
                        </div>
                    </div>'''
            
            # 13. IDLE TIME
            if summary['idle_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Idle Time</div>
                        <div class="metric-value">{summary['idle_time']}</div>
                        <div class="metric-percent">{summary['idle_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill idle" style="width: {summary['idle_percent']}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_in_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA In Time</div>
                        <div class="metric-value">{summary['dma_in_time']}</div>
                        <div class="metric-percent">{summary['dma_in_percent']}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_in_percent']}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_out_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Out Time</div>
                        <div class="metric-value">{summary['dma_out_time']}</div>
                        <div class="metric-percent">{summary['dma_out_percent']}%</div>
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
                        <button class="open-btn" onclick="event.stopPropagation(); openTraceFromUrl('{download_url}', '{file_path}')">Open in Perfetto</button>
                        <a href="{download_url}" class="open-btn" style="background-color: #6c757d;" onclick="event.stopPropagation();">Download Trace File</a>
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
        
        # Build collapsed summary tiles
        collapsed_summary = ''
        if summary['available']:
            summary_tiles = []
            if summary['total_duration']:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Duration</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['total_duration']}</span></div></div>")
            
            if summary['dma_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>DMA+CDMA</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['dma_time']}</span><span class='tile-percent'>{summary['dma_percent']}%</span></div></div>")
            if summary['compute_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>SLICE+CSS</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['compute_time']}</span><span class='tile-percent'>{summary['compute_percent']}%</span></div></div>")
            
            # Add overlap from .pb file
            if summary['overlap_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Overlap</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['overlap_time']}</span><span class='tile-percent'>{summary['overlap_percent']}%</span></div></div>")
            
            if summary['idle_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Idle</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{summary['idle_time']}</span><span class='tile-percent'>{summary['idle_percent']}%</span></div></div>")
            collapsed_summary = ''.join(summary_tiles)
        
        # Create type badge
        type_display = model_type.upper() if model_type != 'other' else 'Other'
        type_badge = f'<span class="type-badge type-badge-{model_type}">{type_display}</span>'
        
        # Only include data-encoded if we have it (backward compatibility for standalone HTML)
        data_encoded_attr = f'data-encoded="{encoded_data}"' if encoded_data else ''
        
        trace_item = f'''        <div class="trace-item" onclick="toggleExpand(this)" data-filename="{file_path}" {data_encoded_attr} data-type="{model_type}" data-original-index="{i}">
            <div class="trace-header">
                <div class="trace-info">
                    <span class="expand-icon">▶</span>
                    <div class="trace-name">{model_name}</div>
                    {type_badge}
                </div>
                <div class="trace-summary">
                    {collapsed_summary}
                </div>
            </div>
            {overview_html}
        </div>'''
        trace_items.append(trace_item)
    
    traces_html = '\n\n'.join(trace_items)
    
    # Generate filter buttons
    filter_buttons = []
    filter_buttons.append('<button class="filter-btn active" onclick="filterByType(\'all\')" data-type="all">All</button>')
    for model_type in sorted(model_types):
        display_type = model_type.upper() if model_type != 'other' else 'Other'
        filter_buttons.append(f'<button class="filter-btn" onclick="filterByType(\'{model_type}\')" data-type="{model_type}">{display_type}</button>')
    filters_html = '\n                '.join(filter_buttons)
    
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
        
        .trace-header {{
            padding: 20px 24px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            min-height: 60px;
        }}
        
        .trace-info {{
            flex: 1 1 250px;
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .expand-icon {{
            color: #667eea;
            font-size: 12px;
            transition: transform 0.3s ease;
            flex-shrink: 0;
        }}
        
        .trace-name {{
            font-weight: 600;
            color: #2d3748;
            font-size: 16px;
            word-wrap: break-word;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 12px;
            flex-shrink: 0;
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
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
            align-items: center;
            flex: 0 1 auto;
            min-width: 0;
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
            min-width: 120px;
            max-width: 180px;
            height: 45px;
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
            opacity: 0.9;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .tile-value {{
            font-size: 14px;
            font-weight: 700;
            line-height: 1.2;
            text-align: right;
        }}
        
        .tile-percent {{
            font-size: 12px;
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
            gap: 12px;
            align-items: center;
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
            
            .trace-summary {{
                width: 100%;
                justify-content: flex-start;
            }}
            
            .summary-tile {{
                min-width: 100px;
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
            <div class="filter-section">
                <div class="filter-label">Filter by Type</div>
                <div class="filter-buttons">
                    {filters_html}
                </div>
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
        </div>
        
        <div class="trace-grid">
{traces_html}
            
            <div class="no-results" id="noResults">
                <div class="no-results-icon">🔍</div>
                <h3>No matches found</h3>
                <p>Try adjusting your search terms</p>
            </div>
        </div>
        
        <footer>
            <p>Torq Profiling System</p>
        </footer>
    </div>

    <div id="status" class="status"></div>

    <script>
        function toggleExpand(element) {{
            // Close all other expanded items
            const allItems = document.querySelectorAll('.trace-item');
            allItems.forEach(item => {{
                if (item !== element && item.classList.contains('expanded')) {{
                    item.classList.remove('expanded');
                }}
            }});
            
            // Toggle this item
            element.classList.toggle('expanded');
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

                // Send trace data to Perfetto UI
                let attemptCount = 0;
                const maxAttempts = 15;
                
                const sendTrace = setInterval(() => {{
                    attemptCount++;
                    
                    try {{
                        perfettoWindow.postMessage({{
                            perfetto: {{
                                buffer: traceData,
                                title: fileName,
                            }}
                        }}, 'https://ui.perfetto.dev');
                        
                        if (attemptCount === 3) {{
                            showStatus(`✓ Trace loaded successfully: ${{fileName}}`, 'success');
                        }}
                    }} catch (e) {{
                        console.log('PostMessage attempt', attemptCount, 'failed:', e);
                    }}
                    
                    if (attemptCount >= maxAttempts) {{
                        clearInterval(sendTrace);
                    }}
                }}, 500);
                
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

                        // Send trace data to Perfetto UI
                        let attemptCount = 0;
                        const maxAttempts = 15;
                        
                        const sendTrace = setInterval(() => {{
                            attemptCount++;
                            
                            try {{
                                perfettoWindow.postMessage({{
                                    perfetto: {{
                                        buffer: traceData,
                                        title: fileName,
                                    }}
                                }}, 'https://ui.perfetto.dev');
                                
                                if (attemptCount === 3) {{
                                    showStatus(`✓ Trace loaded successfully: ${{fileName}}`, 'success');
                                }}
                            }} catch (e) {{
                                console.log('PostMessage attempt', attemptCount, 'failed:', e);
                            }}
                            
                            if (attemptCount >= maxAttempts) {{
                                clearInterval(sendTrace);
                            }}
                        }}, 500);
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
        
        function filterByType(type) {{
            currentTypeFilter = type;
            
            // Update active button
            const buttons = document.querySelectorAll('.filter-btn');
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
        
        function filterTraces() {{
            const searchInput = document.getElementById('searchInput').value.toLowerCase();
            const traceItems = document.querySelectorAll('.trace-item');
            const noResults = document.getElementById('noResults');
            const traceGrid = document.querySelector('.trace-grid');
            let visibleCount = 0;
            
            // Split search input into words for flexible matching
            const searchWords = searchInput.trim().split(/\s+/).filter(word => word.length > 0);
            
            // Create array of items with their match scores
            const itemsWithScores = Array.from(traceItems).map(item => {{
                const name = item.querySelector('.trace-name').textContent.toLowerCase();
                const itemType = item.getAttribute('data-type');
                const originalIndex = parseInt(item.getAttribute('data-original-index'));
                
                // Calculate match score (number of search words that match)
                let matchScore = 0;
                if (searchWords.length > 0) {{
                    matchScore = searchWords.filter(word => name.includes(word)).length;
                }}
                
                const matchesSearch = searchWords.length === 0 || matchScore > 0;
                const matchesType = currentTypeFilter === 'all' || itemType === currentTypeFilter;
                const isVisible = matchesSearch && matchesType;
                
                return {{
                    element: item,
                    score: matchScore,
                    visible: isVisible,
                    originalIndex: originalIndex
                }};
            }});
            
            // Sort: if search is empty, use original order; otherwise sort by score
            if (searchWords.length === 0) {{
                itemsWithScores.sort((a, b) => a.originalIndex - b.originalIndex);
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
