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
            'cdma_time': None,
            'cdma_percent': None,
            'dma_in_time': None,
            'dma_in_percent': None,
            'dma_out_time': None,
            'dma_out_percent': None,
            'compute_time': None,
            'compute_percent': None,
            'slice_time': None,
            'slice_percent': None,
            'css_time': None,
            'css_percent': None,
            'idle_time': None,
            'idle_percent': None,
            'available': False
        }
        
        # Extract overall duration
        overall_match = re.search(r'OVERALL:\s*(\d+)', content)
        if overall_match:
            summary['total_duration'] = int(overall_match.group(1))
            summary['available'] = True
        
        # Extract DMA+CDMA combined total
        dma_combined_match = re.search(r'DMA\+CDMA\s+total:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if dma_combined_match:
            summary['dma_time'] = int(dma_combined_match.group(1))
            summary['dma_percent'] = float(dma_combined_match.group(2))
        
        # Extract DMA ONLY (exclusive) statistics
        dma_only_match = re.search(r'DMA\s+ONLY\s*\([^)]*\):\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if dma_only_match:
            summary['dma_only_time'] = int(dma_only_match.group(1))
            summary['dma_only_percent'] = float(dma_only_match.group(2))
        
        # Extract plain DMA total (without CDMA)
        plain_dma_match = re.search(r'(?<!DMA\+)(?<!C)DMA\s+total:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if plain_dma_match:
            summary['cdma_time'] = int(plain_dma_match.group(1))
            summary['cdma_percent'] = float(plain_dma_match.group(2))
        
        # Extract DMA In statistics (from track-level data)
        dma_in_match = re.search(r'DMA[\s_-]*In[^|]*\|\s*dur=(\d+)\s*\(([0-9.]+)%\)', content)
        if dma_in_match:
            summary['dma_in_time'] = int(dma_in_match.group(1))
            summary['dma_in_percent'] = float(dma_in_match.group(2))
        
        # Extract DMA Out statistics (from track-level data)
        dma_out_match = re.search(r'DMA[\s_-]*Out[^|]*\|\s*dur=(\d+)\s*\(([0-9.]+)%\)', content)
        if dma_out_match:
            summary['dma_out_time'] = int(dma_out_match.group(1))
            summary['dma_out_percent'] = float(dma_out_match.group(2))
        
        # Extract SLICE+CSS combined total
        compute_combined_match = re.search(r'SLICE\+CSS\s+total:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if compute_combined_match:
            summary['compute_time'] = int(compute_combined_match.group(1))
            summary['compute_percent'] = float(compute_combined_match.group(2))
        
        # Extract SLICE total (without CSS)
        slice_match = re.search(r'(?<!SLICE\+)SLICE\s+total:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if slice_match:
            summary['slice_time'] = int(slice_match.group(1))
            summary['slice_percent'] = float(slice_match.group(2))
        
        # Extract CSS statistics
        css_match = re.search(r'CSS\s+total:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if css_match:
            summary['css_time'] = int(css_match.group(1))
            summary['css_percent'] = float(css_match.group(2))
        
        # Extract idle time
        idle_match = re.search(r'IDLE:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if idle_match:
            summary['idle_time'] = int(idle_match.group(1))
            summary['idle_percent'] = float(idle_match.group(2))
        
        return summary
    except Exception as e:
        print(f"  ⚠ Warning: Could not extract summary from {pb_file_path}: {e}")
        return {'available': False}


def get_pb_files():
    """Get all .pb files in the current directory"""
    current_dir = Path('.')
    pb_files = sorted(current_dir.glob('*.pb'))
    return pb_files


def generate_html(pb_files):
    """Generate HTML content with all trace files"""
    
    trace_items = []
    for i, pb_file in enumerate(pb_files):
        model_name = extract_model_name(pb_file.name)
        file_path = pb_file.name
        
        # Read and base64 encode the file
        with open(pb_file, 'rb') as f:
            file_data = f.read()
            encoded_data = base64.b64encode(file_data).decode('utf-8')
        
        # Extract overview summary from pb file
        summary = extract_perfetto_summary(pb_file)
        
        # Format duration with appropriate unit
        def format_duration(microseconds):
            if microseconds is None:
                return 'N/A'
            return f'{microseconds} µs'
        
        # Build overview HTML
        overview_html = ''
        if summary['available']:
            metrics_html = ''
            
            if summary['total_duration']:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Total Duration</div>
                        <div class="metric-value">{format_duration(summary['total_duration'])}</div>
                    </div>'''
            
            if summary['dma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA+CDMA Total</div>
                        <div class="metric-value">{format_duration(summary['dma_time'])}</div>
                        <div class="metric-percent">{summary['dma_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_only_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Only (Exclusive)</div>
                        <div class="metric-value">{format_duration(summary['dma_only_time'])}</div>
                        <div class="metric-percent">{summary['dma_only_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_only_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['cdma_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Total</div>
                        <div class="metric-value">{format_duration(summary['cdma_time'])}</div>
                        <div class="metric-percent">{summary['cdma_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['cdma_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_in_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA In Time</div>
                        <div class="metric-value">{format_duration(summary['dma_in_time'])}</div>
                        <div class="metric-percent">{summary['dma_in_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_in_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['dma_out_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">DMA Out Time</div>
                        <div class="metric-value">{format_duration(summary['dma_out_time'])}</div>
                        <div class="metric-percent">{summary['dma_out_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['dma_out_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['compute_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE+CSS Total</div>
                        <div class="metric-value">{format_duration(summary['compute_time'])}</div>
                        <div class="metric-percent">{summary['compute_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['compute_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['slice_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">SLICE Total</div>
                        <div class="metric-value">{format_duration(summary['slice_time'])}</div>
                        <div class="metric-percent">{summary['slice_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['slice_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['css_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">CSS Time</div>
                        <div class="metric-value">{format_duration(summary['css_time'])}</div>
                        <div class="metric-percent">{summary['css_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: {summary['css_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
            if summary['idle_time'] is not None:
                metrics_html += f'''
                    <div class="metric-card">
                        <div class="metric-label">Idle Time</div>
                        <div class="metric-value">{format_duration(summary['idle_time'])}</div>
                        <div class="metric-percent">{summary['idle_percent']:.2f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill idle" style="width: {summary['idle_percent']:.1f}%"></div>
                        </div>
                    </div>'''
            
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
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Duration</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{format_duration(summary['total_duration'])}</span></div></div>")
            
            if summary['dma_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>DMA+CDMA</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{format_duration(summary['dma_time'])}</span><span class='tile-percent'>{summary['dma_percent']:.1f}%</span></div></div>")
            if summary['compute_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>SLICE+CSS</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{format_duration(summary['compute_time'])}</span><span class='tile-percent'>{summary['compute_percent']:.1f}%</span></div></div>")
            
            # Add overlap information
            if summary['dma_time'] is not None and summary['dma_only_time'] is not None and summary['compute_time'] is not None:
                overlap_time = summary['dma_time'] - summary['dma_only_time']
                overlap_percent = (overlap_time / summary['total_duration'] * 100) if summary['total_duration'] else 0
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Overlap</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{format_duration(overlap_time)}</span><span class='tile-percent'>{overlap_percent:.1f}%</span></div></div>")
            
            if summary['idle_time'] is not None:
                summary_tiles.append(f"<div class='summary-tile'><span class='tile-label'>Idle</span><div style='display:flex; flex-direction:column; align-items:flex-end;'><span class='tile-value'>{format_duration(summary['idle_time'])}</span><span class='tile-percent'>{summary['idle_percent']:.1f}%</span></div></div>")
            collapsed_summary = ''.join(summary_tiles)
        
        trace_item = f'''        <div class="trace-item" onclick="toggleExpand(this)" data-filename="{file_path}" data-encoded="{encoded_data}">
            <div class="trace-header">
                <div class="trace-info">
                    <span class="expand-icon">▶</span>
                    <div class="trace-name">{model_name}</div>
                </div>
                <div class="trace-summary">
                    {collapsed_summary}
                </div>
            </div>
            {overview_html}
        </div>'''
        trace_items.append(trace_item)
    
    traces_html = '\n\n'.join(trace_items)
    
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
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            border-radius: 12px;
            padding: 30px 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2d3748;
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
        
        .search-box {{
            position: relative;
            width: 100%;
        }}
        
        .search-input {{
            width: 100%;
            padding: 14px 50px 14px 20px;
            font-size: 15px;
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
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }}
        
        .trace-info {{
            flex: 1;
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
        
        .trace-summary {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
            align-items: center;
        }}
        
        .trace-item.expanded .trace-summary {{
            display: none;
        }}
        
        .summary-tile {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            min-width: 150px;
            height: 45px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            transition: all 0.2s ease;
        }}
        
        .summary-tile:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .tile-label {{
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.9;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .tile-value {{
            font-size: 13px;
            font-weight: 700;
            line-height: 1.2;
            text-align: right;
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
            font-size: 12px;
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
            margin-left: 20px;
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
        <header>
            <h1>
                <span>🔬</span>
                Torq Profiling Trace Viewer
            </h1>
            <p class="subtitle">Performance analysis and trace visualization powered by Perfetto</p>
            <div class="stats">
                <div class="stat-badge"><strong>{len(pb_files)}</strong> Trace Files Available</div>
            </div>
        </header>
        
        <div class="info-banner">
            <h3>How to Use</h3>
            <p>Click any model to expand and view performance overview. Click "Open in Perfetto" button to launch detailed trace analysis.</p>
        </div>
        
        <div class="controls">
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
        
        function filterTraces() {{
            const searchInput = document.getElementById('searchInput').value.toLowerCase();
            const traceItems = document.querySelectorAll('.trace-item');
            const noResults = document.getElementById('noResults');
            let visibleCount = 0;
            
            traceItems.forEach(item => {{
                const name = item.querySelector('.trace-name').textContent.toLowerCase();
                
                if (name.includes(searchInput)) {{
                    item.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    item.classList.add('hidden');
                    // Collapse hidden items
                    item.classList.remove('expanded');
                }}
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
