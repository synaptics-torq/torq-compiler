"""Common MLIR utilities for operator extraction and line number parsing."""

import re


def extract_line_numbers_from_location(location_str):
    """
    Extract line numbers from MLIR location string.
    Handles both simple callsites and fused locations.
    
    Examples:
        loc(callsite("...":4:10 at "...":2:3)) -> [4]
        loc(fused[callsite("...":147:12 at "...":2:3), callsite("...":145:12 at "...":2:3)]) -> [147, 145]
    
    Args:
        location_str: MLIR location string to parse
        
    Returns:
        List of line numbers extracted from the location string. Returns [None] if no line numbers found.
    """
    line_numbers = []
    # Find all callsite(...) patterns
    for callsite_match in re.finditer(r'callsite\([^)]+\)', location_str):
        callsite_str = callsite_match.group(0)
        # Extract the first line:col before " at " (this is the actual location, not the call site)
        match = re.search(r'":?(\d+):(\d+)(?:\s+at\s+|")', callsite_str)
        if match:
            line_num = int(match.group(1))
            line_numbers.append(line_num)
    
    return line_numbers if line_numbers else [None]


def get_operator_from_mlir_line(line):
    """
    Extract operator name from an MLIR source line.
    Handles TOSA, ONNX, and other operator patterns.
    
    Examples:
        "%0 = tosa.mul ..." -> "tosa.mul"
        'torch.operator "onnx.Add"' -> "onnx.Add"
        '"tosa.conv2d"' -> "tosa.conv2d"
    
    Args:
        line: A single line from an MLIR file
        
    Returns:
        Operator name as a string, or empty string if no operator found.
    """
    line = line.strip()
    if not line:
        return ""
    
    # Remove result assignment (e.g., "%0 = ...")
    if "=" in line:
        lhs = line.split("=", 1)[0].strip()
        # Only split if the LHS looks like a result definition (starts with % or ()
        if lhs.startswith("%") or lhs.startswith("("):
            line = line.split("=", 1)[1].strip()
    
    # Handle torch.operator "..." pattern (covers ONNX and other operators)
    if line.startswith('torch.operator '):
        match = re.search(r'"([^"]+)"', line)
        if match:
            return match.group(1)

    # Handle TOSA operators (both quoted and unquoted forms)
    if line.startswith('"tosa.'):
        match = re.search(r'"([^"]+)"', line)
        if match:
            return match.group(1)
    elif line.startswith('tosa.'):
        # Return the first token (e.g., "tosa.conv2d")
        return line.split()[0]

    return ""
