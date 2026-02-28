"""
TFLite Layer Extractor

Extracts individual layers from TFLite models by manipulating the
FlatBuffer structure directly. Quantization parameters (scale, zero_point)
and all builtin options (strides, padding, etc.) are preserved exactly.
"""

import copy
import numpy as np
import flatbuffers
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tensorflow.lite.python import schema_py_generated as tflite_schema


@dataclass
class QuantizationParams:
    """Quantization parameters for a tensor."""
    scale: List[float]
    zero_point: List[int]
    quantized_dimension: int = 0


@dataclass
class TensorInfo:
    """Information about a tensor in TFLite model."""
    index: int
    name: str
    shape: List[int]
    dtype: str
    quantization: Optional[QuantizationParams] = None


@dataclass
class OperatorInfo:
    """Information about an operator in TFLite model."""
    index: int
    opcode_index: int
    op_name: str
    inputs: List[int]
    outputs: List[int]


class TFLiteModelParser:
    """Parse TFLite model using TF interpreter and flatbuffer schema."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._interpreter = None
        self._model_content = None
        self._model_obj = None
        
    def _get_interpreter(self):
        """Lazy load TensorFlow interpreter."""
        if self._interpreter is None:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
        return self._interpreter
    
    def _load_model_bytes(self):
        """Load raw model bytes."""
        if self._model_content is None:
            with open(self.model_path, 'rb') as f:
                self._model_content = f.read()
        return self._model_content

    def get_model_object(self) -> 'tflite_schema.ModelT':
        """
        Parse model into a mutable flatbuffer object tree (ModelT).
        This gives full access to operators, tensors, buffers, quantization,
        and all builtin options without needing TF APIs.
        """
        if self._model_obj is None:
            buf = self._load_model_bytes()
            self._model_obj = tflite_schema.ModelT.InitFromPackedBuf(
                bytearray(buf), 0
            )
        return self._model_obj
    
    def get_tensor_details(self) -> Dict[int, TensorInfo]:
        """Get all tensor details including quantization."""
        interp = self._get_interpreter()
        tensors = {}
        
        for t in interp.get_tensor_details():
            quant = None
            if 'quantization_parameters' in t:
                qp = t['quantization_parameters']
                scales = qp.get('scales', [])
                zeros = qp.get('zero_points', [])
                if len(scales) > 0:
                    quant = QuantizationParams(
                        scale=list(scales),
                        zero_point=list(zeros),
                        quantized_dimension=qp.get('quantized_dimension', 0),
                    )
            
            tensors[t['index']] = TensorInfo(
                index=t['index'],
                name=t['name'],
                shape=list(t['shape']),
                dtype=str(t['dtype']).replace("<class 'numpy.", "").replace("'>", ""),
                quantization=quant,
            )
        
        return tensors
    
    def get_operator_details(self) -> List[OperatorInfo]:
        """Get all operator details."""
        interp = self._get_interpreter()
        ops = []
        
        try:
            op_details = interp._get_ops_details()
            for idx, op in enumerate(op_details):
                ops.append(OperatorInfo(
                    index=idx,
                    opcode_index=op.get('index', idx),
                    op_name=op.get('op_name', f'UNKNOWN_{idx}'),
                    inputs=list(op.get('inputs', [])),
                    outputs=list(op.get('outputs', [])),
                ))
        except Exception as e:
            print(f"Warning: Could not get ops: {e}")
        
        return ops


class TFLiteLayerExtractor:
    """
    Extract individual layers from TFLite models preserving quantization.
    """
    
    def __init__(self, model_path: str):
        self.parser = TFLiteModelParser(model_path)
        self.model_path = model_path
        self.tensors = self.parser.get_tensor_details()
        self.operators = self.parser.get_operator_details()
        self.model_obj = self.parser.get_model_object()
        
    def get_layer_info(self) -> List[Dict]:
        """Get information about all layers in the model."""
        layers = []
        for op in self.operators:
            if op.op_name == 'DELEGATE':
                continue
                
            input_tensors = [self.tensors[i] for i in op.inputs if i in self.tensors]
            output_tensors = [self.tensors[i] for i in op.outputs if i in self.tensors]
            
            layers.append({
                'index': op.index,
                'op_name': op.op_name,
                'inputs': [{
                    'index': t.index,
                    'name': t.name,
                    'shape': t.shape,
                    'dtype': t.dtype,
                    'quantized': t.quantization is not None,
                    'quantization': {
                        'scale': t.quantization.scale,
                        'zero_point': t.quantization.zero_point,
                    } if t.quantization else None,
                } for t in input_tensors],
                'outputs': [{
                    'index': t.index,
                    'name': t.name,
                    'shape': t.shape,
                    'dtype': t.dtype,
                    'quantized': t.quantization is not None,
                    'quantization': {
                        'scale': t.quantization.scale,
                        'zero_point': t.quantization.zero_point,
                    } if t.quantization else None,
                } for t in output_tensors],
            })
        return layers
    
    def extract_layer_as_tflite(self, layer_index: int, output_path: Path) -> bool:
        """
        Extract a single layer as a standalone TFLite model.
        
        Uses flatbuffer manipulation to slice the operator out of the model,
        preserving exact quantization parameters, builtin options (strides,
        padding, fused activations, etc.), and all tensor data.
        
        Args:
            layer_index: Index of the layer/operator to extract
            output_path: Path to save the extracted TFLite model
            
        Returns:
            True if extraction succeeded, False otherwise
        """
        if layer_index >= len(self.operators):
            return False
            
        op = self.operators[layer_index]
        if op.op_name == 'DELEGATE':
            return False
        
        # Get tensors for this layer
        input_tensors = [self.tensors[i] for i in op.inputs if i in self.tensors and i >= 0]
        output_tensors = [self.tensors[i] for i in op.outputs if i in self.tensors and i >= 0]
        
        if not input_tensors or not output_tensors:
            return False
        
        # Build using flatbuffer approach — op-agnostic, preserves everything
        try:
            return self._build_layer_flatbuffer(layer_index, output_path)
        except Exception as e:
            print(f"  Error extracting layer {op.op_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_layer_flatbuffer(
        self,
        layer_index: int,
        output_path: Path,
    ) -> bool:
        """
        Build a standalone TFLite model for a single layer using flatbuffer
        manipulation. This is op-agnostic — it works for any TFLite operator
        by copying the operator, its tensors, buffers, builtin options, and
        quantization parameters verbatim from the source model.

        Args:
            layer_index: Index of the operator in the source model's subgraph.
            output_path: Path to write the new single-layer TFLite model.

        Returns:
            True on success.
        """
        src_model = self.model_obj
        src_subgraph = src_model.subgraphs[0]
        src_op = src_subgraph.operators[layer_index]

        # -- Collect all tensor indices used by this operator ----------------
        all_tensor_indices = []
        for idx in list(src_op.inputs) + list(src_op.outputs):
            if idx >= 0 and idx not in all_tensor_indices:
                all_tensor_indices.append(idx)

        # Build old->new tensor index mapping
        tensor_remap = {old: new for new, old in enumerate(all_tensor_indices)}

        # -- Collect needed buffers and build buffer remap -------------------
        # Buffer 0 is always an empty sentinel in TFLite models
        new_buffers = [tflite_schema.BufferT()]  # buffer 0 = empty sentinel
        buffer_remap = {0: 0}

        for old_tidx in all_tensor_indices:
            src_tensor = src_subgraph.tensors[old_tidx]
            old_bidx = src_tensor.buffer
            if old_bidx not in buffer_remap:
                new_bidx = len(new_buffers)
                buffer_remap[old_bidx] = new_bidx
                # Deep copy the buffer (preserves weight/bias data)
                new_buf = tflite_schema.BufferT()
                src_buf = src_model.buffers[old_bidx]
                new_buf.data = copy.copy(src_buf.data) if src_buf.data is not None else None
                new_buf.offset = src_buf.offset
                new_buf.size = src_buf.size
                new_buffers.append(new_buf)

        # -- Build new tensors (deep copy with remapped buffer index) --------
        new_tensors = []
        for old_tidx in all_tensor_indices:
            src_t = src_subgraph.tensors[old_tidx]
            t = copy.deepcopy(src_t)
            t.buffer = buffer_remap[src_t.buffer]
            new_tensors.append(t)

        # -- Build new operator code ----------------------------------------
        src_opcode = src_model.operatorCodes[src_op.opcodeIndex]
        new_opcode = copy.deepcopy(src_opcode)

        # -- Build new operator with remapped tensor indices ----------------
        new_op = copy.deepcopy(src_op)
        new_op.opcodeIndex = 0  # only one opcode in the new model
        new_op.inputs = np.array(
            [tensor_remap[i] if i >= 0 else i for i in src_op.inputs],
            dtype=np.int32,
        )
        new_op.outputs = np.array(
            [tensor_remap[i] if i >= 0 else i for i in src_op.outputs],
            dtype=np.int32,
        )

        # -- Determine subgraph inputs and outputs --------------------------
        # Subgraph inputs = operator input tensors that have NO buffer data
        # (i.e. activations / dynamic inputs, not weights/biases/constants).
        # Subgraph outputs = all operator output tensors.
        subgraph_inputs = []
        seen_inputs = set()
        for old_idx in src_op.inputs:
            if old_idx < 0:
                continue
            new_idx = tensor_remap[old_idx]
            if new_idx in seen_inputs:
                continue
            src_buf = src_model.buffers[src_subgraph.tensors[old_idx].buffer]
            is_constant = src_buf.data is not None and len(src_buf.data) > 0
            if not is_constant:
                subgraph_inputs.append(new_idx)
                seen_inputs.add(new_idx)

        subgraph_outputs = [
            tensor_remap[i] for i in src_op.outputs if i >= 0
        ]

        # If no dynamic inputs found (all inputs are constants), treat
        # the first operator input as the subgraph input anyway so the
        # model has at least one input.
        if not subgraph_inputs and len(src_op.inputs) > 0:
            first_valid = next((i for i in src_op.inputs if i >= 0), None)
            if first_valid is not None:
                subgraph_inputs = [tensor_remap[first_valid]]

        # -- Assemble new subgraph ------------------------------------------
        new_subgraph = tflite_schema.SubGraphT()
        new_subgraph.tensors = new_tensors
        new_subgraph.operators = [new_op]
        new_subgraph.inputs = np.array(subgraph_inputs, dtype=np.int32)
        new_subgraph.outputs = np.array(subgraph_outputs, dtype=np.int32)
        new_subgraph.name = src_subgraph.name

        # -- Assemble new model ---------------------------------------------
        new_model = tflite_schema.ModelT()
        new_model.version = src_model.version
        new_model.operatorCodes = [new_opcode]
        new_model.subgraphs = [new_subgraph]
        new_model.buffers = new_buffers
        new_model.description = b"Extracted single layer"
        # Clear metadata/signature fields that reference the full model
        new_model.metadata = None
        new_model.metadataBuffer = None
        new_model.signatureDefs = None

        # -- Serialize -------------------------------------------------------
        builder = flatbuffers.Builder(1024 * 1024)
        packed = new_model.Pack(builder)
        builder.Finish(packed, b"TFL3")
        new_model_buf = builder.Output()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(bytes(new_model_buf))

        # -- Optional verification -------------------------------------------
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(output_path))
            interp.allocate_tensors()
        except Exception as e:
            print(f"  Warning: extracted model verification failed: {e}")

        return True



def extract_all_layers(
    model_path: str,
    output_dir: str,
    max_layers: int = 0,
    force: bool = False,
) -> List[Dict]:
    """
    Extract all layers from a TFLite model.
    
    Args:
        model_path: Path to TFLite model
        output_dir: Directory to save extracted layers
        max_layers: Maximum number of layers to extract (0 = all)
        force: Force re-extraction even if cached
        
    Returns:
        List of dicts with extraction results
    """
    extractor = TFLiteLayerExtractor(model_path)
    layers = extractor.get_layer_info()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    seen_signatures = set()
    extracted = 0
    
    model_stem = Path(model_path).stem
    
    for layer in layers:
        if layer['op_name'] == 'DELEGATE':
            continue
        
        # Deduplicate by signature
        sig = f"{layer['op_name']}_" + "_".join(
            str(tuple(inp['shape'])) for inp in layer['inputs']
        )
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        
        if max_layers > 0 and extracted >= max_layers:
            break
        
        layer_name = f"{model_stem}_layer_{layer['op_name']}_{layer['index']}"
        layer_file = output_path / f"{layer_name}.tflite"
        
        success = False
        if not layer_file.exists() or force:
            print(f"  Extracting {layer['op_name']}...")
            success = extractor.extract_layer_as_tflite(layer['index'], layer_file)
        else:
            success = True
            # print(f"  Using cached {layer['op_name']}")
        
        results.append({
            'layer_index': layer['index'],
            'op_name': layer['op_name'],
            'layer_file': str(layer_file) if success else None,
            'success': success,
            'inputs': layer['inputs'],
            'outputs': layer['outputs'],
            'is_quantized': any(inp.get('quantized', False) for inp in layer['inputs']),
        })
        
        extracted += 1
    
    return results


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tflite_layer_extractor.py <model.tflite> [output_dir] [max_layers]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./extracted_layers"
    max_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    print(f"Extracting layers from: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # First, print layer info
    extractor = TFLiteLayerExtractor(model_path)
    layers = extractor.get_layer_info()
    
    print(f"\nFound {len(layers)} layers:")
    for layer in layers:
        quant_info = "Q" if any(inp.get('quantized') for inp in layer['inputs']) else " "
        print(f"  [{quant_info}] {layer['index']:3d}: {layer['op_name']}")
        for inp in layer['inputs']:
            q = f" (scale={inp['quantization']['scale'][0]:.6f})" if inp.get('quantization') else ""
            print(f"       in:  {inp['shape']} {inp['dtype']}{q}")
        for out in layer['outputs']:
            q = f" (scale={out['quantization']['scale'][0]:.6f})" if out.get('quantization') else ""
            print(f"       out: {out['shape']} {out['dtype']}{q}")
    
    print(f"\nExtracting layers...")
    results = extract_all_layers(model_path, output_dir, max_layers, force=True)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\nExtraction complete: {success_count}/{len(results)} layers extracted")
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        quant = "Q" if r.get('is_quantized') else " "
        print(f"  [{status}][{quant}] {r['op_name']}: {r.get('layer_file', 'FAILED')}")
