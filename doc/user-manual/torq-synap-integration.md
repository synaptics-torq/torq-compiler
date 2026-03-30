# What is SyNAP?

SyNAP is Synaptics AI software framework designed to enable efficient deployment, execution, and management of machine learning models on Synaptics hardware platforms. It provides a unified API for model inference, pre- and post-processing, and supports integration with various backends, including Torq. SyNAP also offers tools for model conversion, runtime management, and GStreamer-based pipelines for multimedia AI applications.

- [SyNAP Public GitHub Repository](https://github.com/synaptics-synap)

---


## SyNAP - Torq Backend

The Torq backend is fully integrated within the SyNAP framework, allowing user to leverage SyNAP’s advanced pre- and post-processing capabilities and GStreamer pipelines while utilizing the Torq runtime for model inference. This integration enables:

- Seamless use of Torq as an inference backend from SyNAP APIs and tools.
- Direct access to SyNAP’s pre- and post-processing, including image, video, and tensor operations, before and after model execution on Torq hardware.
- Utilization of GStreamer pipelines (e.g., GSTSynap) for multimedia AI applications, combining SyNAP’s flexible processing with Torq’s hardware acceleration.

This approach allows developers to build complex AI pipelines, chaining together preprocessing, inference, and postprocessing, all managed through the SyNAP interface while benefiting from Torq’s performance.

## Torq Model to Synap model conversion

This section describes how to prepare, convert ML TORQ models to SyNAP models to execute on Torq HW.

**Conversion Flow:**
- **For Torq:**
  - Use `torq-compile` to convert supported models (e.g., TFLite, ONNX, MLIR) into the Torq-specific VMFB format for execution with the Torq runtime.
  - See detailed conversion steps and usage examples in the [Torq-compiler Step-by-Step Examples](https://github.com/synaptics-torq/torq-compiler/blob/main/doc/user-manual/step_by_step_examples.md).

  **Example: Converting YOLOv8 Model for SyNAP/Torq**

  1. **Download and Convert YOLOv8 Model to TFLite**

    Install dependencies:
    ```sh
    pip install ultralytics tensorflow
    ```

    Python script to export YOLOv8 to TFLite (INT8, 320x320 input):
    ```python
    from ultralytics import YOLO

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Export to TFLite with INT8 quantization and 320x320 input size
    model.export(format="tflite", int8=True, imgsz=320, data="coco8.yaml")
    ```

    This produces a TFLite model file for YOLO.

  2. **Convert TFLite to Torq VMFB**

    Use `torq-compile` to convert the TFLite model to the Torq `.vmfb` format. (See [Torq-compiler examples](https://github.com/synaptics-torq/torq-compiler/blob/main/doc/user-manual/step_by_step_examples.md) for details.)

  3. **Create SyNAP Model Bundle**

    Package the converted files into a `.synap` zip file with the following structure:
    ```
    model.synap
    ├── 0
    │   ├── subgraph_0.json
    │   └── subgraph_0.vmfb
    └── bundle.json
    ```
    - `subgraph_0.vmfb`: The compiled Torq model.
    - `subgraph_0.json`: Metadata for the subgraph.
    - `bundle.json`: Model bundle metadata.


    Example contents for the required JSON files:

    **Example: bundle.json**
    ```json
    {
      "graph": [
        {
          "inputs": [
            {
              "subgraph": -1,
              "out": 0
            }
          ],
          "model": "0/subgraph_0.vmfb",
          "meta": "0/subgraph_0.json"
        }
      ],
      "outputs": [
        {
          "subgraph": 0,
          "out": 0
        }
      ]
    }
    ```


    The input and output parameter names, shapes, types, and quantization details for subgraph_0.json can be obtained by opening your model in [Netron](https://netron.app), which visually displays all model inputs and outputs.
    
    **Example: subgraph_0.json**
    ```json
    {
      "Inputs": {
        "images": {
          "name": "serving_default_images:0",
          "shape": [1, 320, 320, 3],
          "format": "nhwc",
          "quantizer": "asymmetric_affine",
          "quantize": {
            "qtype": "i8",
            "scale": 0.003921568859368563,
            "zero_point": -128
          },
          "data_format": "",
          "scale": 255,
          "mean": [0]
        }
      },
      "Outputs": {
        "output_0": {
          "name": "PartitionedCall:0",
          "shape": [1, 84, 2100],
          "format": "nhwc",
          "quantizer": "asymmetric_affine",
          "quantize": {
            "qtype": "i8",
            "scale": 0.004194467328488827,
            "zero_point": -128,
            "min_value": -0.5368928160365649,
            "max_value": 0.5327103517120861
          },
          "data_format": "yolov8 bb_normalized=1"
        }
      },
      "Recurrent_connections": {},
      "secure": false,
      "delegate": "torq npu=1"
    }
    ```


## Runtime Integration

This section details how the SyNAP and Torq runtimes, libraries, and kernel modules are deployed and integrated to enable seamless interoperability on supported platforms.

**Required Components:**
- **Torq:**
  - `torq-runtime` (core runtime and inference tools)
  - `torq-hosttools` (host-side utilities)
  - `torq-kernel-module` (NPU kernel driver)
- **SyNAP:**
  - `synap-runtime` (core SyNAP runtime libraries)
  - `synap-prebuilts` (prebuilt SyNAP libraries and headers)
  - `synap-models` (reference models and data)

**Platform-Specific Integration:**
- On SL2610, the SyNAP runtime is built with Torq runtime integration enabled, allowing selection of the backend at runtime.
- The Yocto build system ensures all required binaries and libraries are deployed to the target device.
---

# SyNAP - Torq Tests

This section provides practical examples and references for running inference and testing the Torq backend through the SyNAP framework.

## synap_cli Examples

The `synap_cli` tool allows you to run inference, benchmark models, and interact with the SyNAP runtime (including the Torq backend) from the command line. For detailed usage and options, refer to the official documentation:
- [synap_cli Application Guide](https://synaptics-synap.github.io/doc/v/latest/docs/manual/getting_started.html#synap-cli-application)


**Example: Running Inference on YOLO Object Detection Model with synap_cli**

To run inference on a YOLO object detection model using `synap_cli`, use the following command:

```sh
synap_cli -m model.synap -r 5 random
```

Sample output:
```
Loading network: model.synap
Flush/invalidate: yes
Loop period (ms): 0
Network inputs: 1
Network outputs: 1
Input buffer: serving_default_images:0 size: 307200 : random
Output buffer: PartitionedCall:0 size: 176400

Predict #0: 48.03 ms
Predict #1: 47.25 ms
Predict #2: 47.31 ms
Predict #3: 47.22 ms
Predict #4: 47.30 ms

Inference timings (ms):  load: 28.18  init: 59.88  min: 47.22  median: 47.30  max: 48.03  stddev: 0.31  mean: 47.42
```

This example demonstrates running 5 inferences with random input data on the YOLO model using the SyNAP CLI tool.

  

## GStreamer Pipelines and AI Examples

SyNAP provides GStreamer elements (e.g., GSTSynap) for building multimedia AI pipelines that leverage the Torq backend.

> **Note:** The `model.synap` file used in the following GStreamer pipeline example is created using the steps described above in the "Converting YOLOv8 Model for SyNAP/Torq" section.

- **GStreamer AI Object Detection Pipeline Example:**
  ```sh
  gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,format=YUY2,width=640,height=480 \
    ! tee name=t_data \
    t_data. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream \
    ! synavideoconvertscale ! video/x-raw,width=320,height=320,format=RGB \
    ! synapinfer model=/usr/share/synap/models/object_detection/coco/npu/model.synap mode=detector frameinterval=20 \
    ! overlay.inference_sink \
    t_data. ! queue ! synavideoconvertscale ! synapoverlay name=overlay label=/usr/share/synap/models/object_detection/coco/info.json \
    ! waylandsink
  ```
  This pipeline captures video from a camera, preprocesses it, runs object detection using a SyNAP model, overlays results, and displays the output.

## OOBE AI Demo

For an out-of-the-box experience, refer to the OOBE AI Demo, which demonstrates end-to-end AI inference using SyNAP and Torq integration:
- [OOBE AI Demo Documentation](https://synaptics-astra.github.io/doc/v/latest/quickstart/oobe.html#syna-ai-on-sl261x)

---


**References:**
- [Model Conversion Tutorial](https://synaptics-synap.github.io/doc/v/latest/docs/manual/working_with_models.html#model-conversion-tutorial)
- [Running SyNAP Tools](https://synaptics-synap.github.io/doc/v/latest/docs/manual/working_with_models.html#running-synap-tools)
- [synap_cli Application Guide](https://synaptics-synap.github.io/doc/v/latest/docs/manual/getting_started.html#synap-cli-application)

---