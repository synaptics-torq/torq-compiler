# ADR0001 - Pluggable tools for `torq-lab`

## Status

Accepted

## Context

The Torq software suite contains the following Python packages:

- `torq-compiler`: provides the compiler, python bindings and the torq-lab tool
- `torq-runtime`: provides runtime libraries and python bindings
- `torq-tools`: provides tools to import and convert various models in a format
   suitable to be used with the Torq software suite.

The project also includes the `torq-examples` repository, which provides
example code to run models using the Torq software suite. This repository
provides both generic code to validate tasks specific metrics (such as
accuracy) and model specific code to perform inference for a specific 
task (e.g. pose estimation or text generation).

The main entry point for users of the Torq software suite is the `torq-lab`
tool, which provides a single interface to build, deploy and validate models.

In the current setup a user must call `torq-lab`, `torq-tools` and 
`torq-examples` code depending on what they want to achieve, which is
causing confusion. To provide a simple user workflow, we would like to 
make sure all interactions are done through the `torq-lab` tool.

Testing and validation of the software suite also requires acquiring,
generating and verifying models. At the moment code for these tasks is 
scattered across packages and there are many functions reimplemented
in the various packages, leading to validation mismatches and redundant
implementations. Moreover there is no automated process to publish
models generated with a given release.

Another challenge is that the `torq-lab` tool cannot be extended 
without modifying the `torq-compiler` package itself, which makes it
very difficult for third parties and customers to extend its 
functionality to support new models. It also makes it difficult
to maintain support for private models.

The dependencies of `torq-tools` are extensive and not 
required for general use of the `torq-compiler` package. This has to
be taken into account when restructuring the packages.

## Decision

We restructure `torq-lab` to use a plugin-based architecture using
standard Python entry points.

The following types of plugins are supported:

- Model Downloaders: acquire models from an external repository (e.g. Hugging Face or YOLO download interfaces)

- Data Downloaders: acquire datasets from an external repository (e.g. Hugging Face Datasets)

- Importers: handle the process of importing models from various formats into the Torq software suite (e.g. translate ONNX or TFLite models to MLIR)

- Optimizers: use external tools to transform models into a more efficient representation (e.g. quantize models using ONNX framework)

- Validators: perform checks and validations on models to ensure correctness and compliance with expected results (e.g. test model accuracy against some reference dataset or comparison against a reference inference engine)

- Models: combines the other plugins to define the full process to download, prepare, compile and validate a given model. Models may provide validation of intermediate steps of the model processing pipeline.

The `torq-lab` tool is extended to dynamically discover and 
provide a CLI and API for all the plugins.

```
torq-lab list-model-downloaders
torq-lab download-model <downloader_name> <downloader_parameters> <model_file>
torq-lab list-dataset-downloaders
torq-lab download-dataset <downloader_name> <downloader_parameters> <dataset_file>
torq-lab list-optimizers
torq-lab optimize <optimizer_name> <optimizer_parameters> <input_model_file> <output_model_file>
torq-lab list-importers
torq-lab import <importer_name> <importer_parameters> <input_model_file> <model.mlir>
torq-lab list-validators
torq-lab validate <validator_name> <validator_parameters> <input_model_file>
torq-lab list-models
torq-lab model build <model_name> --target <target_chip> <model.vmfb>
torq-lab model get <model_name> --target <target_chip> <model.vmfb>
torq-lab model verify <model_name> <model.vmfb>
```

We abolish the `torq-tools` package entirely; its code is shipped as
part of the `torq-compiler` package.

The code of `torq-tools` is split in two parts: 

- Plugin interfaces
- Implementation of the tools themselves

The plugin interfaces are included in all installations of the 
`torq-compiler` package, ensuring that the necessary hooks for extending
the toolset are always available. This package also contains
the implementation of generic plugins. Existing ML framework
extras provide additional plugins relevant to those frameworks.

We add a new extra `torq-compiler[tools]` to allow users to 
install the additional plugins previously provided by the `torq-tools` 
package. The additional dependencies are pulled in only when 
the extra is specified.

We package `torq-examples` as a separate pure python package that can run
both on target and host environments. This package provides validation
plugins for different model types that leverage the inference code they
contain.

The plugins are divided as follows:

- `torq-compiler`: General plugins that do not depend on specific task or model
- `torq-tools`: Plugins that provide downloaders/optimizers/validators that work
  on specific models and produce results in original ML frameworks formats 
  (ONNX, ...). Plugins that provide models.
- `torq-examples`: Plugins that provide dataset downloaders and validators
  that use real data and task-specific criteria (e.g. accuracy) to validate
  models.

A typical user installs the `torq-compiler` package on the host environment,
which provides the `torq-lab` and generic compiler tools. It may 
install `torq-compiler[tools]` that includes additional tools specific 
to the model. They also may install `torq-runtime` and `torq-examples` for
the host if they want to validate models with emulation on the host environment.
On the target, they install `torq-runtime` and optionally `torq-examples` 
if they want to run and validate a specific model on the target environment.

A developer of `torq-tools` that does not need to develop the compiler
will install pre-built wheels and will override the python search path 
to load the implementation of the tools from their local `torq-tools` 
repository checkout.

A developer of `torq-examples` will check out the `torq-examples` 
repository and use pre-built wheels for the other packages.

A developer of `torq-compiler` will work on the `torq-compiler` repository.
The sources for `torq-tools` and `torq-examples` are available in this
environment as git submodules.

Testing of the `torq-tools`, `torq-compiler`, and `torq-examples` packages 
is done by building all the packages from a common tree. The test framework
can list all available model plugins and call their validation routines.

The release process uses the models plugin to generate a list of officially
supported models, builds them and then uploads them to Hugging Face.

A user or third-party developer that wants to provide a new plugin
will implement it in their own separate repository and package, installing
pre-built wheels for the other packages.

The plugins may use caching where appropriate to improve performance 
by avoiding redundant computations and downloads.

## Alternatives

The `torq-tools` plug-ins could be packaged as a standalone package, but this
increases the number of packages we need to deliver. Embedding the `torq-tools` 
plug-ins directly doesn't significantly increase the wheel size.

The `torq-examples` package could be split into two parts, inferencing code and
a library of reusable components and utilities for model inferencing. This
repository, however, is already used as a library by the demos, so the additional
split might not provide significant benefits and would complicate the
development and maintenance process.

We could avoid adding a plugin interface, but this would limit the extensibility 
of the system for users and third parties, making the `torq-lab` tool less useful
for them.

We could also avoid merging everything under the `torq-lab` umbrella, but this
would make the discovery of components much more complex. Some level of dynamic 
discovery will still be necessary for testing.

## Consequences

### Positive

- Users need to deal with a single `torq-lab` entry point
- Fewer packages to deliver
- Clear workflows for standalone `torq-examples` and `torq-tools` development
- Consistent validation of models and examples
- Third parties can extend the tooling
- Automated release of officially supported models
- Ensure synchronization of releases and examples
- Reduce code duplication in tests
  
### Negative

- Increased complexity of `torq-lab` due to dynamic plugin management.
- All plugins must work in a single virtual environment so they need to have 
  compatible versions
- Risks of interface mismatches. **Mitigation**: we release all packages
  in lockstep. We include a plugin compatibility check.
- Stale cache risks. **Mitigation**: CI clears cache at critical points in the workflow.

