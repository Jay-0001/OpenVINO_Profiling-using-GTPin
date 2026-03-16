# Proof of Concept – GTPin Hooks in OpenVINO GPU Execution Flow

## Overview
Based on the given suggestions, this proof of concept explores how GTPin profiling can be integrated into OpenVINO’s GPU inference workflow with minimal interference to the runtime. The goal is to demonstrate where profiling hooks should be introduced, how profiling can be enabled without modifying the public OpenVINO API, and how GTPin profiling can be triggered around GPU kernel launch & execution.

The PoC mirrors the internal execution path used by the GPU plugin and demonstrates how profiling hooks align with the plugin’s execution flow. A minimal program showcasing the proposed flow is implemented in `project10_poc.cpp`.

## OpenVINO GPU Inference Execution Flow
After analyzing sample applications provided in the repository, mainly `benchmark_app` and `hello_classification`, I understood the public runtime API as:


Upon further exploring the GPU plugin implementation, I realized that the GPU inference lifecycle can be summarized as:

`Runtime initialization → ov::Core::compile_model() → GPU plugin builds primitive graph → ov::CompiledModel → create_infer_request() → ov::InferRequest::infer() → SyncInferRequest::infer() → Graph::execute() → GPU kernels executed`

In context of GPU execution flow, the `compile_model()` stage converts the IR representation of the pre-trained model into a GPU specific primitive graph and generates the kernels that will be executed during inference.

The `infer()` stage in the public runtime API triggers `SyncInferRequest::infer()` inside the GPU plugin, where the inference is actually implemented. The GPU inference implementation enqueues execution of the primitive graph and launches GPU kernels.

[Execution Flow Diagram](Images/runtime_flow.png)

(can be stronger)
## Proposed Implementation Strategy
My proposed strategy directly builds upon the suggestions provided and introduces two logical hooks aligned with the GPU execution flow.

The first hook is placed at the `Plugin::compile_model()` stage inside the. This hook uses an `enable_gtpin` flag to enable GTPin profiling and register the kernels to be profiled. During model compilation, the GPU plugin builds the primitive graph and generates kernels to be launched during inference. Therefore, introducing the `enable_gtpin` flag t this stage allows GTPin to register the kernels and inject measurement instructions into the kernels.

The second hook is placed at the `SyncInferRequest::infer()` stage inside the . The purpose of this hook is to isolate kernel execution from other runtime overhead and allow precise performance analysis using the GTPin Funtime sample tool. This ensures profiling occurs only during the inference window.

## Proposed Profiling Execution Flow
With the proposed GTPin integration, the runtime execution flow becomes:

`Runtime → compile_model() → GTPin kernel instrumentation → CompiledModel → create_infer_request() → infer() → start profiling → Graph::execute() → GPU kernels → stop profiling`

[Proposed Profiling Flow](Images/proposed_flow.png)

This approach ensures that kernels are registered during compilation and profiling is limited to the actual inference window within the GPU plugin's .

## Proof of Concept Implementation
I've come up with a simple PoC implementation using wrapper classes that mirror the GPU plugin execution path.

A **Core wrapper** intercepts model compilation through `CoreWrapperGTPin::compile_model()` and sets a `enable_gtpin` flag that initializes the GTPin profiling. The execution then follows the normal compile path.

An **InferRequest wrapper** builds upon the behavior of `SyncInferRequest::infer()` and tightly bounds the execution window with profiling control logic. The GTPin Funtime sample tool will be used in the functions surrounding the kernel execution window.

## Mapping to existing OpenVINO Source Code
During repository exploration I identified the public runtime implementations located in:

- `src/inference/src/cpp/core.cpp`
- `src/inference/src/cpp/compiled_model.cpp`
- `src/inference/src/cpp/infer_request.cpp`

However, the actual GPU specific execution path is implemented in the GPU plugin layer:

- `src/plugins/intel_gpu/src/plugin/plugin.cpp`
- `src/plugins/intel_gpu/src/plugin/compiled_model.cpp`
- `src/plugins/intel_gpu/src/plugin/sync_infer_request.cpp`
- `src/plugins/intel_gpu/src/plugin/graph.cpp`

Understanding this mapping allowed me to identify finer scoped hook points instead of stopping at the public runtime API abstraction level.


## Proof of Concept Output
The output below verifies that the proposed conceptual hooks execute in the expected order without interrupting the existing execution flow.

[Proof of Concept Output](Images/poc_output.png)

## Conclusion
My proof of concept verifies that GTPin profiling can be integrated into OpenVINO’s GPU inference workflow using two hooks located at the GPU specific `compile_model()` and `infer()` stages without affecting the existing execution flow.

The `compile_model()` hook registers kernels with GTPin and injects measurement instructions during model compilation. And the `infer()` hook creates a tightly bound profiling window around GPU kernel execution.