# Proof of Concept – GTPin Hooks in OpenVINO GPU Execution Flow

## Overview
Based on the given objectives, I explored how GTPin profiling can be integrated into OpenVINO’s GPU inference workflow with minimal interference to the runtime. The goal is to demonstrate where profiling hooks should be introduced, how profiling can be enabled without modifying the public OpenVINO API, and how GTPin profiling can be triggered around GPU kernel launch & execution.

The PoC mirrors the internal execution path used by the GPU plugin and demonstrates how profiling hooks align with the plugin’s execution flow. A minimal program showcasing the proposed flow is implemented in `project10_poc.cpp`.

## OpenVINO GPU Inference Execution Flow

From the sample applications such as `benchmark_app` and `hello_classification`, the public runtime API can be summarized as:

`Core → read_model() -> compile_model() -> CompiledModel -> create_infer_request() -> infer()`

However, the public runtime API abstracts the device specific implementations of compile_model and infer stages. Upon further exploration of the GPU plugin, the actual  GPU execution flow can be summarised as:

`Core::compile_model() -> Plugin::compile_model() -> primitive graph -> CompiledModel -> create_infer_request() -> ov::InferRequest::infer() -> SyncInferRequest::infer() -> network::execute() -> primitive_inst::execute() -> stream -> GPU kernel launch`

From this flow, I have understood that:
- `compile_model()` transforms the IR device agnostic representation into GPU primitive graph and prepares kernels to beused during inference.
- The `infer()` method simply serves as a trigger for execution inside the GPU plugin, and is not the lowest level before kernel launch.
- Actual GPU kernel execution occurs at the primitive level, where each primitive instance of a graph  dispatches kernels through the GPU stream.

[Execution Flow Diagram](Images/runtime_flow.png)

## Proposed Implementation Strategy

My proposed strategy directly builds upon the given objectives and introduces two logical hooks aligned with the GPU execution flow.

- The first hook is placed at the `Plugin::compile_model()` stage. The enable_gtpin flag can be introduced as a configuration property into the plugin configuration map (Inspired from other properties in the Plugin.cpp). This property is passed by the program through the high level public API version of `ov::Core::compile_model()` into the GPU plugin.
  In the plugin, the `enable_gptin` is stored inside the execution configuration and can be accessed during graph creation. This positioning aligns with the GTPin usage, since it allows GTPin to register kernels and inject performance measurement instructions into the kernels before execution begins.

- The second hook should be placed at the exact boundary before and after the GPU kernels launch, to precisely measure kernel specific metrics. My initial profiling boundary suggestion would be `SyncInferRequest::infer()`. However, I realise that the actual kernel execution happens deeper inside the GPU plugin at the level of `primitive_inst::execute()`.

 My inital implementation ideas within the repository would be extending the GPU plugin configuration in `plugin.cpp` to accept the `enable_gtpin` property. Then propagating it through `ExecutionConfig` during model_compile and allow infer_request to access this property. This can allow the GTPin profiling around the exact kernel launch boundary only when the `enable_gtpin` is set.
 
## Proposed Profiling Execution Flow

With the proposed GTPin integration, the execution flow becomes:

`Runtime → compile_model() → propagate ENABLE_GTPIN_PROFILING → primitive graph build + kernel instrumentation → CompiledModel → create_infer_request() → infer() → primitive execution → start profiling → kernel dispatch → stop profiling`

[Proposed Profiling Flow](Images/proposed_flow.png)

## Proof of Concept Implementation
I've come up with a simple PoC simulation using wrapper classes that mirror the high-level runtime execution flow. I have 

A **Core wrapper** intercepts model compilation through `CoreWrapperGTPin::compile_model()` and simulates propagation of a profiling flag into the execution flow. The execution then follows the normal compile path.

An **InferRequest wrapper** builds upon the behavior of `SyncInferRequest::infer()` and bounds the execution window with profiling control logic.

This PoC operates at the high level runtime level and simulates hook placement. In the actual implementation, the second hook would be placed deeper in the GPU plugin execution path, specifically around `primitive_inst::execute()` where GPU kernels are dispatched.

## Proof of Concept Output

The output below verifies that the proposed conceptual hooks execute in the expected order without interrupting the existing execution flow.

[Proof of Concept Output](Images/poc_output.png)

## Conclusion

This proof of concept suggests that GTPin profiling can be integrated into OpenVINO’s GPU inference workflow using two hooks aligned with the GPU plugin execution flow.

The `compile_model()` hook enables kernel instrumentation through a propagated configuration property, while the execution hook isolates the kernel launch boundary within the GPU plugin.

From preliminary analysis of the high level runtime, this approach maintains the original execution flow and  aligns well with the existing flow. 