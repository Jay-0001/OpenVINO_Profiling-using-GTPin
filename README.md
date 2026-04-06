# Aim

To analyze the existing GPU execution flow, evaluate the feasibility of introducing a  `enable_gtpin` flag and interpret GTPin profiling data. This pre-requisite task demonstrates understanding of the OpenVINO GPU side and relevant technologies required for the `OpenVINO Profiling using VTune & GTPin` project.


# Understanding the GPU Execution Flow

The GPU execution flow can be broadly summarised as:

Model Compile Time -

ov::Core::compile_model() ->  ov::intel_gpu::Plugin::compile_model()  -> ExecutionConfig -> ov::intel_gpu::CompiledModel  -> Graph Construction

Inference time -

SyncInferRequest::infer()  -> network::execute() -> primitive_inst::execute()  -> primitive_impl::execute()  -> primitive_impl::execute()  -> stream.enqueue_kernel() -> OpenCL kernel launch 

The `Plugin::compile_model()` carries the user arguments from `ov::Core` and is the key area where propertiesa are registered and transferred to the CompiledModel, subsequently to be accessed during inference time. 

The `SyncInferRequest::infer()` is the highest level boundary that wraps all kernels launched during a single inference request.   

# Interpreting GTPin Profiling data

I have used the `funtime` sample tool offered by `GTPin` to analyse the profiling output of a simple vector addition openCL kernel and the output of the `benchmark_app` for the ResNet50 model. The corresponding funtime reports are stored in [funtime_vect.txt](gtpin_exploration/gtpin_output/GTPIN_PROFILE_FUNTIME3/Session_Final/funtime_vect.txt) and [funtime_res.txt](gtpin_exploration/gtpin_output/GTPIN_PROFILE_FUNTIME1/Session_Final/funtime_res.txt)  This allowed me to understand the runtime behavior of GTPin and the limitations of the funtime sample tool. 

For the same ResNet50 workload, I obtained the detailed [performance counter report](resnet_pc_report.txt) using the `benchmark_app` . I have also obtained a [VTune report](vtune_resnet_report.txt) and analysed these reports to find relationships between all kernel level metrics, layer level execution and hardware statistics. 

I have observed that GTPin does not operate at the individual kernel launch level by default, it's profiling output consists of all kernels launched by the process. It does not associate kernel executions with respective inferences. We'll have to enforce per-inference mapping to derive useful insights from the profiling data.

Through this analysis, I've updated my earlier mental model of attempting to starting/stopping GTPin within the `sync_infer_request::infer()` and shifted towards finding correlation between different levels.

# Proposed Implementation Strategy

Upon observing the execution flow and noticing similar functionalities, I propose to implement the objectives in the following manner:  

### **1) Propagation of `enable_gtpin` flag**

**Location – `src/plugins/intel_gpu/src/plugin.cpp`**

- The `enable_gtpin` flag can be introduced as a GPU plugin property by defining it in `internal_properties.hpp` (can be added as a user level property in the actual implementation), and registering it with `ExecutionConfig.hpp` by adding the property to `options.inl`. This allows the enable_gtpin property to be accessed at runtime.
- The `enable_gtpin` flag flows from the user input to `ov::compile_model` function and reaches `Plugin::compile_model` through the `AnyMap` parameter. Here, the property is parsed and stored within the `ExecutionConfig`. 
- This `ExecutionConfig` is then stored in the `CompiledModel` object and propagates into the `Graph` during graph construction.  
- Finally, the flag can be accessed inside `sync_infer_request` during inference time through the `ExecutionConfig` carried within the `Graph` object. 


### **2) Inference boundary and correlation**

**Location – `src/plugins/intel_gpu/src/sync_infer_request.cpp`**
- As observed earlier, GTPin operates at the process level and collects metrics for all GPU kernels without segregating them at the inference level. While we can attempt to use filter arguments and other feasible methods to globally isolate specific kernels, inference level mapping requires stronger correlation analysis.
- `SyncInferRequest::infer()` is the ideal location to introduce correlation into GTPin profiling data, since it serves as a logical boundary that encapsulates all kernel launches associateed with a particular inference. 
- Additionally, within this boundary the `enable_gtpin` flag can be used at inference time to enable/disable correlation by adding some inference specific metadata without modifying kernel execution.
- This stage focuses on  adding context instead of directly controlling GTPin profiling itself. The final correlation needs data from gtpin, benchmark_app performance counters, kernel metadata to construct mapping across different levels.  
  
# Identified Limitations
The process scope of GTPin, and the lack of default association between kernels and corresponding inference requests.

The funtime sample tool simple outputs the GPU cycles taken by each kernel launched by the process, we require a custom GTPin tool to better evaluate individual kernels and identify hotspots.

The first objective primarily concentrates on the sync_infer_request, I need to further explore the GPU execution flow for asynchronous infer requests.  

# Next Steps

I'd like to deep dive into the GTPin documents to further understand how to create a custom GTPin tool for the purposes of this project. 

I understand that correlating kernels across different granularities is the primary motive of this project. So I plan to analyse the JIT mechanisms involved in kernel dispatch, kernel metadata and read upon the Intel Instrumentation Technology used throughout OpenVINO. 

# Conclusion
 In this pre-requisite task, I have verified that the `enable_gtpin` can be propagated as a property during compile time and explored potential ways to correlate GTPin profiling data. Through my exploration and implementation efforts, I have familiarised myself with the GPU execution flow in OpenVINO, understood GTPin instrumentation and the significance of correlation in this project.
