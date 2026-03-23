Aim 
	To analyze the GPU execution flow and evaluate the feasibility of introducing a `enable_gtpin` flag to control GTPin profiling. And to identify an ideal kernel profiling boundary during inference time.


GPU Execution Flow
	The flow of control starting from the OpenVINO runtime to the kernel launch can be summarized as follows:

ov::Core::compile_model()
 → ov::intel_gpu::Plugin::compile_model()
 → ExecutionConfig
 → ov::intel_gpu::CompiledModel
 → Graph
 → SyncInferRequest::infer()
 → network::execute()
 → primitive_inst::execute()
 → primitive_impl::execute()
 → stream.enqueue_kernel()
 → OpenCL kernel launch

The Plugin::compile_model() and SyncInferRequest::infer() are the primary locations of interest with respect to the given objectives.


Proposed Implementation Strategy
	Upon observing the execution flow and noticing similar functionalities, I propose to implement the objectives in the following manner:

1) Propagation of `enable_gtpin` flag
Location – 
The `enable_gtpin` flag flows from the user input to the ov::compile_model function and eventually reaches the Plugin::compile_model through the AnyMap parameter. 
The flag can then be introduced as a GPU plugin property in the Plugin and stored in the `ExecutionConfig`, similar to the other properties inside the plugin.cpp file. 
The execution config is stored in the CompiledModel object and flows into Graph.cpp Finally the flag can be accessed inside sync_infer_request.cpp through the Graph object.


2)Profiling boundary hook 
Location – 
After examining the kernel launch at different levels, I have identified that the individual kernel launch occurs at `stream.enqueue_kernel`. However, this is not the ideal profiling boundary since GTPin instruments all kernels launched by an application at once. Setting up GTPin at individual kernel launches will add excessive overhead to the execution flow.

Since the profiling boundary should have access to the `enable_gtpin` flag, `sync_infer_request::infer()` would serve as the ideal location to introduce GTPin. where the funtime tool can capture details of all kernels launched by the application and identify individual kernel metrics.



Simple Runtime Implementation

A conceptual version of the proposed idea at the public runtime level is implemented in `simple_poc.cpp`. This program simulates the proposed execution flow as:

output.png


This still operates at the public compile_model and infer level. However the true wrappers will be implemented within the plugin.cpp and sync_infer_request.cpp


Constraints 
The proposed implementation does not support Asynchronous infer requests


Conclusion
Through this proposal I have identified the mechanism for introducing `enable_gtpin` and the ideal GTPin profiling boundary per inference. The proposed implementation aligns well with existing mechanisms and execution flow. 
