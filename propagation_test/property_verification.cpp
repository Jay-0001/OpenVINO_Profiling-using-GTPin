#include <openvino/openvino.hpp>
#include <iostream>

int main() {
    try {
        // 1. Create core
        ov::Core core;

        // 2. Load model (use any small IR model you already have)
        std::string model_path = "../resnet-50-pytorch/FP16/resnet-50-pytorch.xml";  // adjust path

        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        // 3. Prepare properties
        ov::AnyMap props;
        props["GPU_ENABLE_GTPIN"] = true;

        std::cout << "[MAIN] enable_gtpin set to true" << std::endl;

        // 4. Compile model on GPU
        auto compiled_model = core.compile_model(model, "GPU", props);

        std::cout << "[MAIN] Model compiled" << std::endl;

        // 5. Create infer request
        auto infer_request = compiled_model.create_infer_request();

        std::cout << "[MAIN] Infer request created" << std::endl;

        // 6. Run inference (this should hit SyncInferRequest::infer)
        infer_request.infer();

        std::cout << "[MAIN] Inference complete" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}