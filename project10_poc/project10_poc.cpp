//What is the clang on/off comments that I run into?
#include "openvino/openvino.hpp"
#include <iostream>
#include <string>

class CoreWrapperGTPin{
    private:
        ov::Core runtime;
        bool enable_gtpin;

    public:
        CoreWrapperGTPin(bool gtpin_flag){
            enable_gtpin=gtpin_flag;
        }

        std::shared_ptr<ov::Model> read_model(const std::string& model_path){
            return runtime.read_model(model_path);
        }

        ov::CompiledModel compile_model(std::shared_ptr<ov::Model> loaded_model, const std::string& device){
            if(enable_gtpin){
                //Establishes the profiling content to look for when the actual profiling happens in infer_request
                //GTPin tool injects performance measurement instructions into the kernel at compilation
                std::cout<<"First Hook: compile_model intercepted & GTPin enabled"<<std::endl;
            }
            //Reusing the original implementation once the flag is set
            return runtime.compile_model(loaded_model,device);
        }

        ////Helper to transfer the flag information to the inference request
        bool is_gtpin_enabled() const {
            return enable_gtpin;
        }
};

//Which exact files must these be placed in? Under the GPU plugin
void start_gtpin_profiling() {
    std::cout<<"starting GTPin profiling\n";
    /*GTPin Funtime tool usage here*/
}

void stop_gtpin_profiling() {
    /*Storing the results observed by the Funtime tool*/
    std::cout<<"stopping GTPin profiling\n";
}

class InferRequestWithGTPin {

    private:
        ov::InferRequest infer_request;
        bool enable_gtpin;
    
    public:
        InferRequestWithGTPin(ov::InferRequest request, bool enable_flg){
            std::cout<<"Second Hook: creating a wrapper around ov::InferRequest\n";
            infer_request=request; 
            enable_gtpin=enable_flg;
        }

        void infer() {
            if(enable_gtpin){
                start_gtpin_profiling();
            }

            std::cout<<"calling ov::InferRequest::infer()\n";
            infer_request.infer();

            if(enable_gtpin){
                stop_gtpin_profiling();
            }
        }
};



int main(int argc, char** argv){
    std::string def_model_path="W:/Building/gsoc_prep/openvino/build/models/public/resnet-50-pytorch/FP16/resnet-50-pytorch.xml";
    std::string device="GPU";
    bool enable_gtpin=true;

    if(argc==3){
        def_model_path=argv[1];
        enable_gtpin=(std::string(argv[2])=="true");
    }

    try{
        //Initialize a OpenVINO runtime
        CoreWrapperGTPin runtime(enable_gtpin);
        std::cout<<"Initialized OpenVINO runtime"<<std::endl;

        //Load the given pre-trained model
        std::shared_ptr<ov::Model> ir_model = runtime.read_model(def_model_path);
         std::cout<<"Successfully loaded Pre-trained model"<<std::endl;

        //Compile the loaded model into a device specific format (primitive graph)
        ov::CompiledModel gpu_model = runtime.compile_model(ir_model,device);

        //Create a inference request with inputs
        ov::InferRequest inference_request = gpu_model.create_infer_request();

        //Hook for GTPin profiling
        //Wrapping the inference request object
        InferRequestWithGTPin wrapped_request(inference_request, runtime.is_gtpin_enabled());

        //Run inference 
        wrapped_request.infer();

        std::cout<<"End of Inference"<<std::endl;
    }
    catch(std::exception &exc){
        //Should i be using slog instead?
        std::cout<<"Error: "<<exc.what()<<std::endl;
        return 1;
    }
    return 0;
}