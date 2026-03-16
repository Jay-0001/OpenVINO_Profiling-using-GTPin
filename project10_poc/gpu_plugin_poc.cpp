#include <openvino/openvino.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

class CoreWrapperGTPin{
    private:
        bool enable_gtpin;
        ov::Core core;
        std::shared_ptr<ov::Model> model;
        ov::CompiledModel compiled_model;

    public:
        CoreWrapperGTPin(bool gtpin_flag){
            enable_gtpin=gtpin_flag;
        }

        std::shared_ptr<ov::Model> read_model(const std::string& model_path){
            std::cout<<"Successfully loaded Pre-trained model"<<std::endl;
            model=core.read_model(model_path);
            return model;
        }

        ov::CompiledModel compile_model(std::shared_ptr<ov::Model> loaded_model,const std::string& device){

            if(enable_gtpin){
                //Establishes the profiling context before kernels are generated
                std::cout<<"First Hook: compile_model intercepted & GTPin enabled"<<std::endl;
            }

            compiled_model=core.compile_model(loaded_model,device);
            std::cout<<"GPU primitive graph built"<<std::endl;
            return compiled_model;
        }

        bool is_gtpin_enabled() const{
            return enable_gtpin;
        }
};

void start_gtpin_profiling(){
    std::cout<<"starting GTPin profiling\n";
}

void stop_gtpin_profiling(){
    std::cout<<"stopping GTPin profiling\n";
}

class InferRequestGTPin{
    private:
        ov::InferRequest request;
        bool enable_gtpin;

    public:
        InferRequestGTPin(ov::CompiledModel model,bool enable_flg){
            std::cout<<"Second Hook: creating a wrapper around ov::InferRequest\n";
            request=model.create_infer_request();
            enable_gtpin=enable_flg;
        }

        void prepare_input(){
            ov::Tensor input_tensor=request.get_input_tensor();
            float* data=input_tensor.data<float>();
            size_t tensor_size=input_tensor.get_size();

            //Filling the tensor with random input as in benchmark_app
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution(0.0,1.0);

            for(size_t i=0;i<tensor_size;i++){
                data[i]=distribution(generator);
            }

            std::cout<<"Input tensor initialized with random values\n";
        }

        void infer(){
            if(enable_gtpin){
                start_gtpin_profiling();
            }
            //The original runtime infer
            std::cout<<"calling ov::InferRequest::infer()\n";
            request.infer();

            if(enable_gtpin){
                stop_gtpin_profiling();
            }
        }

        void verify_output(){
            ov::Tensor output_tensor=request.get_output_tensor();
            const float* output_data=output_tensor.data<const float>();
            size_t output_size=output_tensor.get_size();

            std::cout<<"Sample output values: ";
            //verification
            for(size_t i=0;i<3;i++){
                std::cout<<output_data[i]<<" ";
            }

            std::cout<<"\nOutput tensor size: "<<output_size<<std::endl;
        }
};

int main(int argc,char** argv){
    std::string def_model_path="../resnet-50-pytorch/FP16/resnet-50-pytorch.xml";
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
        std::shared_ptr<ov::Model> ir_model=runtime.read_model(def_model_path);

        //Compile the loaded model into a device specific format (primitive graph)
        ov::CompiledModel gpu_model=runtime.compile_model(ir_model,device);

        //Hook for GTPin profiling
        //Wrapping the inference request object
        InferRequestGTPin wrapped_request(gpu_model,runtime.is_gtpin_enabled());
        
        //Create dummy input
        wrapped_request.prepare_input();
        //Run inference 
        wrapped_request.infer();
        //Verify if inference output generated
        wrapped_request.verify_output();
        std::cout<<"End of Inference"<<std::endl;
    }

    catch(std::exception &exc){
        std::cout<<"Error: "<<exc.what()<<std::endl;
        return 1;
    }

    return 0;
}