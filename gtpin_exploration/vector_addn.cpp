#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#define N 1024

std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_int err;

    //debugging to force assign intel platform
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enumerate OpenCL platforms.\n";
        return 1;
    }

    for (cl_platform_id p : platforms) {
        char pname[256] = {};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);

        if (std::string(pname).find("Intel") == std::string::npos)
            continue;

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0)
            continue;

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS)
            continue;

        platform = p;
        device = devices[0];
        break;
    }

    if (!platform || !device) {
        std::cerr << "Failed to find an Intel GPU OpenCL device.\n";
        return 1;
    }

    char pname[256] = {};
    char dname[256] = {};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);

    std::cout << "Selected platform: " << pname << '\n';
    std::cout << "Selected device  : " << dname << '\n';

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);

    //execution logic
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 0.0f);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, nullptr);

    std::string kernelSrc = loadKernel("vector_addn.cl");
    const char* src = kernelSrc.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    
    int n = N;
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t globalSize = N;

    for(int i = 0; i < 10; i++){
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    }

    clFinish(queue);
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * N, C.data(), 0, nullptr, nullptr);
    std::cout << "C[0] = " << C[0] << std::endl;

    return 0;
}