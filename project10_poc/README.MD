# Execution Procedures

-Requires a locally built openVINO repository

-TBB binaries must be added to the environment path variable 

-Compilation should include path to the OpenVINO includes and library files (given below)  

-Execution requires the dynamic linked libraries in this folder  

# Compilation Command

cl /std:c++17 gpu_plugin_poc.cpp /I "openvino\src\inference\include" /I "openvino\src\core\include" /link /LIBPATH:"openvino\bin\intel64\Release" openvino.lib

cl /std:c++17 property_verification.cpp /I "W:\Building\gsoc_prep\openvino\src\inference\include" /I "W:\Building\gsoc_prep\openvino\src\core\include" /link /LIBPATH:"W:\Building\gsoc_prep\openvino\bin\intel64\Release" openvino.lib