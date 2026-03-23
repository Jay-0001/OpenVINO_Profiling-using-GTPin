# Execution Procedures

Requires a locally built openVINO repository  
Compilation should include path to the includes and library files  
Execution requires the dynamic linked libraries in this folder  

# Compilation Command

cl /std:c++17 gpu_plugin_poc.cpp /I "openvino\src\inference\include" /I "openvino\src\core\include" /link /LIBPATH:"openvino\bin\intel64\Release" openvino.lib