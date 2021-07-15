Select from here(https://pytorch.org/) the command for installing pytorch based on the CUDA version you have and put the command inside 'gpu_rep.py' under the variable 'CUDA_compatible_command'

Uncomment skip = True inside 'gpu_req.py' to installing the dependencies for GPU for the first time
Uncomment skip = False inside 'gpu_req.py' to stop re-installing the dependencies for GPU

On Linux terminal :
    CPU Run : bash main_Cpu.sh
    GPU Run : bash main_Gpu.sh


