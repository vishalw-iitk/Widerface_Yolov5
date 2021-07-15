Dataset link :
    http://shuoyang1213.me/WIDERFACE/index.html


Train : 12880
Validation : 3226
Test : 16097

RAW_DATASET_FORMAT :
    RAW_DATASET
        train
            images
                images_folders_of_various_classes
            labels
                .mat and .txt file with img_names and labels
        validation
            images
                images_folders_of_various_classes
            labels
                labels in .mat and .txt file with img_names and labels
        test
            images
                images_folders_of_various_classes
            labels
                labels in .mat and .txt file with img_names ONLY



Select from here(https://pytorch.org/) the command for installing pytorch based on the CUDA version you have and put the command inside 'gpu_rep.py' under the variable 'CUDA_compatible_command'

Uncomment skip = True inside 'gpu_req.py' to installing the dependencies for GPU for the first time
Uncomment skip = False inside 'gpu_req.py' to stop re-installing the dependencies for GPU

On Linux terminal :
    CPU Run : bash main_Cpu.sh
    GPU Run : bash main_Gpu.sh


