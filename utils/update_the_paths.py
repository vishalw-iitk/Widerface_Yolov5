def update_to_running_paths_with_pretrianed(running_model_paths, pre_trained_model_paths, skip_train, skip_QAT_train, skip_Pruning, skip_P1_training, skip_P2_training, skip_P3_training, skip_P4_training):
    if skip_train:
        running_model_paths['Regular']['Pytorch']['fp32'] = pre_trained_model_paths['Regular']['Pytorch']['fp32']
        running_model_paths['Regular']['Pytorch']['fp16'] = pre_trained_model_paths['Regular']['Pytorch']['fp16']
        running_model_paths['Regular']['Tflite']['fp32'] = pre_trained_model_paths['Regular']['Tflite']['fp32']
    
    if skip_QAT_train:
        running_model_paths['Quantization']['Pytorch']['QAT'] = pre_trained_model_paths['Quantization']['Pytorch']['QAT']
    
    if skip_Pruning:
        running_model_paths['Pruning']['Pytorch']['P1'] = pre_trained_model_paths['Pruning']['Pytorch']['P1']
        running_model_paths['Pruning']['Pytorch']['P2'] = pre_trained_model_paths['Pruning']['Pytorch']['P2']
        running_model_paths['Pruning']['Pytorch']['P3'] = pre_trained_model_paths['Pruning']['Pytorch']['P3']
        running_model_paths['Pruning']['Pytorch']['P4'] = pre_trained_model_paths['Pruning']['Pytorch']['P4']
    else:
        if skip_P1_training:
            running_model_paths['Pruning']['Pytorch']['P1'] = pre_trained_model_paths['Pruning']['Pytorch']['P1']
        if skip_P2_training:
            running_model_paths['Pruning']['Pytorch']['P2'] = pre_trained_model_paths['Pruning']['Pytorch']['P2']
        if skip_P3_training:
            running_model_paths['Pruning']['Pytorch']['P3'] = pre_trained_model_paths['Pruning']['Pytorch']['P3']
        if skip_P4_training:
            running_model_paths['Pruning']['Pytorch']['P4'] = pre_trained_model_paths['Pruning']['Pytorch']['P4']

    return running_model_paths
