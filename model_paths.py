'''
This file stores the relative paths of every model.
It is expected to maintain the RAW_DATASET and Pre_trained_model folders path wrt dts folder so as to run the code properly
'''

# Libraries
import os

def frameworks(skip_training, running_model_paths, pre_trained_model_paths):
    '''
    Stores the path of fp32 models of different frameworks. Required in the export model task.
    ARGS :
        skip_training : Updation based whether on pre-trained models are used or runs(training) models are used for export
        running_model_paths : Stores the path of model on which inference could be done
        pre_trained_model_paths : Stores the path of pre-trained-models
    RETURNS :
        framework_path : Updated framework paths as per the skip-training status
    '''

    if skip_training == False: #training
        framework_path = {
            'Pytorch':{
                'Regular_Pytorch': running_model_paths['Regular']['Pytorch']['fp32']
            },
            'ONNX':{
                'Regular_Pytorch': os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.onnx')
            },
            'tf_pb':{
                'Regular_Pytorch': os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.pb', 'saved_model.pb'),
                'Quantized_Tflite_fp16' : os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.pb', 'saved_model.pb'),
                'Quantized_Tflite_int8' : os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.pb', 'saved_model.pb')
            },
            'tflite':{
                'Regular_Pytorch': os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.tflite'),
                'Quantized_Tflite_fp16' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'fp16', 'best.tflite'),
                'Quantized_Tflite_int8' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'int8', 'best.tflite')
            }
        }
    else: # skip-training : means paths are to be updated with pre-trained model paths
        framework_path = {
            'Pytorch':{
                'Regular_Pytorch': pre_trained_model_paths['Regular']['Pytorch']['fp32']
            },
            'ONNX':{
                'Regular_Pytorch': os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.onnx')
            },
            'tf_pb':{
                'Regular_Pytorch': os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.pb', 'saved_model.pb'),
                'Quantized_Tflite_fp16' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.pb', 'saved_model.pb'),
                'Quantized_Tflite_int8' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.pb', 'saved_model.pb')
            },
            'tflite':{
                'Regular_Pytorch': os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.tflite'),
                'Quantized_Tflite_fp16' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'fp16', 'best.tflite'),
                'Quantized_Tflite_int8' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'int8', 'best.tflite')
            }
        }
    return framework_path

def running_model_dictionary():
    '''
    Stores the path of model on which inference could be done(training sometimes). It is maintained during the entire pipeline run.
    It could be runs/(training/trained models) or pre-trained models
    RETURNS :
        running_model_paths : Stores the path of model on which inference could be done
    '''

    running_model_paths = {
        'Regular': {
            'Pytorch' : {
                'fp32' : os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.pt'), #'path-if-uncloned- TimeTaker(store-expicitly)-get-fp16',
                'fp16' : os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp16', 'best.pt')#'path-cloned-Time-Taker(store-expicitly)-get-fp32None'
            },
            'Tflite' : {
                'fp32' : os.path.join('..', 'runs', 'Regular', 'Regular_Pytorch', 'weights', 'fp32', 'best.tflite') #'path-if-unclonde-getfp16',
            }
        },
        'Quantization': {
            'Pytorch' : {
                'PTQ' : os.path.join('..', 'Model_compression', 'Quantization', 'Pytorch', 'PTQ', 'best.pt'), #'path',
                'QAT' : os.path.join('..', 'runs', 'Quantization', 'QAT', 'Quantized_Pytorch_QAT', 'weights', 'best.pt') #'path-Time-Taker(store-expicitly)'
                },
            'Tflite' : {
                'fp16' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'fp16', 'best.tflite'),#'path',
                'int8' : os.path.join('..', 'Model_compression', 'Quantization', 'Tflite', 'int8', 'best.tflite') #'path'
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : os.path.join('..', 'runs', 'Pruning', 'P1', 'Pruned_Pytorch_P1', 'weights', 'best.pt'), #'path-Time-Taker(store-expicitly)',
                'P2' : os.path.join('..', 'runs', 'Pruning', 'P2', 'Pruned_Pytorch_P2', 'weights', 'best.pt'), #'path'
                'P3' : os.path.join('..', 'runs', 'Pruning', 'P3', 'Pruned_Pytorch_P3', 'weights', 'best.pt'), #'path-Time-Taker(store-expicitly)',
                'P4' : os.path.join('..', 'runs', 'Pruning', 'P4', 'Pruned_Pytorch_P4', 'weights', 'best.pt') #'path'
            },
            'Tflite' : {
                'P1' : os.path.join('..', 'Model_compression', 'Pruning', 'Tflite', 'P1', 'best.pt'), #'path',
                'P2' : os.path.join('..', 'Model_compression', 'Pruning', 'Tflite', 'P2', 'best.pt') #'path'
            }
        }
    }
    return running_model_paths

def pre_trained_model_dictionary():
    '''
    Stores the path of pre-trained model on which inference could be done(training sometimes).
    Pre-model model paths are useful specially when training process takes a lot time
    RETURNS :
        pre_trained_model_paths : Stored the paths of pre-trained models.
                                PTQ/exports are done anyhow quickly, so pre-trained for this same as the model \
                                stored in Model_Compression generated folder
    '''
    pre_trained_model_paths = {
        'Regular' : {
            'Pytorch' : {
                'fp32' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.pt'), #'path',
                'fp16' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp16', 'best.pt') #'path'
            },
            'Tflite' : {
                'fp32' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.tflite') #'path',
                }
        },
        'Quantization' : {
            'Pytorch' : {
                'QAT' : os.path.join('..', 'Pre_trained_model', 'Model_compression', 'Quantization', 'QAT', 'best.pt') #'path'
            }
        },
        'Pruning' : {
            'Pytorch' : {
                'P1' : os.path.join('..', 'Pre_trained_model', 'Model_compression', 'Pruning', 'P1', 'best.pt'), #'path'
                'P2' : os.path.join('..', 'Pre_trained_model', 'Model_compression', 'Pruning', 'P2', 'best.pt'),
                'P3' : os.path.join('..', 'Pre_trained_model', 'Model_compression', 'Pruning', 'P3', 'best.pt'),
                'P4' : os.path.join('..', 'Pre_trained_model', 'Model_compression', 'Pruning', 'P4', 'best.pt'),
                'theta0' : os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch', 'fp32', 'best.pt')
            }
        }
    }
    return pre_trained_model_paths

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


def prune_with_pre_trained_only(running_model_paths, pre_trained_model_paths):
    running_model_paths['Pruning']['Pytorch']['P1'] = pre_trained_model_paths['Pruning']['Pytorch']['P1']
    running_model_paths['Pruning']['Pytorch']['P2'] = pre_trained_model_paths['Pruning']['Pytorch']['P2']
    running_model_paths['Pruning']['Pytorch']['P3'] = pre_trained_model_paths['Pruning']['Pytorch']['P3']
    running_model_paths['Pruning']['Pytorch']['P4'] = pre_trained_model_paths['Pruning']['Pytorch']['P4']

    return running_model_paths

def train_results_dictionary():
    train_results_paths = {
        'Regular': {
            'Pytorch' : {
                'fp32' : os.path.join('..', 'runs', 'Regular')
            }
        },
        'Quantization': {
            'Pytorch' : {
                'QAT' : os.path.join('..', 'runs', 'Quantization', 'QAT')
                }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : os.path.join('..', 'runs', 'Pruning', 'P1'),
                'P2' : os.path.join('..', 'runs', 'Pruning', 'P2'),
                'P3' : os.path.join('..', 'runs', 'Pruning', 'P3'),
                'P4' : os.path.join('..', 'runs', 'Pruning', 'P4')               
            }
        }
    }
    return train_results_paths

def infer_results_dictionary():
    infer_paths = {
        'Regular': {
            'Pytorch' : {
                'fp32' : os.path.join('..', 'Model_performance', 'Inference_results', 'Regular', 'Pytorch', 'val')
            },
            'Tflite' : {
                'fp32' : os.path.join('..', 'Model_performance', 'Inference_results', 'Regular', 'Tflite', 'val')
            }
        },
        'Quantization': {
            'Pytorch' : {
                'PTQ' : os.path.join('..', 'Model_performance', 'Inference_results', 'Quantization', 'Pytorch', 'PTQ', 'val'),
                'QAT' : os.path.join('..', 'Model_performance', 'Inference_results', 'Quantization', 'Pytorch', 'QAT', 'val')
                },
            'Tflite' : {
                'fp16' : os.path.join('..', 'Model_performance', 'Inference_results', 'Quantization', 'Tflite', 'fp16', 'val'),
                'int8' : os.path.join('..', 'Model_performance', 'Inference_results', 'Quantization', 'Tflite', 'int8', 'val')
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Pytorch', 'P1', 'val'),
                'P2' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Pytorch', 'P2', 'val'),
                'P3' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Pytorch', 'P3', 'val'),
                'P4' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Pytorch', 'P4', 'val')
            },
            'Tflite' : {
                'P1' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Tflite', 'P1', 'val'),
                'P2' : os.path.join('..', 'Model_performance', 'Inference_results', 'Pruning', 'Tflite', 'P2', 'val')
            }
        }
    }
    return infer_paths

def test_results_dictionary():
    test_results_paths = {
        'Regular': {
            'Pytorch' : {
                'fp32' : os.path.join('..', 'Model_performance', 'TestData_results', 'Regular', 'Pytorch')
            },
            'Tflite' : {
                'fp32' : os.path.join('..', 'Model_performance', 'TestData_results', 'Regular', 'Tflite')
            }
        },
        'Quantization': {
            'Pytorch' : {
                'PTQ' : os.path.join('..', 'Model_performance', 'TestData_results', 'Quantization', 'Pytorch', 'PTQ'),
                'QAT' : os.path.join('..', 'Model_performance', 'TestData_results', 'Quantization', 'Pytorch', 'QAT')
                },
            'Tflite' : {
                'fp16' : os.path.join('..', 'Model_performance', 'TestData_results', 'Quantization', 'Tflite', 'fp16'),
                'int8' : os.path.join('..', 'Model_performance', 'TestData_results', 'Quantization', 'Tflite', 'int8')
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Pytorch', 'P1'),
                'P2' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Pytorch', 'P2'),
                'P3' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Pytorch', 'P3'),
                'P4' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Pytorch', 'P4')
            },
            'Tflite' : {
                'P1' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Tflite', 'P1'),
                'P2' : os.path.join('..', 'Model_performance', 'TestData_results', 'Pruning', 'Tflite', 'P2')
            }
        }
    }
    return test_results_paths

def model_defined_names():
    model_names = {
        'Regular': {
            'Pytorch' : {
                'fp32' : 'Regular_Pytorch'
            },
            'Tflite' : {
                'fp32' : 'Regular_Tflite_fp32'
            }
        },
        'Quantization': {
            'Pytorch' : {
                'PTQ' : 'Quantized_Pytorch_PTQ',
                'QAT' : 'Quantized_Pytorch_QAT'
                },
            'Tflite' : {
                'fp16' : 'Quantized_Tflite_fp16',
                'int8' : 'Quantized_Tflite_int8'
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : 'Pruned_Pytorch_P1',
                'P2' : 'Pruned_Pytorch_P2',
                'P3' : 'Pruned_Pytorch_P3',
                'P4' : 'Pruned_Pytorch_P4'
            },
            'Tflite' : {
                'P1' : 'Pruned_Tflite_P1',
                'P2' : 'Pruned_Tflite_P2'
            }
        }
    }
    return model_names

def plot_dictionary():
    plot_results = {
        'Regular': {
            'Pytorch' : {
                'fp32' : {
                    'mAP50' : 100,
                    'mAP' : 100,
                    'fitness' : None,
                    'latency' : 10,
                    'GFLOPS' : None,
                    'size' : 10
                }
            },
            'Tflite' : {
                'fp32' : {
                    'mAP50' : 100,
                    'mAP' : 100,
                    'fitness' : None,
                    'latency' : 10,
                    'GFLOPS' : None,
                    'size' : 10,
                }
            }
        }, # fp32 if fp32 not None else fp16
        'Quantization': {
            'Pytorch' : {
                'PTQ' : {
                    'mAP50' : 100,
                    'mAP' : 100,
                    'fitness' : None,
                    'latency' : 10,
                    'GFLOPS' : None,
                    'size' : 10
                },
                'QAT' : {
                    'mAP50' : None,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                }
                },
            'Tflite' : {
                'fp16' : {
                    'mAP50' : None,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                },
                'int8' : {
                    'mAP50' : None,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1
                }
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1
                },
                'P2' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                },
                'P3' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                },
                'P4' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                }
            },
            'Tflite' : {
                'P1' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                },
                'P2' : {
                    'mAP50' : 40,
                    'mAP' : 10,
                    'fitness' : 10,
                    'latency' : 10,
                    'GFLOPS' : 100,
                    'size' : 1,
                }
            }
        }
    }
##
