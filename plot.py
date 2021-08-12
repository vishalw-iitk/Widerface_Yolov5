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
                    'size' : 1,
                    'sparsity' : 30
                },
                'P2' : {
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