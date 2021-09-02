plot_results = {
        'Regular': {
            'Pytorch' : {
                'fp32' : {
                    'mAP50' : 59.15,
                    'mAP' : 30.3,
                    'fitness' : 33.19,
                    'latency' : 163.05,
                    'GFLOPS' : 6.89,
                    'size' : 27.2
                }
            },
            'Tflite' : {
                'fp32' : {
                    'mAP50' : 59.19,
                    'mAP' : 30.33,
                    'fitness' : 33.22,
                    'latency' : 168.8,
                    'GFLOPS' : 27.3,
                    'size' : 6.89,
                }
            }
        },
        'Quantization': {
            'Pytorch' : {
                'PTQ' : {
                    'mAP50' : 53.48,
                    'mAP' : 23.65,
                    'fitness' : 26.64,
                    'latency' : 140.62,
                    'GFLOPS' : None,
                    'size' : 7.25
                },
                'QAT' : {
                    'mAP50' : 54.24,
                    'mAP' : 24.76,
                    'fitness' : 27.7,
                    'latency' : 131.89,
                    'GFLOPS' : None,
                    'size' : 7.07
                }
                },
            'Tflite' : {
                'fp16' : {
                    'mAP50' : 59.19,
                    'mAP' : 30.33,
                    'fitness' : 33.22,
                    'latency' : 168.04,
                    'GFLOPS' : 6.9,
                    'size' : 13.7,
                },
                'int8' : {
                    'mAP50' : None,
                    'mAP' : None,
                    'fitness' : None,
                    'latency' : None,
                    'GFLOPS' : None,
                    'size' : None
                }
            }
        },
        'Pruning': {
            'Pytorch' : {
                'P1' : {
                    'mAP50' : 54,
                    'mAP' : 26.71,
                    'fitness' : 29.47,
                    'latency' : 162.29,
                    'GFLOPS' : 6.89,
                    'size' : 27.1
                },
                'P2' : {
                    'mAP50' : 55.43,
                    'mAP' : 27.43,
                    'fitness' : 30.22,
                    'latency' : 161.86,
                    'GFLOPS' : 27.1,
                    'size' : 6.89
                },
                'P4' : {
                    'mAP50' : 36.73,
                    'mAP' : 16.64,
                    'fitness' : 18.65,
                    'latency' : 162.41,
                    'GFLOPS' : 6.89,
                    'size' : 27.1
                }
            }
        }
    }