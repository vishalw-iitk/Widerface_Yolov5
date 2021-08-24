import numpy as np
import matplotlib.pyplot as plt
import os

def none_to_zero(lst):
    """
        Replace all None values in the list with zeros
    """
    array = np.array(lst)
    array[array==None]=0
    return array


def plot_each_metric(metric_list,names_list,metric_name,nums,save_dir):
    """
    plot each metric to a seperate file in save_dir directory
    """
    fig = plt.figure(figsize = (10, 5))
    # creating the bar plot
    plt.bar(range(len(names_list)), metric_list, color =['green']*nums[0]+['blue']*nums[1]+['cyan']*nums[2],width = 0.4)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.xticks(range(len(names_list)),names_list)
    plt.xlabel("Technique")
    plt.ylabel("Value of "+metric_name)
    plt.title(metric_name+" with different techniques")
    plt.savefig(os.path.join(save_dir,metric_name+'.png'),bbox_inches = 'tight')


def run(plot_results,save_dir):
    """
    extract metrics from dictionary and plot to seperate files
    """
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    mAP50_list = []
    mAP_list = []
    fitness_list = []
    latency_list = []
    GFLOPS_list = []
    size_list = []
    names_list = []
    metric_names = ['mAP50','mAP','fitness','latency','GFLOPS','size']
    #nums to track number of methods/platforms under each optimization technique
    nums = []
    #iterate through the dictionary
    for technique, platform_results in plot_results.items():
        name = ''
        i = 0
        for platform,x in platform_results.items():
            for bit, res_dict in x.items():
                i += 1
                name = technique+'_'+platform+'_'+bit
                names_list.append(name)
                mAP50_list.append(res_dict.get('mAP50'))            
                mAP_list.append(res_dict.get('mAP'))
                fitness_list.append(res_dict.get('fitness'))
                latency_list.append(res_dict.get('latency')[1])
                GFLOPS_list.append(res_dict.get('GFLOPS'))
                size_list.append(res_dict.get('size'))
        nums.append(i)
    mAP50_list = none_to_zero(mAP50_list)
    mAP_list = none_to_zero(mAP_list)
    fitness_list = none_to_zero(fitness_list)
    latency_list = none_to_zero(latency_list)
    GFLOPS_list = none_to_zero(GFLOPS_list)
    size_list = none_to_zero(size_list)
    plot_each_metric(mAP50_list,names_list,'mAP50',nums,save_dir)
    plot_each_metric(mAP_list,names_list,'mAP',nums,save_dir)
    plot_each_metric(fitness_list,names_list,'fitness',nums,save_dir)
    plot_each_metric(latency_list,names_list,'Latency',nums,save_dir)
    plot_each_metric(GFLOPS_list,names_list,'GFLOPS',nums,save_dir)
    plot_each_metric(size_list,names_list,'size',nums,save_dir) 