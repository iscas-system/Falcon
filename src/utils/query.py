import os,sys,platform
import csv
import math
import numpy as np

gputype_list = ['M60','K80','T4','V100']
current_dir = os.path.abspath('.')
all_m = ['bert-large','densenet-201','gru','inception-v2','inception-v4','mobilenet-v2','resnet-101','resnet-152-v2','roberta','tacotron2','transformer','vgg16']
all_o = ['add','batch_norm','concat','conv1d','conv2d','dense','multiply','relu','sigmoid','split','strided_slice','tanh','transpose']
baselines = ['morphling', 'vesta', 'heterbo', 'ernest']

def query_model_data(m,batchsize,cpus,gpumem,gpupower,gputype):
    retv = []
    gputype = gputype_list[gputype-1]
    source_directory_path = os.path.join(current_dir,"../data/models/sources")
    if platform.system() == "Windows":
        target_directory_path = os.path.join(current_dir,"../data/models/targets")
    else:
        target_directory_path = os.path.join(current_dir,"/home/falcon/data/models/targets")
    contents = os.listdir(target_directory_path)
    files = filter(lambda f: os.path.isfile(os.path.join(target_directory_path,f)),contents)
    for file in files:
        if m in file:
            file = os.path.join(target_directory_path, file)
            with open(file, 'r') as fr:
                f_csv = csv.reader(fr)
                headers = next(f_csv)
                for row in f_csv:
                    if set([str(batchsize),str(cpus),str(gpumem),str(gpupower),gputype]).issubset(set(row)):
                        retv = row
        else:
            continue
    return retv

def query_op_data(m,op,file_number, para1, para2):
    retv = []
    if platform.system() == "Windows":
        target_directory_path = None
        target_dir_parent = "../data/models/model-operator/" + m
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if op in file:
                    target_directory_path = os.path.join(target_dir_parent,file)
                    break
                
    else:
        target_directory_path = None
        target_dir_parent = "/home/falcon/data/models/model-operator/%s/" % (m)
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if op in file:
                    target_directory_path = os.path.join(target_dir_parent,file)
                    break   
#         target_directory_path = os.path.join(current_dir,"../data/models/model-operator/" + m + "/" + op)
#     else:
#         target_directory_path = os.path.join(current_dir,"/home/falcon/data/models/model-operator/op" + m + "/" + op)
    contents = os.listdir(target_directory_path)

    files = filter(lambda f: os.path.isfile(os.path.join(target_directory_path,f)),contents)
    i = 0
    for file in files:
        if i != file_number:
            i += 1
            continue
        i += 1
        file = os.path.join(target_directory_path, file)
        with open(file, 'r') as fr:
            f_csv = csv.reader(fr)

            for row in f_csv:
                temp = [float(i) for i in row]

                if para1 == temp[0] and para2 == temp[1]:
                    retv = row
    return retv

def query_op_data1(m,op,file_number, para1, para2):
    retv = []
    if platform.system() == "Windows":
        target_directory_path = None
        target_dir_parent = "../../data/models/model-operator/" + m
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if op in file:
                    target_directory_path = os.path.join(target_dir_parent,file)
                    break
                
    else:
        target_directory_path = None
        target_dir_parent = "/home/falcon/data/models/model-operator/%s/" % (m)
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if op in file:
                    target_directory_path = os.path.join(target_dir_parent,file)
                    break   
#         target_directory_path = os.path.join(current_dir,"../data/models/model-operator/" + m + "/" + op)
#     else:
#         target_directory_path = os.path.join(current_dir,"/home/falcon/data/models/model-operator/op" + m + "/" + op)
    contents = os.listdir(target_directory_path)

    files = filter(lambda f: os.path.isfile(os.path.join(target_directory_path,f)),contents)
    i = 0
    for file in files:
        if i != file_number:
            i += 1
            continue
        i += 1
        file = os.path.join(target_directory_path, file)
        with open(file, 'r') as fr:
            f_csv = csv.reader(fr)

            for row in f_csv:
                temp = [float(i) for i in row]

                if para1 == temp[0] and para2 == temp[1]:
                    retv = row
    return retv
                    
if __name__ == '__main__':
    print(query_model_data('bert-large', 64, 3, 1.2, 150, 2))
#     query_model_data('densenet')
#     print(u'\u2588\u2588'+' '+u'\u2588\u2588'+' '+u'\u2588\u2588'+' '+u'\u2588\u2588' + '  43%')
