import sys
import csv
import mctsbo
import os,platform
import numpy as np
from utils.logger import set_logger

if platform.system() == "Windows":
    LOG = "D:/falcon.log"
else:
    LOG = "/var/log/falctl.log"
logger = set_logger(os.path.basename(__file__), LOG)

all_m = ['bert-large','densenet-201','gru','inception-v2','inception-v4','mobilenet-v2','resnet-101','resnet-152-v2','roberta','tacotron2','transformer','vgg16']
all_gputype = ['M60','K80','T4','V100']
current_dir = os.path.abspath('.')

def find_a_near_optimal_configuration(model):
    islast = False
    operator_list = []
    near_optimal_configuration = None
    if platform.system() == "Windows":
        path = "..\\data\\models\\model-operator\\" + model
    else:
        path = "/home/falcon/data/models/model-operator/%s" % model
#     print("file:", (path))
    for file in os.listdir(path):
        if file[-4:] != ".csv":
            file_split = file.split('-')
            if len(file_split) > 1:
                operator_list.append(file_split[1])
            else:
                operator_list.append(file)
    if platform.system() == "Windows":
        csv_path = os.path.join(current_dir,"../data/experiments/falcon-%s.csv" % model)
    else:
        csv_path = "/home/falcon/data/experiments/falcon-%s.csv" % model            
    with open(csv_path, 'w', newline='') as fr:
        writer = csv.writer(fr)
        writer.writerow(['model','iteration','batchsize','cpus','gpumem','gpupower','gputype','norm.(rps/budget)'])
    iteration = 0
    for op in operator_list:
        print('***Starting MCTS-BO for the %s operator in the %s model***' % (op, model))
        # print("model:", model)
        # print("operator:", op)
        # print("arg:", [model, op])
        # print("__________________")
        if op == operator_list[-1]:
            islast = True
        retv = mctsbo.main([model, op, islast, iteration])
        iteration = retv[2]
        if retv[0] > 0.9:
            next_point = retv[1]
            print('***Find a near-optimal configuration: %s (norm.(rps/budget)=%s)***' %([int(next_point[0]), int(next_point[1]),
                                                next_point[2], int(next_point[3]), 
                                                all_gputype[int(next_point[4])-1]],retv[0]))
            near_optimal_configuration = retv[1]
            return near_optimal_configuration
        else:
            continue
        
def _create_dir():
    if platform.system() == "Windows":
        if not os.path.exists('../data/experiments'):
            os.mkdir('../data/experiments')
    else:
        if not os.path.exists('/home/falcon/data/experiments'):
            os.mkdir('/home/falcon/data/experiments')

def output_results(model=None):
    output={}
    if platform.system() == "Windows":
        target_directory_path = os.path.join(current_dir,"../data/experiments")
    else:
        target_directory_path = '/home/falcon/data/experiments'
    contents = os.listdir(target_directory_path)
    files = filter(lambda f: os.path.isfile(os.path.join(target_directory_path,f)),contents)
    for file in files:
        rps_list = []
        row_num = 0
        if "falcon-%s" % (model) in file:
            file = os.path.join(target_directory_path, file)
            with open(file, 'r') as fr:
                f_csv = csv.reader(fr)
                headers = next(f_csv)
                for row in f_csv:
                    rps_list.append(float(row[-1]))
                    row_num += 1
            output[model] = {'iterations':row_num, 'mean.norm.(rps/budget)':np.mean(rps_list), 'max.norm.(rps/budget)':max(rps_list), 'min.norm.(rps/budget)':min(rps_list), 'std.norm.(rps/budget)':np.std(rps_list,ddof=1)}
    return output

def main():
    logger.debug('Running Falcon for the target DL models in /home/falcon/data/models/targets/')
    logger.debug('***Results are stored in CSV files in /home/falcon/data/experiments/***')
    print('Running Falcon for the target DL models in /home/falcon/data/models/targets/')
    print('***Results are stored in CSV files in /home/falcon/data/experiments/***')
    output = []
    if len(sys.argv) == 2:
        model = sys.argv[1]
    else:
        model = None
    if model:
        logger.debug('(falcon)(1/1)Processing DL model %s...' % (model))
        print('(falcon)(1/1)Processing DL model %s...' % (model))
        find_a_near_optimal_configuration(model)
        output.append(output_results(model))
    else:
        for num,model in enumerate(all_m):
            logger.debug('(falcon)(%s/%s)Processing DL model %s...' % (num+1,len(all_m),model))
            print('(falcon)(%s/%s)Processing DL model %s...' % (num+1,len(all_m),model))
            find_a_near_optimal_configuration(model)
            output.append(output_results(model))
    print('Output Falcon\'s results.')
    if platform.system() == "Windows":
        with open('../data/falcon.output', 'w') as f:
            for i in output:
                print(i)
                f.write(str(i)+"\n")
    else:
        with open('/home/falcon/data/falcon.output', 'w') as f:
            for i in output:
                print(i)
                f.write(str(i)+"\n")


if __name__ == '__main__':
    _create_dir()
    main()
#     main()