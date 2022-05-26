import os
import csv
import numpy as np
import platform
try:
    from utils.query import query_op_data
    from utils.standardize import standardize
except:
    from query import query_op_data
    from standardize import standardize
all_o = ['add','batch_norm','concat','conv1d','conv2d','dense','multiply','relu','sigmoid','split','strided_slice','tanh','transpose']
batchsize_set = [4, 8, 16, 32, 64, 128]
cpus_set = [1, 2, 3, 4, 5]
gpumem_set = [0.8, 1.2, 1.6, 2.4]
gpupower_set = [50, 75, 100]
gputype_set = [1, 2, 3, 4]


def devide_region(model, op):


    region1 = []
    region2 = []
    total_result = [["batchsize", "cpus", "gpumem", ""]]
    for batchsize in batchsize_set:
        for cpus in cpus_set:
            for gpumem in gpumem_set:
                for gpupower in gpupower_set:
                    for gputype in gputype_set:

                        sum_y = np.array(
                            [float(query_op_data(model, op, 0, float(batchsize), float(gputype))[-1]),
                             float(query_op_data(model, op, 1, float(cpus), float(gputype))[-1]),
                             float(query_op_data(model, op, 2, float(cpus), float(gpupower))[-1]),
                             float(query_op_data(model, op, 3, float(cpus), float(gpumem))[-1])])
                        y = np.mean(sum_y)
                        total_result.append([batchsize, cpus, gpumem, gpupower, gputype, y])
                        if y > 0.85:
                            # best.append([batchsize, cpus, gpumem, gpupower, gputype])
                            region1.append((np.array(standardize([float(batchsize), float(cpus), float(gpumem),
                                                     float(gpupower), float(gputype)])), float(y)))
                        elif 0.72 < y <= 0.85:
                            # good.append([batchsize, cpus, gpumem, gpupower, gputype])
                            region2.append((np.array(standardize([float(batchsize), float(cpus), float(gpumem),
                                                     float(gpupower), float(gputype)])), float(y)))

    return region1, region2

# print(devide_region("inception-v2", "conv2d"))
# print(devide_region("add")[0])
# print(devide_region("add")[1])

# print(query_op_data("conv2d", 0, 4, 1))
