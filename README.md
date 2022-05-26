# Falcon
A configuration recommender system for deep learning inference services.

<img src="https://github.com/dos-lab/Falcon/blob/main/pictures/logo/logo.gif" width="150" height="150" alt="falcon logo"><br/>

## Requirements
1. Linux OS (e.g., CentOS, RedHat, Ubuntu).
2. Docker >= 20.10.12.

## How To
### Step 1: pull docker image
```
docker pull frankwu2016/falcon:latest
```
### Step 2: run docker container runtime
```
docker run -it frankwu2016/falcon:latest /bin/bash
```
### Step 3: execute falctl command
#### 1) The ```falctl evaluate``` command
##### Evaluate all DL models. Note that, the maximum number of iterations is ```30```.
```
falctl evaluate --all
```
##### (Optional) Evaluate a particular DL model. Note that, the maximum number of iterations is ```30```.
```
falctl evaluate --model [model-name]
```
Supported DL models: [bert-large|densenet-201|gru|inception-v2|inception-v4|mobilenet-v2|resnet-101|resnet-152-v2|roberta|tacotron2|transformer|vgg16]
#### 2) The ```falctl analyze``` command
##### (Optional) Analyze a particular DL model.
```
falctl analyze --model [model-name]
```
Supported DL models: [bert-large|densenet-201|gru|inception-v2|inception-v4|mobilenet-v2|resnet-101|resnet-152-v2|roberta|tacotron2|transformer|vgg16]
##### (Optional) Analyze a particular DL operator.
```
falctl analyze --operator [operator-name]
```
Supported DL operators: [add|batch_norm|concat|conv1d|conv2d|dense|multiply|relu|sigmoid|split|strided_slice|tanh|transpose]
### Step 4: check the results
#### 1) Experimental data
##### Experimental outputs.
```
/home/falcon/data/[name].output
```
##### Experimental results.
```
/home/falcon/data/experiments/[name]-[model name].csv
```
#### 2) Source data
##### Model's data. Please check the following directory.
```
/home/falcon/data/models/
```
##### Opeartor's data. Please check the following directory.
```
/home/falcon/data/operators/
```
#### 3) Log file
Please check the following log file.
```
/var/log/falcon.log
```

## License
Our implementation is released under [Apache License 2.0](./LICENSE) license except for the code derived from TuRBO.
