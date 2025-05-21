# Distributed-ML
A university project on distributed computing

## Description
Our main goal was measuring and comparing distributed and local training metrics for a particular model and dataset.

All the models used in our experiments along with dataset preprocessing are located at `/models`

* `baseline_cifar10.ipynb` contains the current baseline measured. We did not try to achieve the best metrics for the task due to comparing distributed and local metrics only

* `baseline_old_dataset.ipynb` was initially used as a baseline, but we have not measured the metrics. In this case, it may be used as an additional example for better comparison

* `distributed_cpu.py`/`distributed_gpu.py` provide examples of a minimalist model that allows to train across multiple devices. You can follow the code structure to recreate the experiments with your own models and datasets

## How to start

### on a single device:

1. install dependencies from `requrements.txt`:

`pip3 install -r requirements.txt`

2. open `baseline_cifar10.ipynb` and run press run all

3. check if cuda is available before training if you want to use your gpu:

```
import torch
torch.cuda.is_available()
```

### on multiple devices:

1. torchrun is currently supported for linux only. Make sure you have multiple linux nodes that can ping each other by their ipv4

2. install dependencies from a `requirements.txt` file (see command above)

3. run any of the distributed examples:

**on a master node (rank=0):** 
```
torchrun --nproc_per-node=1 --nnodes=2 --node_rank=0 --master_addr=0.0.0.0 --master_port=29500 distributed_cpu.py
```
Replace `nproc_per_node` if you have more than one gpu on a node. Replace `master_addr` with the actual ip address of this machine

**on a worker node (rank=1,2,3,...)**
```
torchrun --nproc_per-node=1 --nnodes=2 --node_rank=i --master_addr=0.0.0.0 --master_port=29500 distributed_cpu.py
```
replace `i` with the rank of your worker node
