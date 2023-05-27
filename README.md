# Course Project of Math8013

### Code Structure 

- [`experiments/`](./experiments/): settings for experiments
- [`models/`](./models/): model definitions
- [`utils.py`](./utils.py): utilities for training and testing
- [`0_train_kernel_size.py`](./0_train_kernel_size.py): training script for kernel size experiments
- [`1_train_feature_size.py`](./1_train_feature_size.py): training script for feature size experiments
- [`2_train_cnns.py`](./2_train_cnns.py): training script for cnn experiments
- [`3_train_channels.py`](./3_train_channels.py): training script for channels experiments
- [`4_train_mnist.py`](./4_train_mnist.py): training script for different datasets (MNIST and subsets in MedMNIST)
- [`plot_loss.py`](./plot_loss.py): plot the loss curves and accuracy curves


### Requirements

- PyTorch
- pyyaml
- easydict>=1.9
- future>=0.17.1
- tensorboard>=1.14.0
- matplotlib
- medmnist

Higher (or lower) versions should also work (perhaps with minor modifications).

### Usage 

simply run the cmd for the training:

```bash
### Experiment 1: kernel size experiments
## 5-5-3-3-3
CUDA_VISIBLE_DEVICES=0 python 0_train_kernel_size.py --work_path ./experiments/cifar10/0_kernel_size/alexnet/5-5-3-3-3
## 7-5-3-3-3
CUDA_VISIBLE_DEVICES=0 python 0_train_kernel_size.py --work_path ./experiments/cifar10/0_kernel_size/alexnet/7-5-3-3-3
## 11-5-3-3-3
CUDA_VISIBLE_DEVICES=0 python 0_train_kernel_size.py --work_path ./experiments/cifar10/0_kernel_size/alexnet/11-5-3-3-3

### Experiment 2: feature size experiments
## 16-8-8-4-1
CUDA_VISIBLE_DEVICES=0 python 1_train_feature_size.py --work_path ./experiments/cifar10/1_feature_size/alexnet/16-8-8-4-1

### Experiment 3: cnn experiments
## VGG16
CUDA_VISIBLE_DEVICES=0 python 2_train_cnns.py --work_path ./experiments/cifar10/2_cnns/vgg16

### Experiment 4: channels experiments
## 64-192-384-256-256
CUDA_VISIBLE_DEVICES=0 python 3_train_channels.py --work_path ./experiments/cifar10/3_channels/alexnet/64-192-384-256-256

### Experiment 5: different datasets
## MNIST
CUDA_VISIBLE_DEVICES=0 python 4_train_mnist.py --work_path ./experiments/mnist/alexnet/cfg_1
## PathMNIST
CUDA_VISIBLE_DEVICES=0 python 4_train_mnist.py --work_path ./experiments/pathmnist/alexnet/cfg_1

``` 

## Acknowledgments

Provided codes borrow a lot from

- [BIGBALLON/CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO)
