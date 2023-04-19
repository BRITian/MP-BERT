![image](./images/MP-BERT-logo.png)


# MindSpore Protein BERT
[![](https://img.shields.io/badge/Language-python=3.7-green.svg?style=for-the-badge)]()
[![](https://img.shields.io/badge/Framework-mindspore=1.8-blue.svg?style=for-the-badge)](https://www.mindspore.cn/en)

## Install Requirements
### Huawei Atlas Server (Linux, with Huawei Ascend 910 NPU)
[![](https://img.shields.io/badge/Environment-Docker>=18.03-yellow.svg??style=flat-square)](https://www.docker.com/) 

To run MP-BERT at Ascend using the MindSpore framework, it is recommended to use Docker, an open source application container engine that allows developers to package their applications, as well as dependency packages, into a lightweight, portable container.<br> By using Docker, rapid deployment of MindSpore can be achieved and isolated from the system environment.

Download Ascend traning base image for MindSpore framework from ascendhub: <br>
https://ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo#/detail/ascend-mindspore

Note: Ascend and CANN firmware and drivers need to be installed in advance before installation.<br>
Confirm installation of ARM-based Ubuntu 18.04/CentOS 7.6 64-bit operating system.

### Nvidia GPU Server (Linux)
Linux server with GPU
Support for docker, conda and pip installation environments, see:<br>
https://www.mindspore.cn/install

### CPU Device (Linux and Windows)
Pre-training of MP-BERT is not supported using the CPU and fine-tuning of training on large datasets is not recommended. predictions calculated by the CPU are recommended to be installed using conda or pip:<br>
https://www.mindspore.cn/install

## Structure of MP-BERT and Finetune Task
M-BERT is trained using publicly available unlabelled pure sequence protein sequences, and BERT is trained by self-supervised learning.<br>
We train and provide several different pre-trained models with different BERT Hidden Layer sizes, different training data and different data compositions.

