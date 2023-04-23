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

### Online Service (Huawei ModelArts Platform, with Huawei Ascend 910 NPU)
Huawei provides an online graphical training platform, ModelArts, for pre-training and fine-tuning of the MP-BERT, as detailed in:
https://www.huaweicloud.com/product/modelarts.html?utm_source=3.baidu.com&utm_medium=organic&utm_adplace=kapian

## Structure of MP-BERT and Finetune Task
MP-BERT is trained using publicly available unlabelled pure sequence protein sequences, by self-supervised learning in Figure a.<br>
We train and provide several different pre-trained models with different MP-BERT Hidden Layer sizes, different training data and different data compositions.
A fine-tuned framework for classification, regression and sites prediction is currently available, as shown in Figures b and c.


![structure](./images/structure.jpg)

## MP-BERT Pre-training
As MP-BERT needs to be trained on a large dataset, we recommend using a trained pre-trained model or contacting us.<br>
In our study, we used 8 * Ascend 910 32GB computing NPUs, 768GB Memory on a Huawei Atlas 800-9000 training server to complete the training.<br>
The data processing and pre-training code is stored under Pretrain_code and the training data is taken from the UniRef dataset.<br>
Current results for the pre-training task of sequence pairs using Pfamily to establish links between sequences, predicted using the [ProtENN](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split) .<br>
 

