# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from argparse import ArgumentParser
# import tensorflow as tf
import mindspore.mindrecord as mm
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
from tqdm import tqdm
import collections
import random
from argparse import ArgumentParser
import tokenization
import numpy as np
from mindspore.mindrecord import FileWriter
import os
from tqdm import tqdm
import glob


def vis_tfrecord(file_name):
    tfe = tf.contrib.eager
    tfe.enable_eager_execution()
    raw_dataset = tf.data.TFRecordDataset(file_name)
    # raw_dataset is iterator: you can use raw_dataset.take(n) to get n data
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print("Start print tfrecord example:", example, flush=True)


def vis_mindrecord(file_name):
    data_set = ds.MindDataset(file_name,shuffle=False,num_parallel_workers=1)
    iterator=data_set.create_dict_iterator(num_epochs=-1, output_numpy=False)
    for item in tqdm(iterator):
        print(list(item["input_ids"].asnumpy()))
        print(list(item["input_mask"].asnumpy()))
        print(list(item["masked_lm_ids"].asnumpy()))
        print(list(item["masked_lm_positions"].asnumpy()))
        print(list(item["masked_lm_weights"].asnumpy()))
        print(list(item["next_sentence_labels"].asnumpy()))
        print(list(item["segment_ids"].asnumpy()))
        # you can use break here to get one data



def main():
    """
    vis tfrecord or vis mindrecord
    """
    parser = ArgumentParser(description='vis tfrecord or vis mindrecord.')
    parser.add_argument("--file_name", type=str, default='/data1/bert/Mindspore_bert/datas/bert_CN_data/output_data', help="the file name.")
    parser.add_argument("--vis_option", type=str, default='vis_mindrecord', choices=['vis_tfrecord', 'vis_mindrecord'],
                        help="option of transfer vis_tfrecord or vis_mindrecord, default is vis_tfrecord.")
    args = parser.parse_args()
    if args.vis_option == 'vis_tfrecord':
        print("start vis tfrecord: ", args.file_name, flush=True)
        vis_tfrecord(args.file_name)
    elif args.vis_option == 'vis_mindrecord':
        print("start vis mindrecord: ", args.file_name, flush=True)
        vis_mindrecord(args.file_name)
    else:
        raise ValueError("Unsupported vis option: ", args.vis_option)

if __name__ == "__main__":
    main()
