
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger



def create_seq_dataset(batch_size=1, data_file_path=None, do_shuffle=True, drop_remainder=True):
    type_cast_op = C.TypeCast(mstype.int32)

    dataset = ds.MindDataset([data_file_path],
                                 columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                 shuffle=do_shuffle)
    dataset = dataset.map(operations=type_cast_op, input_columns="label_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="segment_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_mask")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_ids")
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def create_classification_dataset(batch_size=1,  data_file_path=None, do_shuffle=True):

    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset([data_file_path],
                              columns_list=["input_ids", "input_mask", "segment_ids","label_ids"],
                              shuffle=do_shuffle)

    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set




