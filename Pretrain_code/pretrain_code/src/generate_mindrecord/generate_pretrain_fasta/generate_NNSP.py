import os
import math
import numpy as np
import mindspore.common.dtype as mstype
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



def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert")
    parser.add_argument("--input_file", type=str, default="/data2/bert/mindspore/datas/swissprot/pfam/mr_files/",help="Input raw text file (or comma-separated list of files).")
    # parser.add_argument("--input_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/wiki_processed/AS/wiki_31",help="Input raw text file (or comma-separated list of files).")

    parser.add_argument("--output_file", type=str, default="/data2/bert/mindspore/datas/swissprot/NNSP/mr_files/",
                        help="Output MindRecord file (or comma-separated list of files).")
    parser.add_argument("--file_index", type=str, default="0",
                        help="Output MindRecord file (or comma-separated list of files).")

    args_opt = parser.parse_args()
    return args_opt


args = parse_args()

file=args.input_file
print(file)


data_set = ds.MindDataset(file,shuffle=False,num_parallel_workers=1)

output_file = os.path.join(args.output_file,file.split("/")[-1].strip(".mindrecord")+"_NNSP.mindrecord")

print(output_file)

iterator=data_set.create_dict_iterator(num_epochs=-1, output_numpy=False)

schema = {
    "input_ids": {"type": "int32", "shape": [-1]},
    "input_mask": {"type": "int32", "shape": [-1]},
    "segment_ids": {"type": "int32", "shape": [-1]},
    "masked_lm_positions": {"type": "int32", "shape": [-1]},
    "masked_lm_ids": {"type": "int32", "shape": [-1]},
    "masked_lm_weights": {"type": "float32", "shape": [-1]},
    "next_sentence_labels": {"type": "int32", "shape": [-1]},
}
writer = FileWriter(output_file, overwrite=True)
writer.add_schema(schema)

total_written=0
for item in tqdm(iterator):
    all_data = []
    input_ids=item["input_ids"].asnumpy()
    input_mask=item["input_mask"].asnumpy()
    assert max(input_ids)<25
    masked_lm_ids=item["masked_lm_ids"].asnumpy()
    masked_lm_positions=item["masked_lm_positions"].asnumpy()
    masked_lm_weights=item["masked_lm_weights"].asnumpy()
    next_sentence_labels=item["next_sentence_labels"].asnumpy()
    segment_ids=item["segment_ids"].asnumpy()
    first_sep=np.where(input_ids==3)[0][0]


    input_ids_2=list(input_ids[:first_sep+1])
    next_sentence_labels_2=[0]
    masked_lm_ids_2=[]
    masked_lm_positions_2=[]
    masked_lm_weights_2=[]
    segment_ids_2=[0]*len(segment_ids)
    input_mask_2=[1]*len(input_ids_2)

    for i in range(len(masked_lm_ids)):
        if masked_lm_positions[i]<=first_sep:
            masked_lm_ids_2.append(masked_lm_ids[i])
            masked_lm_positions_2.append(masked_lm_positions[i])
            masked_lm_weights_2.append(masked_lm_weights[i])

    while len(masked_lm_positions_2) < len(masked_lm_positions):
        masked_lm_positions_2.append(0)
        masked_lm_ids_2.append(0)
        masked_lm_weights_2.append(0.0)

    while len(input_ids_2) < len(input_ids):
        input_ids_2.append(0)
        input_mask_2.append(0)

    assert len(input_ids_2) == len(input_ids)
    assert len(input_mask_2) == len(input_mask)
    assert len(segment_ids_2) == len(segment_ids)

    assert len(masked_lm_positions_2) == len(masked_lm_positions)
    assert len(masked_lm_ids_2) == len(masked_lm_ids)
    assert len(masked_lm_weights_2) == len(masked_lm_weights)

    input_ids_2 = np.array(input_ids_2, dtype=np.int32)
    input_mask_2 = np.array(input_mask_2, dtype=np.int32)
    segment_ids_2 = np.array(segment_ids_2, dtype=np.int32)
    masked_lm_positions_2 = np.array(masked_lm_positions_2, dtype=np.int32)
    masked_lm_ids_2 = np.array(masked_lm_ids_2, dtype=np.int32)
    masked_lm_weights_2 = np.array(masked_lm_weights_2, dtype=np.float32)
    next_sentence_labels_2 = np.array(next_sentence_labels_2, dtype=np.int32)

    data = {'input_ids': input_ids_2,
            "input_mask": input_mask_2,
            "segment_ids": segment_ids_2,
            "masked_lm_positions": masked_lm_positions_2,
            "masked_lm_ids": masked_lm_ids_2,
            "masked_lm_weights": masked_lm_weights_2,
            "next_sentence_labels": next_sentence_labels_2}
    all_data.append(data)
    if all_data:
        writer.write_raw_data(all_data)
        total_written += 1

writer.commit()
print("Wrote %d total instances", total_written)
