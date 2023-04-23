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
"""Generate pretrain MindRecord for BERT."""

import collections
import random
from argparse import ArgumentParser
import tokenization
import numpy as np
from mindspore.mindrecord import FileWriter
import os
from tqdm import tqdm
from pretrain_valid_funcs import create_instances_from_document,write_instance_to_example_files


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert")
    parser.add_argument("--input_file", type=str, default="/data2/bert/mindspore/datas/uniref50/pfam_files/",help="Input raw text file (or comma-separated list of files).")
    # parser.add_argument("--input_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/wiki_processed/AS/wiki_31",help="Input raw text file (or comma-separated list of files).")

    parser.add_argument("--output_file", type=str, default="/data2/bert/mindspore/datas/uniref50/mr_files/uniref_50_all_data",
                        help="Output MindRecord file (or comma-separated list of files).")
    parser.add_argument("--vocab_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/chinese_L-12_H-768_A-12/vocab.txt",help="The vocabulary file that the BERT model was trained on.")
    # parser.add_argument("--vocab_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/bert-base-chinese-vocab.txt",help="The vocabulary file that the BERT model was trained on.")

    parser.add_argument("--index_num", type=str,default="0")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. "
                             "Should be True for uncased models and False for cased models.")
    parser.add_argument("--do_whole_word_mask", type=bool, default=False,
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=128,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed for data generation.")
    parser.add_argument("--dupe_factor", type=int, default=5,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of creating sequences which are shorter than the maximum length.")
    args_opt = parser.parse_args()
    return args_opt


def create_training_instances(input_file, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    print("start read fastas")
    print("from "+str(25*int(args.index_num)) +" to "+ str(25*(int(args.index_num)+1)))
    folders=os.listdir(input_file)
    folders.sort()
    folders=folders[25*int(args.index_num):25*(int(args.index_num)+1)]
    print(folders)
    print(len(folders))
    count_files=0
    count_seqs=0
    for folder in tqdm(folders):
        pfam_files=os.listdir(os.path.join(input_file,folder))
        for file in pfam_files:
            if file not in all_documents.keys():
                all_documents[file]=[]
            with open(os.path.join(input_file,folder,file), "r") as reader:
                lines=reader.readlines()
                for line in lines:
                    line=line.strip()
                    if line.startswith(">"):
                        continue
                    else:
                        count_seqs += 1
                        tokens = tokenizer.tokenize(list(line))
                        all_documents[file].append(tokens)
                count_files+=1
        print("******\n******\n******")

    all_documents = list(all_documents.values())
    caculate_only_one_seq = len(all_documents) / count_seqs
    print("files number: " + str(count_files))
    print("pfam number " + str(len(all_documents)))
    print("seq number" + str(count_seqs))
    print("minus prob" + str(caculate_only_one_seq))

    rng.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab_dict.keys())
    instances = []
    for instances_iter in range(dupe_factor):

        for document_index in tqdm(range(len(all_documents))):
            instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length,
                                                            short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                                                            vocab_words, rng,0.5-caculate_only_one_seq))

        rng.shuffle(instances)
        print("write_instance_to_example_files", flush=True)
        #fsfii:fasta files index

        write_instance_to_example_files(instances, tokenizer, args.max_seq_length,
                                        args.max_predictions_per_seq, args.output_file+"_fasta_index_"+str(args.index_num)+"_iter_"+str(instances_iter)+".mindrecord",args.vocab_file)
        instances=[]
    return instances

def main():
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    rng = random.Random(args.random_seed)
    print("before create_training_instances", flush=True)
    instances = create_training_instances(
        args.input_file, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
        rng)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main()