import os
import csv
from argparse import ArgumentParser
import numpy as np
from mindspore.mindrecord import FileWriter
import pandas as pd
import sys
sys.path.append("..")
import tokenization

def parse_args():
    parser = ArgumentParser(description="MP-BERT sequence")
    parser.add_argument("--data_dir", type=str, default=r"F:\bert\results_for_paper\data\finetune_data\PPI\PPI_Site\PDB\csv_data\one_six",
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file", type=str, default="F:/S500/mindspore/bert/src/generate_mindrecord/vocab_v2.txt",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default=r"F:\bert\results_for_paper\data\finetune_data\PPI\PPI_Site\PDB\mr_data\one_six",
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=False, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", type=bool, default=True,
                        help="Whether to run the model in inference mode on the test set.")
    args_opt = parser.parse_args()
    return args_opt


class InputExample():

    def __init__(self,  text_a, text_b=None, label_a=None,label_b=None,data_format=None):

        self.text_a = text_a
        self.text_b = text_b
        self.label_a = label_a
        self.label_b = label_b
        self.data_format=data_format

class InputFeatures():
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class PaddingInputExample():
    """
    Fake example
    """

class DataProcessor():

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_val_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file):

        return pd.read_csv(input_file)

class MPB_SEQ_Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")))

    def get_val_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")))

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines.iterrows()):

            if "seq_0" in lines.columns and "label_0" in lines.columns :
                text_a = list(line[1]['seq_0'])
                label_a = list(line[1]['label_0'])
                text_b = list(line[1]['seq_1'])
                label_b = list(line[1]['label_1'])
                data_format=2
            elif "seq" in lines.columns and "label" in lines.columns:
                text_a = list(line[1]['seq'])
                label_a = list(line[1]['label'])
                text_b = list(line[1]['seq'])
                label_b = list(line[1]['label'])
                data_format=1
            examples.append(InputExample(text_a=text_a, label_a=label_a,text_b=text_b, label_b=label_b,data_format=data_format))
        return examples

def truncate_seq_pair_1x(tokens_a, tokens_b,label_a,label_b, max_length):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b,label_a,label_b
    else:
        tokens_a=tokens_a[:max_length]
        label_a=label_a[:max_length]
        tokens_b=tokens_b[:max_length-len(tokens_a)]
        label_b=label_b[:max_length-len(label_a)]
        return tokens_a,tokens_b,label_a,label_b

def truncate_seq_pair_2x(tokens_a, tokens_b,label_a,label_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            label_a.pop()
        else:
            tokens_b.pop()
            label_b.pop
    return tokens_a, tokens_b,label_a,label_b


def convert_single_example(ex_index, example,  max_seq_length, tokenizer):

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=[0] * max_seq_length)


    if "0" not in example.label_a:
        label_map={"A":0,"C":1,"D":3,"E":4,"G":5,"H":6,"I":7,"L":8,"M":9,"N":10,"O":11,"P":12,"R":13,"S":14,"T":15,"U":16,"V":17,"X":18,"Y":19}
        label_a=[label_map[x] for x in example.label_a]
        label_b=[label_map[x] for x in example.label_b]
    else:
        label_a = example.label_a
        label_b = example.label_b

    max_token_length=max_seq_length - 3
    tokens_a = example.text_a
    tokens_a=tokenizer.tokenize(tokens_a)
    tokens_b = example.text_b
    tokens_b=tokenizer.tokenize(tokens_b)
    data_format=example.data_format

    if data_format == 1:
        tokens_a, tokens_b, label_a, label_b=truncate_seq_pair_1x(tokens_a, tokens_b,label_a,label_b,max_token_length)
    elif data_format == 2:
        tokens_a, tokens_b, label_a, label_b=truncate_seq_pair_2x(tokens_a, tokens_b,label_a,label_b,max_token_length)
    else:
        raise "Unknown data format"

    tokens = []
    segment_ids = []
    label_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)

    for i, (token,label) in enumerate(zip(tokens_a,label_a)):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(int(label))

    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    for i, (token,label) in enumerate(zip(tokens_b,label_b)):
        tokens.append(token)
        segment_ids.append(1)
        label_ids.append(int(label))

    tokens.append("[SEP]")
    segment_ids.append(1)
    label_ids.append(0)

    assert len(label_ids)==len(tokens)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)


    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 1:
        print("*** Example ***")
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s " % (label_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids)
    return feature


def file_based_convert_examples_to_features(
        examples,  max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a MINDRecord file."""

    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0
    skip_seq=0

    total_label=[]

    for (ex_index, example) in enumerate(examples):
        all_data = []
        if len(example.text_a)!=len(example.label_a):
            raise "length not equal"
        if len(example.text_b)!=len(example.label_b):
            raise "length not equal"
        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)
        total_label.extend(feature.label_id)
        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_id, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)
    print("skip "+str(skip_seq))
    print(np.sum(total_label)/len(total_label))

def main():

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processor = MPB_SEQ_Processor()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=False)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, train_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))

    if args.do_eval:
        eval_examples = processor.get_val_examples(args.data_dir)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(args.output_dir, "val.mindrecord")
        file_based_convert_examples_to_features(eval_examples,
                                                args.max_seq_length, tokenizer,
                                                eval_file)
        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
              len(eval_examples), num_actual_eval_examples,
              len(eval_examples) - num_actual_eval_examples)

    if args.do_test:
        predict_examples = processor.get_test_examples(args.data_dir)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(args.output_dir, "test.mindrecord")
        file_based_convert_examples_to_features(predict_examples,
                                                args.max_seq_length, tokenizer,
                                                predict_file)

        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
              len(predict_examples), num_actual_predict_examples,
              len(predict_examples) - num_actual_predict_examples)


if __name__ == "__main__":
    args = parse_args()
    main()
