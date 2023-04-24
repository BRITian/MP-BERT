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
"""Generate MindRecord for BERT finetuning runner: TNEWS."""
import os
import pandas as pd
import random
random.seed(100)
import json
import csv
from argparse import ArgumentParser
import six
import numpy as np
import tokenization
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert task: Protein")
    parser.add_argument("--data_dir", type=str, default="F:/bert/data_for_classification_with_val/stability")
    parser.add_argument("--task_name", type=str, default="tnews", help="The name of the task to train.")
    parser.add_argument("--vocab_file", type=str, default="F:/S500/mindspore/bert/src/generate_mindrecord/vocab_v2.txt",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default="F:/bert/data_for_classification_with_val/stability",
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=False, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", type=bool, default=True,
                        help="Whether to run the model in inference mode on the test set.")
    args_opt = parser.parse_args()
    return args_opt


class PaddingInputExample():
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures():
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(list(example.text_a))
    tokens_b = tokenizer.tokenize(list(example.text_b))

    data_format = example.data_format
    if data_format == 1:
        tokens_a, tokens_b=truncate_seq_pair_1x(tokens_a, tokens_b, max_seq_length - 3)
    elif data_format == 2:
        tokens_a, tokens_b=truncate_seq_pair_2x(tokens_a, tokens_b, max_seq_length - 3)
    else:
        raise "Unknown data format"


    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)


    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = float(example.label)
    if ex_index < 20:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a MindRecord file."""

    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "float32", "shape": [-1]},
        "is_real_example": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0
    random.shuffle(examples)
    for (ex_index, example) in enumerate(examples):
        all_data = []
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_id, dtype=np.float32)
        is_real_example = np.array(feature.is_real_example, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
                "is_real_example": is_real_example}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)


def truncate_seq_pair_1x(tokens_a, tokens_b, max_length):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b
    else:
        tokens_a=tokens_a[:max_length]
        tokens_b=tokens_b[:max_length-len(tokens_a)]
        return tokens_a,tokens_b

def truncate_seq_pair_2x(tokens_a, tokens_b,max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b


# This function is not used by this file but is still used by the Colab and people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_val_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(line.strip().split("_!_"))
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        print("FILE")
        print(input_file)
        lines=pd.read_csv(input_file)
        return lines

class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,data_format=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.data_format=data_format


class TnewsProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_val_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, i)
            if "seq_0" in lines.columns :
                text_a = lines['seq_0'][i]
                text_b = lines['seq_1'][i]
                data_format=2
            else:
                text_a = lines['seq'][i]
                text_b = lines['seq'][i]
                data_format=1
            label = float(lines['label'][i])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,data_format=data_format))
        return examples


def main():
    processors = {
        "tnews": TnewsProcessor,
    }

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = ["0","1"]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=False)
    if os.path.exists(args.output_dir)==False:
        os.mkdir(args.output_dir)
    if args.do_train:
        print("data_dir:", args.data_dir)
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file)

        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples), flush=True)

    if args.do_predict:
        predict_examples = processor.get_val_examples(args.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(args.output_dir, "val.mindrecord")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                args.max_seq_length, tokenizer,
                                                predict_file)

    if args.do_predict:
        predict_examples = processor.get_test_examples(args.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(args.output_dir, "test.mindrecord")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                args.max_seq_length, tokenizer,
                                                predict_file)

        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
              len(predict_examples), num_actual_predict_examples,
              len(predict_examples) - num_actual_predict_examples, flush=True)


if __name__ == "__main__":
    args = parse_args()
    main()
