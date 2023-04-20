import collections
import tokenization
import numpy as np
from mindspore.mindrecord import FileWriter


class TrainingInstance():
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s
    def __repr__(self):
        return self.__str__()

def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file, vocab_file):
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
    total_written = 0
    for (_, instance) in enumerate(instances):
        all_data = []
        input_ids = tokenization.convert_tokens_to_ids(vocab_file, instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length
        input_ids.extend([0]*max_seq_length)
        input_ids=input_ids[:max_seq_length]
        input_mask.extend([0]*max_seq_length)
        input_mask=input_mask[:max_seq_length]
        segment_ids.extend([0]*max_seq_length)
        segment_ids=segment_ids[:max_seq_length]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenization.convert_tokens_to_ids(vocab_file, instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        masked_lm_positions.extend([0]*max_predictions_per_seq)
        masked_lm_positions=masked_lm_positions[:max_predictions_per_seq]
        masked_lm_ids.extend([0]*max_predictions_per_seq)
        masked_lm_ids=masked_lm_ids[:max_predictions_per_seq]
        masked_lm_weights.extend([0.0]*max_predictions_per_seq)
        masked_lm_weights=masked_lm_weights[:max_predictions_per_seq]
        assert len(masked_lm_positions) == max_predictions_per_seq
        assert len(masked_lm_ids) == max_predictions_per_seq
        assert len(masked_lm_weights) == max_predictions_per_seq
        next_sentence_label = 1 if instance.is_random_next else 0
        if total_written==0:
            print("========input_ids")
            print(input_ids)
            print("========input_mask")
            print(input_mask)
            print("========segment_ids")
            print(segment_ids)
            print("========masked_lm_positions")
            print(masked_lm_positions)
            print("========masked_lm_ids")
            print(masked_lm_ids)
            print("========masked_lm_weights")
            print(masked_lm_weights)
            print("========next_sentence_label")
            print(next_sentence_label)
        input_ids = np.array(input_ids, dtype=np.int32)
        input_mask = np.array(input_mask, dtype=np.int32)
        segment_ids = np.array(segment_ids, dtype=np.int32)
        masked_lm_positions = np.array(masked_lm_positions, dtype=np.int32)
        masked_lm_ids = np.array(masked_lm_ids, dtype=np.int32)
        masked_lm_weights = np.array(masked_lm_weights, dtype=np.float32)
        next_sentence_label = np.array(next_sentence_label, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights,
                "next_sentence_labels": next_sentence_label}
        all_data.append(data)
        if total_written == 0:
            print(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, vocab_words, rng,random_prob):
    document = all_documents[document_index]
    rng.shuffle(document)
    max_num_tokens = max_seq_length - 3
    instances = []
    current_chunk = []
    for i in range(len(document)):
        segment = document[i]
        current_chunk.append(segment)
        index_list=list(range(len(document)))
        index_list.remove(i)
        if len(index_list)>=1:
            randomIndex = rng.sample(index_list, 1)[0]
            current_chunk.append(document[randomIndex])
        (tokens_a, tokens_b, i, is_random_next) = init_tokena_and_tokenb(current_chunk, rng,
                                                                         all_documents, document_index, i,random_prob)
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1
        tokens, segment_ids = init_tokens_and_segment_ids(tokens_a, tokens_b)
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
        current_chunk = []
    return instances


def init_tokena_and_tokenb(current_chunk, rng,  all_documents, document_index, i,random_prob):
    assert len(current_chunk)<=2
    tokens_a = current_chunk[0]
    set_random=rng.random()
    if len(current_chunk) == 1 or set_random < random_prob:
        is_random_next = True
        index_list = list(range(len(all_documents)))
        index_list.remove(document_index)
        random_document_index = rng.sample(index_list, 1)[0]
        random_document = all_documents[random_document_index]
        assert len(random_document)!=0
        if len(random_document)==1:
            random_start=0
        else:
            random_start = rng.randint(0, len(random_document) - 1)
        tokens_b=random_document[random_start]
    else:
        is_random_next = False
        tokens_b=current_chunk[1]
    return (tokens_a, tokens_b, i, is_random_next)



def init_tokens_and_segment_ids(tokens_a, tokens_b):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(tokens_a)
    segment_ids.extend([0]*len(tokens_a))
    tokens.append("[SEP]")
    segment_ids.append(0)
    tokens.extend(tokens_b)
    segment_ids.extend([1]*len(tokens_b))
    tokens.append("[SEP]")
    segment_ids.append(1)
    assert len(tokens)==len(segment_ids)
    return tokens, segment_ids
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
do_whole_word_mask=False


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ('[CLS]', '[SEP]'):
            continue
        cand_indexes.append([i])
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        trunc_tokens.pop()