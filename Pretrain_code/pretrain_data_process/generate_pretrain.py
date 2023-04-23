import random
from argparse import ArgumentParser
import tokenization
import os
from tqdm import tqdm
from generate_funcs import create_instances_from_document,write_instance_to_example_files


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert")
    parser.add_argument("--random_seed", type=int, default=100, help="Random seed for data generation.")
    parser.add_argument("--input_file", type=str, default="",help="Input raw text file (or comma-separated list of files).")
    parser.add_argument("--output_file", type=str, default="",
                        help="Output MindRecord file (or comma-separated list of files).")
    parser.add_argument("--vocab_file", type=str, default="",help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=128,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument("--dupe_factor", type=int, default=1,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of creating sequences which are shorter than the maximum length.")
    args_opt = parser.parse_args()
    return args_opt

def create_training_instances(input_file, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    all_documents = {}
    folders=os.listdir(input_file)
    folders.sort()
    rng.shuffle(folders)
    print("============\n total folders ===========")
    print(folders)
    print("============\n select folders ===========")
    print(folders)
    print("============\n length folders ===========")
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
                        if len(line)<128:
                            continue
                        count_seqs += 1
                        tokens = tokenizer.tokenize(list(line[:max_seq_length-1]))
                        all_documents[file].append(tokens)
                count_files+=1
        print("******\n******\n******")
    all_documents = list(all_documents.values())
    all_documents = [x for x in all_documents if x]

    rng.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab_dict.keys())
    instances = []

    for document_index in tqdm(range(len(all_documents))):
        instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length,
                                                        short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                                                        vocab_words, rng,0.5))
        pass

    rng.shuffle(instances)
    print("instance number"+str(len(instances)))
    print("write_instance_to_example_files", flush=True)

    write_instance_to_example_files(instances, tokenizer, args.max_seq_length,args.max_predictions_per_seq,
                                    args.output_file+"fasta.mindrecord",args.vocab_file)

def main():
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    rng = random.Random(args.random_seed)
    print("before create_training_instances", flush=True)
    create_training_instances(
        args.input_file, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
        rng)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main()
