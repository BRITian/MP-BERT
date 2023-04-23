import os
from argparse import ArgumentParser
from tqdm import tqdm
parser = ArgumentParser(description="predict pfam")
parser.add_argument("--input_file", type=str,default="/data2/bert/mindspore/datas/uniref50/fasta_files/")
parser.add_argument("--out_dir", type=str,default="/data2/bert/mindspore/datas/uniref50/pfam_files/")
parser.add_argument("--gpu", type=str,default="1")
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import numpy as np
import tensorflow.compat.v1 as tf

# Suppress noisy log messages.
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]
def residues_to_one_hot(amino_acid_residues):

  to_return = []
  normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')
  for char in normalized_residues:
    if char in AMINO_ACID_VOCABULARY:
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
      to_return.append(to_append)
    elif char == 'B':  # Asparagine or aspartic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
      to_return.append(to_append)
    elif char == 'Z':  # Glutamine or glutamic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
      to_return.append(to_append)
    elif char == 'X':
      to_return.append(
          np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
    elif char == _PFAM_GAP_CHARACTER:
      to_return.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
    else:
      raise ValueError('Could not one-hot code character {}'.format(char))
  return np.array(to_return)

# def _test_residues_to_one_hot():
#     expected = np.zeros((3, 20))
#     expected[0, 0] = 1.   # Amino acid A
#     expected[1, 1] = 1.   # Amino acid C
#     expected[2, :] = .05  # Amino acid X
#
#     actual = residues_to_one_hot('ACX')
#     np.testing.assert_allclose(actual, expected)
# _test_residues_to_one_hot()
#
#
def pad_one_hot_sequence(sequence: np.ndarray,
                         target_length: int) -> np.ndarray:
  """Pads one hot sequence [seq_len, num_aas] in the seq_len dimension."""
  sequence_length = sequence.shape[0]
  pad_length = target_length - sequence_length
  if pad_length < 0:
    raise ValueError(
        'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
        .format(sequence_length, target_length))
  pad_values = [[0, pad_length], [0, 0]]
  return np.pad(sequence, pad_values, mode='constant')
#
sess = tf.Session()
graph = tf.Graph()

with graph.as_default():
  saved_model = tf.saved_model.load(sess, ['serve'], 'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760')

class_confidence_signature = saved_model.signature_def['confidences']
class_confidence_signature_tensor_name = class_confidence_signature.outputs['output'].name

sequence_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence'].name
sequence_lengths_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence_length'].name


# Load vocab
with open('trained_model_pfam_32.0_vocab.json') as f:
  vocab = json.loads(f.read())

total_seqs=[]
total_seqs.append([])

print("REMOVE CURRENT FILES")
ls = os.listdir(args.out_dir)
for i in ls:

    os.remove(os.path.join(args.out_dir,i))


for file_name in tqdm(os.listdir(args.input_file)):

    print(file_name)

    with open(os.path.join(args.input_file,file_name), "r") as reader:
        lines=reader.readlines()
        for line in tqdm(lines):
            line=line.strip()
            if line.startswith(">"):
                if len(total_seqs[-1])==500:
                    total_seqs.append([])
                total_seqs[-1].append([line[1:],""])
            else:
                total_seqs[-1][-1][-1]+=line[:1024]

    for seqs in total_seqs:

        print("COLLECT "+str(len(seqs))+" SEQS" )


        one_hot_sequence_inputs = [residues_to_one_hot(i[1]) for i in tqdm(seqs)]


        max_len_within_batch = 1024
        padded_sequence_inputs = [pad_one_hot_sequence(s, max_len_within_batch)
                                  for s in tqdm(one_hot_sequence_inputs)]
        # The first run of this cell will be slower; the subsequent runs will be fast.
        # This is because on the first run, the TensorFlow XLA graph is compiled, and
        # then is reused.

        print("RUN PREDICT")
        with graph.as_default():
          confidences_by_class = sess.run(
              class_confidence_signature_tensor_name,
              {
                  sequence_input_tensor_name: padded_sequence_inputs,
                  sequence_lengths_input_tensor_name: [len(i[1]) for i in seqs],
              })

        print("WRITE RESULT")

        for seq_index in range(len(seqs)):
            pfam=vocab[np.argmax(confidences_by_class[seq_index])]
            seq=seqs[seq_index]
            output_file=os.path.join(args.out_dir,pfam+".fasta")
            if os.path.exists(output_file)==False:
                f=open(output_file,"w")
            else:
                f=open(output_file,"a")
            f.write(">"+seq[0]+" | "+pfam+"\n")
            f.write(seq[1]+"\n")
            f.close()

        del pfam
        del confidences_by_class
        del seqs

