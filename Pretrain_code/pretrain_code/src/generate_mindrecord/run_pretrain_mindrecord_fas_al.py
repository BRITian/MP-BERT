

from argparse import ArgumentParser
import os
import threading
from tqdm import tqdm

def write_fasta(seqs,file_name):
    f=open(file_name,"w")
    for seq in seqs:
        f.write(seq[0]+"\n")
        f.write(seq[1]+"\n")

def run_thread_os(file):
    cmd="python /home/ma-user/modelarts/user-job-dir/bert/src/generate_mindrecord/generate_pretrain_mindrecord.py --vocab_file "+args.vocab_file+" --input_file "+file+" --output_file "+os.path.join(mr_files,file.split("/")[-1])
    os.system(cmd)
    print("FINISH "+file)





parser = ArgumentParser(description="Generate MindRecord for bert")
parser.add_argument("--input_file", type=str,
                    default="/data1/bert/Mindspore_bert/datas/bert_CN_data/datas/uniprot_swiss_prot.fasta",
                    help="Input raw text file (or comma-separated list of files).")
# parser.add_argument("--input_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/wiki_processed/AS/wiki_31",help="Input raw text file (or comma-separated list of files).")

parser.add_argument("--output_file", type=str,
                    default="/data1/bert/Mindspore_bert/datas/bert_CN_data/",
                    help="Output MindRecord file (or comma-separated list of files).")
parser.add_argument("--vocab_file", type=str,
                    default="/data1/bert/Mindspore_bert/datas/bert_CN_data/chinese_L-12_H-768_A-12/vocab.txt",
                    help="The vocabulary file that the BERT model was trained on.")



args = parser.parse_args()

fasta_files=os.path.join(args.output_file,"fasta_files")
if os.path.exists(fasta_files)==False:
    os.mkdir(fasta_files)

mr_files=os.path.join(args.output_file,"mr_files")
if os.path.exists(mr_files)==False:
    os.mkdir(mr_files)

seqs=[]
file_index=0
file_name_list=[]
with open(args.input_file, "r") as reader:
    lines=reader.readlines()
    for line in lines:
        line=line.strip()
        if line.startswith(">"):
            if len(seqs) == 100000:
                fasta_file_name = os.path.join(fasta_files, args.input_file.split("/")[-1].strip(".fasta") + "_" + str(
                    file_index) + ".fasta")
                write_fasta(seqs, fasta_file_name)
                file_name_list.append(fasta_file_name)
                print("generate " + fasta_file_name + " , len= " + str(len(seqs)))
                seqs = []
                file_index += 1

            seqs.append([line,""])
        else:
            seqs[-1][-1]+=line

    fasta_file_name = os.path.join(fasta_files,args.input_file.split("/")[-1].strip(".fasta") + "_" + str(file_index) + ".fasta")
    print("generate " + fasta_file_name + " , len= " + str(len(seqs)))
    write_fasta(seqs,fasta_file_name)
    file_name_list.append(fasta_file_name)

print("Found total file "+str(len(file_name_list)))

tsk=[]
tsk_name=[]

for file in file_name_list:
    thread1 = threading.Thread(target = run_thread_os,args=(file,))
    tsk_name.append(file.split("/")[-1])
    tsk.append(thread1)

for t in tqdm(range(len(tsk))):
    print('start Thread ' +tsk_name[t])
    tsk[t].start()

for t in range(len(tsk)):
    print("finish Thread "+tsk_name[t])
    tsk[t].join()