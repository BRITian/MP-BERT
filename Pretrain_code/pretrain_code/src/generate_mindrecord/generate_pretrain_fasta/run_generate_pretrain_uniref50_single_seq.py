import os

for i in range(4):
    i+=7
    cmd="python /sdz/bert/src/generate_mindrecord/generate_pretrain_fasta/generate_pretrain_fasta.py --input_file /sdz/data/uniref_50/uniref_50_v2/fasta_files/pfam_files/ --output_file /sdz/data/uniref_50/NSP_pfam/mr_files --index_num "+str(i)+" --vocab_file /sdz/data/vocab.txt --dupe_factor 1"
    print(cmd)
    os.system(cmd)
