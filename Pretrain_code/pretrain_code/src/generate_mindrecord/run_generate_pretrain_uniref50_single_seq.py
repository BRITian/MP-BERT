import os

for i in range(12):
    i+=11
    cmd="python /sdz/bert/src/generate_mindrecord/generate_pretrain_uniref_single_seq.py --input_file /sdz/data/uniref_50/uniref_50_v2/fasta_files/pfam_files/ --output_file /sdz/data/uniref_50/uniref_50_v2/mr_files/uniref_50 --index_num "+str(i)+" --vocab_file /sdz/data/vocab.txt --dupe_factor 1"
    print(cmd)
    os.system(cmd)
