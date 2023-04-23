seqs=[]
from tqdm import tqdm
with open("/gpu/fanlingxi/database/uniref90.fasta") as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        if line[0]==">":
            seqs.append([line,""])
        else:
            seqs[-1][-1]+=line

with open("/home/bert/data/uniref_90/uniref_90_128.fasta","w") as f:
    for seq in seqs:
        if len(seq[1])>128:
            f.write(seq[0]+"\n"+seq[1][:1024]+"\n")