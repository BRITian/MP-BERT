import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

with open("/data2/bert/mindspore/codes/bert/datas/task_seq.json", 'r') as f:
    # 读取demp.json文件内容
    seq_res = json.load(f)

vocab_list=[i.strip() for i in open("/data2/bert/mindspore/codes/bert/datas/bert_CN_data/wiki_processed/AA/vocab.txt").readlines()]

for seq_info in seq_res:
    seq=[vocab_list[i] for i in seq_info["pretrain_embedding"]["input_ids"][0]]
    seq_index=seq.index("[SEP]")
    seq=seq[1:seq_index]
    pretrain_embedding=np.average(np.array(seq_info["pretrain_embedding"]["embeddings"]),axis=1)[1:seq_index]
    pretrain_embedding=np.array([i/sum(pretrain_embedding) for i in pretrain_embedding])
    finetune_embedding=np.average(np.array(seq_info["finetune_embedding"]["embeddings"]),axis=1)[1:seq_index]
    finetune_embedding=np.array([i/sum(finetune_embedding) for i in finetune_embedding])
    rnn_embedding=np.average(np.sum(np.array(seq_info["finetune_embedding"]["rnn_embeddings"]),axis=2),axis=1)[1:seq_index]
    rnn_embedding=np.array([i/sum(rnn_embedding) for i in rnn_embedding])

    plt.figure(figsize=(20,10))
    plt.plot(range(len(seq)),pretrain_embedding,label="pretrain")
    plt.plot(range(len(seq)),finetune_embedding,label="finetune")
    plt.plot(range(len(seq)),rnn_embedding,label="rnn")
    plt.legend()
    plt.xticks(range(len(seq)),list(seq))
    plt.savefig("/data2/bert/mindspore/codes/bert/datas/task_seq.png")

    print()