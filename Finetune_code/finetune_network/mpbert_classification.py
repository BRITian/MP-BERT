import os
from sklearn import metrics
from glob import glob
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.bert_for_finetune_cpu import BertFinetuneCellCPU
from src.bert_for_finetune import BertFinetuneCell, BertCLS,BertCLSEval
from src.dataset import create_classification_dataset
from src.utils import LossCallBack,  BertLearningRate
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
import numpy as np
import pandas as pd
from src import tokenization
_cur_dir = os.getcwd()

def sensitivity(Y_test,Y_pred,n=2):#n为分类数
    sen = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    return sen

def specificity(Y_test,Y_pred,n=2):
    spe = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return spe

def mcc(Y_test,Y_pred,n=2):
    spe = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        spe.append(spe1)
    return spe



def do_val_models(save_checkpoint_path,ds_val):
    ckpt_files=glob(os.path.join(save_checkpoint_path,"*.ckpt"))
    ckpt_files.sort()
    print("select best model from:")
    print(ckpt_files)

    best_acc=0
    best_ckpt=None

    if len(ckpt_files)==0:
        raise ValueError("No ckpt file in {}".format(save_checkpoint_path))
    if len(ckpt_files)==1:
        if "Best_Model" in ckpt_files[0]:
            return ckpt_files[0]
        else:
            raise ValueError("No ckpt file in {}".format(save_checkpoint_path))

    for ckpt_file in ckpt_files:
        if "Best_Model" in ckpt_file:
            continue
        ACC=do_eval(ds_val, BertCLSEval, args_opt.num_class,  ckpt_file,do_train=True)
        if ACC>best_acc:
            best_ckpt=ckpt_file
            best_acc=ACC
    print("Best ckpt is {}, ACC={:.6f}".format(best_ckpt,best_acc))
    best_ckpt_name="Best_Model_Num_"+best_ckpt.split("_")[-2].split("-")[-1]+".ckpt"
    os.system("cp "+best_ckpt+" "+os.path.join(save_checkpoint_path,best_ckpt_name))
    all_old_model=glob(os.path.join(save_checkpoint_path,"*.ckpt"))
    for old_model in all_old_model:
        if "Best_Model" in old_model:
            continue
        os.system("rm "+old_model)

    return os.path.join(save_checkpoint_path,best_ckpt_name)

def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path=""):
    """ do train """
    if load_checkpoint_path==None or len(load_checkpoint_path)==0:
        print("load_checkpoint_path is None")
    else:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(network, param_dict)

    if args_opt.frozen_bert == True:
        frozen_type= "frozen"
    else:
        frozen_type= "unfrozen"

    epoch_num = args_opt.epoch_num

    ds_train = create_classification_dataset(batch_size=args_opt.train_batch_size,
                                       data_file_path=dataset,
                                       do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))

    steps_per_epoch = ds_train.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")


    for param in network.get_parameters():
        if "bert.bert.bert_encoder" not in param.name:
            print("Unfrozen ==>> ", param)
            param.requires_grad = True
            continue
        if frozen_type=="frozen":
            param.requires_grad = False
            print("Frozen ==>> ", param)
        else:
            param.requires_grad = True
            print("UnFrozen ==>> ", param)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=args_opt.epoch_num)
    ckpoint_cb = ModelCheckpoint(prefix=args_opt.task_name+"_"+frozen_type,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)

    if ms.get_context("device_target") == "CPU":
        netwithgrads = BertFinetuneCellCPU(network, optimizer=optimizer)
    else:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
        netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)
    callbacks = [TimeMonitor(ds_train.get_dataset_size()), LossCallBack(ds_train.get_dataset_size()), ckpoint_cb]

    model.train(epoch_num, ds_train, callbacks=callbacks)

def do_eval(dataset=None, network=None, num_class=None, load_checkpoint_path="",do_plot=False,do_train=False):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    softmax = ms.nn.Softmax()

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    true_labels=[]
    pred_labels=[]

    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits,_,_,_,_ = model.predict(input_ids, input_mask, token_type_id, label_ids)
        true_labels.append(label_ids.asnumpy()[0][0])
        logits=softmax(logits[0])
        pred_labels.append(logits.asnumpy())

    print_result=cal_matrix(true_labels,pred_labels,num_class,load_checkpoint_path)

    if do_train==True:
        return print_result["ACC"]


def cal_matrix(true_labels, pred_labels, num_class, load_checkpoint_path, data_file_name=""):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print_result = {"model": load_checkpoint_path.split("/")[-1][:-5], "data": data_file_name}

    print_result["AUC"] = metrics.roc_auc_score(np.eye(num_class)[true_labels], pred_labels, average="macro")
    pred_labels = np.argmax(pred_labels, axis=1)
    print_result["ACC"] = metrics.accuracy_score(true_labels, pred_labels)
    print_result["precision"] = metrics.precision_score(true_labels, pred_labels, average="macro")
    print_result["Recall"] = metrics.recall_score(true_labels, pred_labels, average="macro")
    print_result["F1"] = metrics.f1_score(true_labels, pred_labels, average="macro")
    print_result["MCC"] = metrics.matthews_corrcoef(true_labels, pred_labels)

    if num_class == 2:
        print_result["Sensitivity"] = sensitivity(true_labels, pred_labels, 2)[1]
        print_result["Specificity"] = specificity(true_labels, pred_labels, 2)[1]
        print_result["MCC"] = mcc(true_labels, pred_labels)[1]

    print("\n========================================")
    print(pd.DataFrame(print_result, index=["model"]))
    print("========================================\n")
    return print_result

def truncate_seq_pair_1x(tokens_a, tokens_b, max_length):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b
    else:
        tokens_a=tokens_a[:max_length]
        tokens_b=tokens_b[:max_length-len(tokens_a)]
        return tokens_a,tokens_b

def truncate_seq_pair_2x(tokens_a, tokens_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b

def generate_predict_seq(predict_format,data,tokenizer,seq_len):
    if predict_format == "2x":
        tokens_a = list(data["seq_0"])
        tokens_b = list(data["seq_1"])
        id = str(data["id_0"]) + "\t" + str(data["id_1"])
    elif predict_format == "1x":
        tokens_a = list(data["seq"])
        tokens_b = list(data["seq"])
        id = str(data["id"])
    tokens_a = tokenizer.tokenize(tokens_a)
    tokens_b = tokenizer.tokenize(tokens_b)
    if predict_format == "1x":
        tokens_a, tokens_b = truncate_seq_pair_1x(tokens_a, tokens_b, seq_len - 3)
    elif predict_format == "2x":
        tokens_a, tokens_b = truncate_seq_pair_2x(tokens_a, tokens_b, seq_len - 3)
    assert len(tokens_a) + len(tokens_b) <= seq_len - 3

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if len(tokens_b)>0:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len

    if "label" in data.keys():
        label_id = data["label"]
    else:
        label_id = -1

    return ms.Tensor([input_ids]),ms.Tensor([input_mask]),ms.Tensor([segment_ids]),ms.Tensor([[label_id]]),id,"".join(tokens_a),"".join(tokens_b)

def do_predict(seq_len=1024, network=None, num_class=2, load_checkpoint_path="",tokenizer=None):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)


    ds_predict = pd.read_csv(args_opt.data_url).to_dict("record")  # predict csv must have["id","seq"] or ["id_0","seq_0","id_1","seq_1"]
    data_file_name=args_opt.data_url.split("/")[-1].strip(".csv")

    if "seq" in ds_predict[0].keys():
        predict_format="1x"
    elif "seq_0" in ds_predict[0].keys():
        predict_format="2x"
    else:
        raise "predict csv format ERROR"

    if args_opt.return_sequence==True or args_opt.return_csv==True:
        write_data=[]

    if args_opt.print_predict==True:
        true_labels=[]
        pred_labels=[]

    for data in ds_predict:
        input_ids, input_mask, token_type_id, label_ids,id,truncate_token_a,truncate_token_b=generate_predict_seq(predict_format,data,tokenizer,seq_len)
        logits,pooled_output,sequence_output,all_polled_output,all_sequence_output = model.predict(input_ids, input_mask, token_type_id, label_ids)
        logits=logits[0]
        data["pred_label"]=np.argmax(logits.asnumpy())

        if args_opt.print_predict == True:
            true_labels.append(data["label"])
            pred_labels.append(logits.asnumpy())

        if args_opt.return_sequence==True:
            data["truncate_0"]=truncate_token_a
            data["truncate_1"]=truncate_token_b
            # data["feature"]=[apo.asnumpy() for apo in all_polled_output]
            data["feature"]=[all_sequence_output[0].asnumpy() ,all_sequence_output[-1].asnumpy()]

        else:
            data["feature"]=None

        if args_opt.return_sequence == True or args_opt.return_csv == True:
            data["dense"] = logits.asnumpy()[1]
            write_data.append(data)

    if args_opt.print_predict==True:
        cal_matrix(true_labels, pred_labels, num_class, load_checkpoint_path,data_file_name)

    if args_opt.return_sequence==True:
        np.save(os.path.join(args_opt.output_url,data_file_name+"_predict_result.npy"),np.array(write_data))
    if args_opt.return_csv==True:
        pd.DataFrame(write_data).drop(["feature"],axis=1).to_csv(os.path.join(args_opt.output_url,data_file_name+"_predict_result.csv"))

def run_classifier():
    """run classifier task"""
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(enable_graph_kernel=True)
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    elif target == "CPU":
        if args_opt.use_pynative_mode:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=args_opt.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=args_opt.device_id)
    else:
        raise Exception("Target error, CPU or GPU or Ascend is supported.")

    train_data_file_path=os.path.join(args_opt.data_url,"train.mindrecord")
    val_data_file_path=os.path.join(args_opt.data_url,"val.mindrecord")
    test_data_file_path=os.path.join(args_opt.data_url,"test.mindrecord")

    if args_opt.do_train==True:
        netwithloss = BertCLS(bert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1)

        print("==============================================================")
        print("processor_name: {}".format(args_opt.device_target))
        print("test_name: {}".format(args_opt.task_name))
        print("batch_size: {}".format(args_opt.train_batch_size))

        do_train(train_data_file_path, netwithloss, args_opt.load_checkpoint_url, args_opt.output_url)

    if args_opt.do_eval==True:

        if args_opt.do_train==True:
            finetune_ckpt_url=args_opt.output_url
        else:
            finetune_ckpt_url=args_opt.load_checkpoint_url

        ds_val = create_classification_dataset(batch_size=args_opt.eval_batch_size,
                                           data_file_path=val_data_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        if finetune_ckpt_url.endswith(".ckpt"):
            best_ckpt=finetune_ckpt_url
        else:
            best_ckpt=do_val_models(finetune_ckpt_url,ds_val)

        ds_test = create_classification_dataset(batch_size=args_opt.eval_batch_size,
                                           data_file_path=test_data_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))

        print("======Val======")
        do_eval(ds_val, BertCLSEval, args_opt.num_class, load_checkpoint_path=best_ckpt)
        print("======Test======")
        do_eval(ds_test, BertCLSEval, args_opt.num_class, load_checkpoint_path=best_ckpt)

    if args_opt.do_predict==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_predict(bert_net_cfg.seq_length, BertCLSEval, args_opt.num_class, load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)
    print("FINISH !!!")

if __name__ == "__main__":
    print(args_opt)
    run_classifier()
