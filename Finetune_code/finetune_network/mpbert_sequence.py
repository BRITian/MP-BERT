# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

'''
Bert finetune and evaluation script.
'''
import os
from sklearn import metrics
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
from src.bert_for_finetune_cpu import BertFinetuneCellCPU
from src.bert_for_finetune import BertFinetuneCell, BertSeq,BertSeqEval
from src.dataset import create_seq_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate
from src.model_utils.config import config as args_opt, optimizer_cfg, bert_net_cfg
from glob import glob
import pandas as pd
from src import tokenization


_cur_dir = os.getcwd()


def sensitivity(Y_test, Y_pred, n=2):  # n为分类数
    sen = []
    con_mat = metrics.confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    return sen


def specificity(Y_test, Y_pred, n=2):
    spe = []
    con_mat = metrics.confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
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
        ACC = do_eval(ds_val, BertSeq, args_opt.num_class, ckpt_file, do_train=True, arg_metrics=args_opt.metrics)
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
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    if args_opt.frozen_bert == True:
        frozen_status = ["frozen", "unfrozen"]
        print("frozen epoch num: ", int(args_opt.epoch_num * (1 / 3)), "unfrozen epoch num: ",
              int(args_opt.epoch_num * (2 / 3)))
    else:
        frozen_status = ["unfrozen"]
        print("epoch num ", args_opt.epoch_num)

    for frozen_type in frozen_status:

        if args_opt.frozen_bert == True:
            if frozen_type == "frozen":
                epoch_num = int(args_opt.epoch_num * (1 / 3))
            else:
                epoch_num = int(args_opt.epoch_num * (2 / 3))
        else:
            epoch_num = args_opt.epoch_num

        ds_train = create_seq_dataset(batch_size=args_opt.train_batch_size, data_file_path=dataset,
                                      dataset_format=args_opt.dataset_format,
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
            layer_name = param.name
            if frozen_type == "frozen":
                if layer_name.startswith("bert.bert.bert_encoder.layers."):
                    layer_num = int(layer_name.strip("bert.bert.bert_encoder.layers.")[0])
                    if layer_num < 3:
                        param.requires_grad = False
                        print("Frozen ==>> ", param)
                    else:
                        param.requires_grad = True
                        print("UnFrozen ==>> ", param)
                else:
                    param.requires_grad = True
                    print("UnFrozen ==>> ", param)
            else:
                param.requires_grad = True
                print("UnFrozen ==>> ", param)

        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=args_opt.epoch_num)
        ckpoint_cb = ModelCheckpoint(prefix=args_opt.task_name + "_" + frozen_type,
                                     directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                     config=ckpt_config)

        if ms.get_context("device_target") == "CPU":
            netwithgrads = BertFinetuneCellCPU(network, optimizer=optimizer)
        else:
            update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
            netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)

        model = Model(netwithgrads)

        callbacks = [TimeMonitor(ds_train.get_dataset_size()), LossCallBack(ds_train.get_dataset_size()), ckpoint_cb]
        model.train(epoch_num, ds_train, callbacks=callbacks)

def cal_metrics(true_labels, pred_labels, num_class,print_result={}):
    """ calculate metrics """
    try:
        print_result["AUC"] = metrics.roc_auc_score(np.eye(num_class)[true_labels], pred_labels)
    except:
        print(
            "[Warning] Can't caculate AUC, this may be due to the fact that the predicted category is only one category, CONTINUE")
    pred_labels = np.argmax(pred_labels, axis=1)
    print_result["ACC"] = metrics.accuracy_score(true_labels, pred_labels)
    print_result["precision"] = metrics.precision_score(true_labels, pred_labels)
    print_result["Recall"] = metrics.recall_score(true_labels, pred_labels)
    print_result["F1"] = metrics.f1_score(true_labels, pred_labels)
    print_result["MCC"] = metrics.matthews_corrcoef(true_labels, pred_labels)
    if num_class == 2:
        print_result["Sensitivity"] = sensitivity(true_labels, pred_labels, 2)[1]
        print_result["Specificity"] = specificity(true_labels, pred_labels, 2)[1]
    return print_result

def do_eval(dataset=None, network=None, num_class=None, load_checkpoint_path="", do_train=False, arg_metrics="ACC"):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]

    true_labels = np.empty((0))
    pred_labels = np.empty((0, 2))

    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, label_ids = input_data
        logits = model.predict(input_ids, input_mask, segment_ids, label_ids).asnumpy()
        input_ids = input_ids.asnumpy()[0]
        label_ids = label_ids.asnumpy()[0]
        sep_loc = np.where(input_ids == 3)[0]
        if len(sep_loc) == 2:
            seq_0 = input_ids[1:sep_loc[0]]
            seq_1 = input_ids[sep_loc[0] + 1:sep_loc[1]]

            label_0 = label_ids[1:sep_loc[0]]
            label_1 = label_ids[sep_loc[0] + 1:sep_loc[1]]

            logits_0 = logits[1:sep_loc[0]]
            logits_1 = logits[sep_loc[0] + 1:sep_loc[1]]

        else:
            raise "Error data format"

        if len(seq_0) != len(logits_0) != len(label_0) or len(seq_1) != len(logits_1) != len(label_1):
            raise "Error logits"

        true_labels = np.concatenate((true_labels, label_0))
        true_labels = np.concatenate((true_labels, label_1))
        pred_labels = np.vstack((pred_labels, logits_0))
        pred_labels = np.vstack((pred_labels, logits_1))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print_result = {"model": load_checkpoint_path.split("/")[-1].strip(".ckpt")}

    print_result=cal_metrics(true_labels, pred_labels, num_class, print_result)

    print("\n========================================")
    print(pd.DataFrame(print_result, index=["model"]))
    print("========================================\n")
    if do_train == True:
        try:
            return print_result[arg_metrics]
        except:
            raise "Unknown metrics, please select from " + str(list(print_result.keys()))



def truncate_seq_pair_1x(tokens_a, tokens_b,label_a,label_b, max_length):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b,label_a,label_b
    else:
        tokens_a=tokens_a[:max_length]
        label_a=label_a[:max_length]
        tokens_b=tokens_b[:max_length-len(tokens_a)]
        label_b=label_b[:max_length-len(label_a)]
        return tokens_a,tokens_b,label_a,label_b

def truncate_seq_pair_2x(tokens_a, tokens_b,label_a,label_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            label_a.pop()
        else:
            tokens_b.pop()
            label_b.pop
    return tokens_a, tokens_b,label_a,label_b

def generate_predict_seq(predict_format,data,tokenizer,seq_len):
    if predict_format == "2x":
        tokens_a = list(data["seq_0"])
        tokens_b = list(data["seq_1"])
        if "label_0" in data.keys():
            label_a= list(data["label_0"])
            label_b= list(data["label_1"])
        else:
            label_a= None
            label_b= None

        id = str(data["id_0"]) + "|" + str(data["id_1"])
    elif predict_format == "1x":
        tokens_a = list(data["seq"])
        tokens_b = list(data["seq"])
        if "label" in data.keys():
            label_a= list(data["label"])
            label_b= list(data["label"])
        else:
            label_a= [-1]*len(tokens_a)
            label_b= [-1]*len(tokens_b)
        id = str(data["id"])
    tokens_a = tokenizer.tokenize(tokens_a)
    tokens_b = tokenizer.tokenize(tokens_b)
    if predict_format == "1x":
        tokens_a, tokens_b,label_a,label_b = truncate_seq_pair_1x(tokens_a, tokens_b,label_a,label_b, seq_len - 3)
    elif predict_format == "2x":
        tokens_a, tokens_b,label_a,label_b = truncate_seq_pair_2x(tokens_a, tokens_b,label_a,label_b, seq_len - 3)
    assert len(tokens_a) + len(tokens_b) <= seq_len - 3
    assert len(tokens_a)==len(label_a)
    assert len(tokens_b)==len(label_b)

    tokens = []
    segment_ids = []
    label_ids=[]
    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)

    for i, (token, label) in enumerate(zip(tokens_a, label_a)):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(int(label))

    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    for i, (token, label) in enumerate(zip(tokens_b, label_b)):
        tokens.append(token)
        segment_ids.append(1)
        label_ids.append(int(label))

    tokens.append("[SEP]")
    segment_ids.append(1)
    label_ids.append(0)

    assert len(label_ids)==len(tokens)

    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file, tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len
    assert len(label_ids) == seq_len

    return ms.Tensor([input_ids]),ms.Tensor([input_mask]),ms.Tensor([segment_ids]),ms.Tensor([label_ids]),id,"".join(tokens_a),"".join(tokens_b)



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
        true_labels = np.empty((0))
        pred_labels = np.empty((0, 2))

    seq_print_results=[]

    for data in ds_predict:
        input_ids, input_mask, token_type_id, label_ids,id,truncate_token_a,truncate_token_b=generate_predict_seq(predict_format,data,tokenizer,seq_len)

        logits,sequence_output = model.predict(input_ids, input_mask, token_type_id, label_ids)
        logits = logits.asnumpy()
        sequence_output = sequence_output.asnumpy()[0]
        input_ids = input_ids.asnumpy()[0]
        label_ids = label_ids.asnumpy()[0]
        sep_loc = np.where(input_ids == 3)[0]
        if len(sep_loc) == 2:
            seq_0 = input_ids[1:sep_loc[0]]
            seq_1 = input_ids[sep_loc[0] + 1:sep_loc[1]]

            label_0 = label_ids[1:sep_loc[0]]
            label_1 = label_ids[sep_loc[0] + 1:sep_loc[1]]

            logits_0 = logits[1:sep_loc[0]]
            logits_1 = logits[sep_loc[0] + 1:sep_loc[1]]

            sequence_output_0 = sequence_output[1:sep_loc[0]]
            sequence_output_1 = sequence_output[sep_loc[0] + 1:sep_loc[1]]

        else:
            raise "Error data format"

        if len(seq_0) != len(logits_0) != len(label_0) or len(seq_1) != len(logits_1) != len(label_1):
            raise "Error logits"

        if args_opt.print_predict == True:
            true_labels = np.concatenate((true_labels, label_0))
            true_labels = np.concatenate((true_labels, label_1))
            pred_labels = np.vstack((pred_labels, logits_0))
            pred_labels = np.vstack((pred_labels, logits_1))


            seq_print_result=cal_metrics(label_0,logits_0,num_class,{"id":id+"-0"})
            seq_print_results.append(seq_print_result)
            seq_print_result=cal_metrics(label_1,logits_1,num_class,{"id":id+"-1"})
            seq_print_results.append(seq_print_result)

        if args_opt.return_sequence == True or args_opt.return_csv == True:
            data["truncate_0"]=truncate_token_a
            data["truncate_1"]=truncate_token_b
            data["logits_0"]=logits_0
            data["logits_1"]=logits_1
            if -1 not in label_0:
                data["label_0"]=label_0
                data["label_1"]=label_1
            data["pred_label_0"] = np.argmax(logits_0, axis=1)
            data["pred_label_1"] = np.argmax(logits_1, axis=1)
            data["sequence_output_0"]=sequence_output_0
            data["sequence_output_1"]=sequence_output_1
            if -1 not in label_0:
                assert len(data["logits_0"])==len(data["label_0"])==len(data["pred_label_0"])==len(data["sequence_output_0"])
                assert len(data["logits_1"]) == len(data["label_1"]) == len(data["pred_label_1"])==len(data["sequence_output_1"])
            else:
                assert len(data["logits_0"]) == len(data["pred_label_0"]) == len(data["sequence_output_0"])
                assert len(data["logits_1"]) == len(data["pred_label_1"]) == len(data["sequence_output_1"])
            write_data.append(data)

    if args_opt.print_predict==True:
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("\n========================================")
        print_df=pd.DataFrame(seq_print_results)
        print(print_df)
        print("========================================\n")

        print_result = {"model": load_checkpoint_path.split("/")[-1].strip(".ckpt"),"data":data_file_name}
        print_result=cal_metrics(true_labels,pred_labels,num_class,print_result)
        print("\n========================================")
        print(pd.DataFrame(print_result, index=["model"]))
        print("========================================\n")


    if args_opt.return_sequence==True or args_opt.return_csv==True:
        np.save(os.path.join(args_opt.output_url,data_file_name+"_predict_result.npy"),np.array(write_data))



def run_seq():
    epoch_num = args_opt.epoch_num
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

    train_data_file_path = os.path.join(args_opt.data_url, "train.mindrecord")
    val_data_file_path = os.path.join(args_opt.data_url, "val.mindrecord")
    test_data_file_path = os.path.join(args_opt.data_url, "test.mindrecord")

    if args_opt.do_train == True:
        label_percent = np.array(args_opt.label_percent.split(":")).astype(int)
        netwithloss = BertSeq(bert_net_cfg, True, num_labels=args_opt.num_class,
                              with_lstm=False, dropout_prob=0.1, loss_func=args_opt.loss_name,
                              label_percent=label_percent,loss_gama=args_opt.loss_gamma)
        print("==============================================================")
        print("processor_name: {}".format(args_opt.device_target))
        print("test_name: {}".format(args_opt.task_name))
        print("batch_size: {}".format(args_opt.train_batch_size))
        print("label_percent: {}".format(str(label_percent)))

        do_train(train_data_file_path, netwithloss, args_opt.load_checkpoint_url, args_opt.output_url)

    if args_opt.do_eval == True:
        if args_opt.do_train == True:
            finetune_ckpt_url = args_opt.output_url
        else:
            finetune_ckpt_url = args_opt.load_checkpoint_url
        if os.path.exists(val_data_file_path) == True:
            ds_val = create_seq_dataset(batch_size=args_opt.eval_batch_size, data_file_path=val_data_file_path,
                                        dataset_format=args_opt.dataset_format,
                                        do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"), drop_remainder=False)
        if finetune_ckpt_url.endswith(".ckpt"):
            best_ckpt = finetune_ckpt_url
        else:
            if os.path.exists(val_data_file_path) == True:
                best_ckpt = do_val_models(finetune_ckpt_url, ds_val)
            else:
                load_finetune_checkpoint_dir = make_directory(finetune_ckpt_url)
                best_ckpt = LoadNewestCkpt(load_finetune_checkpoint_dir, args_opt.task_name)

                all_old_model = glob(os.path.join(load_finetune_checkpoint_dir, "*.ckpt"))
                for old_model in all_old_model:
                    if best_ckpt in old_model:
                        continue
                    os.system("rm " + old_model)

        ds_test = create_seq_dataset(batch_size=args_opt.eval_batch_size, data_file_path=test_data_file_path,
                                     dataset_format=args_opt.dataset_format,
                                     do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"), drop_remainder=False)

        print("======Val======")
        if os.path.exists(val_data_file_path) == True:
            do_eval(ds_val, BertSeq, args_opt.num_class, load_checkpoint_path=best_ckpt)
        print("======Test======")
        do_eval(ds_test, BertSeq, args_opt.num_class, load_checkpoint_path=best_ckpt)

    if args_opt.do_predict==True:
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file, do_lower_case=False)
        finetune_ckpt_url = args_opt.load_checkpoint_url
        if args_opt.do_eval==False:
            if finetune_ckpt_url.endswith(".ckpt") ==False:
                raise "For predict, if do_eval==False, you should select only one checkpoint file and this file should end with .ckpt"
            else:
                best_ckpt=finetune_ckpt_url
        do_predict(bert_net_cfg.seq_length, BertSeqEval, args_opt.num_class, load_checkpoint_path=best_ckpt,
                   tokenizer=tokenizer)
    print("FINISH !!!")


if __name__ == "__main__":
    print(args_opt)
    bert_net_cfg.num_hidden_layers=bert_net_cfg.num_hidden_layers-int(args_opt.cut_layer_num)
    print(bert_net_cfg)
    run_seq()
