from argparse import ArgumentParser


parser = ArgumentParser(description="Sync Fasta")
parser.add_argument("--sync", type=str)

args = parser.parse_args()

file_list=["/data2/bert/mindspore/datas/uniref50/mindrecord_file/mr_files/uniref50_"+i+".mindrecord" for i in args.sync.split(",")]
db_list=["/data2/bert/mindspore/datas/uniref50/mindrecord_file/mr_files/uniref50_"+i+".mindrecord.db" for i in args.sync.split(",")]

import os

for i in range(len(file_list)):
    os.system("/home/bert/huawei_obs/obsutil_linux_amd64_5.4.6/obsutil cp "+file_list[i]+ " obs://liutuoyu-gf/datas/uniref-50/mindrecord/mr_files/")
    os.system("/home/bert/huawei_obs/obsutil_linux_amd64_5.4.6/obsutil cp "+db_list[i]+ " obs://liutuoyu-gf/datas/uniref-50/mindrecord/mr_files/")