# Pretrain Code


This section contains the data processing code for pre-training and the pre-training code <br>


## Pre-train Data Processing

First you need to build a serial relationship dataset using a tool such as ProtENN
Then organize your input data in the following format

root_data_path
        |
        |---folder1
        |       |---file_name_1.fasta
        |       |---file_name_2.fasta
        |       |---...
        |---folder2
        |       |---file_name_1.fasta
        |       |---file_name_2.fasta
        |       |---...
        |---...
        |---folderN
                |---file_name_1.fasta
                |---file_name_2.fasta
                |---...

