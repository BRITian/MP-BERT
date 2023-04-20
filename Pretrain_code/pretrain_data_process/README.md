# Pretrain Code


This section contains the data processing code for pre-training and the pre-training code <br>


## Pre-train Data Processing

First you need to build a serial relationship dataset using a tool such as ProtENN
Then organize your input data in the following format

root_data_path <br>
&emsp;&emsp;| <br>
&emsp;&emsp;|---folder1 <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folder2 <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folderN <br>
&emsp;&emsp;&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---... <br>
 <br>
For example, when we use Pfamily, the composition is as follows

root_data_path <br>
&emsp;&emsp;| <br>
&emsp;&emsp;|---folder1 <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folder2 <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folderN <br>
&emsp;&emsp;&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---... <br>
