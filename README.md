# Introduction

This is the source code of our TCSVT 2021 paper "MARS: Learning Modality-Agnostic Representation for 
Scalable Cross-media Retrieval". Please cite the following paper if you use our code.

Yunbo Wang and Yuxin Peng, "MARS: Learning Modality-Agnostic Representation for Scalable 
Cross-media Retrieval", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2021.



# Preparation

We use Python 3.7.2, PyTorch 1.1.0, cuda 9.0, and evaluate on Ubuntu 16.04.12

1. Install anaconda downloaded from https://repo.anaconda.com/archive. And create a new environment
   sh Anaconda3-2018.12-Linux-x86_64.sh
   conda create -n MARS python=3.7.2
   conda activate MARS
   
2. Run the followed commands
   conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
   pip install -r requirements.txt



# Training and evaluation

We use the Wikipedia dataset as example, and the data is placed in ./datasets/Wiki. 
In addition, the XMedia&XMediaNet datasets are obtiand via http://59.108.48.34/tiki/XMediaNet/.
The NUS-WIDE dataset is obtained via https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html.

Run the followed command for traning&evaluation, and the configure can be found in main_MARS.py.
python main_MARS.py --datasets wiki --output_shape 128 --batch_size 64 --epochs 50 --lr [1e-4, 5e-4]  #  for Wikipedia

The common representations can be found in folder "features".

For any questions, feel free to contact us. (wangyunbo09@gmail.com)



Welcome to our [Laboratory Homepage](http://59.108.48.34/tiki/yuxinpeng/) for more information.
