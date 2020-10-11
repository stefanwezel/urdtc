#!/bin/bash

# Save current working directory
BASEWD=$(pwd)

# AWA2 Dataset
#mkdir -p data/awa2
#cd data/awa2
#wget https://cvml.ist.ac.at/AwA2/AwA2-data.zip
#unzip AwA2-data.zip
#mv Animals_with_Attributes2/* .
#rm -r Animals_with_Attributes2
#rm AwA2-data.zip
#cd $BASEWD
#python preprocessing/awa_preprocessing.py -data_path data/awa2

# APY Dataset
mkdir -p data/apy
cd data/apy
wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
unzip xlsa17.zip
mv xlsa17/data/APY/* .
rm -r xlsa17
rm xlsa17.zip
cd $BASEWD
python utils/preprocessing/apy_preprocessing.py -data_path data/apy

# CUB Dataset
#mkdir -p data/cub
#cd data/cub
#wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
#tar xf CUB_200_2011.tgz
#mv CUB_200_2011/* .
#rm -r CUB_200_2011
#rm CUB_200_2011.tgz
#cd $BASEWD

# ImageNet Dataset
# mkdir -p data/ilsvrc2012
# cd data/ilsvrc2012
# wget http://datasets.d2.mpi-inf.mpg.de/xian/ILSVRC2012_res101_feature.mat
# cd $BASEWD
# python preprocessing/imnet_preprocessing.py -data_path data/ilsvrc2012
# rm data/ilsvrc2012/ILSVRC2012_res101_feature.mat


