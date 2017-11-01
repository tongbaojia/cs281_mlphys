# CS281 Project
## Use NN to learn physics variables for signal selection and background rejection
by Nicolò Foppiani, Jonah Philion, and Tony Tong

Reference: [ATLAS WWW paper in 8 TeV](https://arxiv.org/abs/1610.05088)

## 0) Setup Environment
``source setup.sh``

This script will activate the environment I generally use on odyssey. Let me know if you want to pip install something before you pip install it. It would be good to create a specialized environment just for this project later.

## 1) Acquire Data
To convert the monte carlo root files to csvs, first make a directory outside your clone, then use ``acquire.py`` to fill this directory. For instance:

``mkdir ../data``

``python acquire.py -i /n/atlasfs/atlascode/backedup/btong/WWW/Analysis/Output/ML/ -o ../data/``

The above is hopefully the only time you have to convert anything to csv. However, in the future, if you only want to convert only certain root files and want to save each file with a tag, use a command like 

``python acquire.py -i /n/atlasfs/atlascode/backedup/btong/WWW/Analysis/Output/ML/ -o ../data/ -mcs 'bkg_ttZ0 bkg_WZqqll signal_WmWpWm' -ext 'Run1'``

## 2) Train and Analyze a Model
An example training script is included in ``train.py``. You may have to change the data directory to run this script. I didn't spend time making this script clean because for now everyone should build their own ``train.py`` suited to their needs.

``python train.py``

