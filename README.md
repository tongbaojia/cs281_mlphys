# CS281 Project
## Use NN to learn physics variables for signal selection and background rejection
by Nicolò Foppiani, Jonah Philion, and Tony Tong

Reference: [ATLAS WWW paper in 8 TeV](https://arxiv.org/abs/1610.05088)

## 1) Acquire Data
To convert the monte carlo root files to csvs, first make a directory outside your clone, then use ``acquire.py`` to fill this directory. For instance:

``mkdir ../data``

``python acquire.py -i /n/atlasfs/atlascode/backedup/btong/WWW/Analysis/Output/ML/ -o ../data/``

The above is hopefully the only time you have to convert anything to csv. However, in the future, if you only want to convert only certain root files and want to save each file with a tag, use a command like 

``python acquire.py -i /n/atlasfs/atlascode/backedup/btong/WWW/Analysis/Output/ML/ -o ../data/ -mcs 'bkg_ttZ0 bkg_WZqqll signal_WmWpWm' -ext 'Run1'``

## Package Requirements
Nothing yet...