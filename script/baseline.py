import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import math


def main():
    print("Done!")

def load(filename):
    df = pd.read_csv(f)
    df.head()
    #list.df
    return df

def cut(dataframe):

    df["cutweight"] = df["weight"].copy()
    print("oringal weight", df.cutweight.sum())
    ## tau veto
    ## TO DO...but minor

    ## lepton pt cut
    df.loc[df.l0_pt <= 20, 'cutweight']= 0
    df.loc[df.l1_pt <= 20, 'cutweight']= 0
    df.loc[df.l2_pt <= 20, 'cutweight']= 0
    ## lepton eta cut
    df.loc[abs(df.l0_eta) >= 2.5, 'cutweight']= 0
    df.loc[abs(df.l1_eta) >= 2.5, 'cutweight']= 0
    df.loc[abs(df.l2_eta) >= 2.5, 'cutweight']= 0
    ## lepton overlap removal dR < 0.1
    df.loc[((df.l0_eta - df.l1_eta) ** 2  + (df.l0_phi - df.l1_phi) ** 2) <= 0.01, 'cutweight']= 0
    df.loc[((df.l0_eta - df.l2_eta) ** 2  + (df.l0_phi - df.l2_phi) ** 2) <= 0.01, 'cutweight']= 0
    df.loc[((df.l2_eta - df.l1_eta) ** 2  + (df.l2_phi - df.l1_phi) ** 2) <= 0.01, 'cutweight']= 0
    ## Met cut > 2.5

    ## Check total weight
    print("final weight", df.cutweight.sum())


# ['runNumber',
#  'lbNumber',
#  'eventNumber',
#  'SFOS',
#  'j0_m',
#  'j0_pt',
#  'j0_eta',
#  'j0_phi',
#  'l0_m',
#  'l0_pt',
#  'l0_eta',
#  'l0_phi',
#  'l0_c',
#  'l0_isEl',
#  'l1_m',
#  'l1_pt',
#  'l1_eta',
#  'l1_phi',
#  'l1_c',
#  'l1_isEl',
#  'l2_m',
#  'l2_pt',
#  'l2_eta',
#  'l2_phi',
#  'l2_c',
#  'l2_isEl',
#  'met_pt',
#  'met_phi',
#  'weight',
#  'cl']

#####################################
if __name__ == '__main__':
    main()