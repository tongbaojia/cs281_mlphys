import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import math

'''
This file runs over the csvs and output the fixed cut
selection efficiencies, and number of events before and after the cuts.
To run it, simply: python baseline.py
'''


def main():
    df = load("../data/signal_WmWpWm.csv")

    for i in [
        # "bkg_Gee.csv",
        # "bkg_Gtautau.csv",  
        # "bkg_stl.csv",   
        # "bkg_ttW0.csv",  
        # "bkg_ttW2.csv",  
        # "bkg_ttZ1.csv",    
        # "bkg_Wtll.csv",    
        # "bkg_Zmumu.csv",    
        # "bkg_ZZllll.csv",  
        # "bkg_ZZvvll.csv",
        # "bkg_Gmumu.csv",
        # "bkg_stlat.csv",    
        # "bkg_ttll.csv",  
        # "bkg_ttW1.csv",  
        # "bkg_ttZ0.csv",  
        # "bkg_Wtllat.csv",
        # "bkg_WZlvll.csv", 
        # "bkg_Zee.csv",     
        # "bkg_Ztautau.csv",  
        # "bkg_ZZqqll.csv", 
        "bkg_WZlvll.csv", 
        "bkg_WZqqll.csv", 
        "signal_WpWpWm.csv",
        "signal_WmWpWm.csv",
        ]:  
        print(i)
        df = load("../data/" + i)
        cut(df)
    print("Done!")

def load(filename):
    df = pd.read_csv(filename)
    df.head()
    #list.df
    return df

def cut(df):

    df["cutweight"] = df["weight"].copy()
    original_weight = df.cutweight.sum()
    original_weight_SFOS0 = df[df.SFOS == 0].cutweight.sum()
    original_weight_SFOS1 = df[df.SFOS == 1].cutweight.sum()
    original_weight_SFOS2 = df[df.SFOS == 2].cutweight.sum()

    ## Tau veto
    ## This is going to be a minor effect. Ignore for now

    ## Fiducial Leptons
    ## lepton pt cut
    df.loc[df.l0_pt <= 20, 'cutweight']= 0
    df.loc[df.l1_pt <= 20, 'cutweight']= 0
    df.loc[df.l2_pt <= 20, 'cutweight']= 0
    ## lepton eta cut
    df.loc[abs(df.l0_eta) >= 2.5, 'cutweight']= 0
    df.loc[abs(df.l1_eta) >= 2.5, 'cutweight']= 0
    df.loc[abs(df.l2_eta) >= 2.5, 'cutweight']= 0

    ## Lepton overlap removal 
    ## dR < 0.1
    df.loc[(dR(df.l0_eta, df.l0_phi, df.l1_eta, df.l1_phi)) <= 0.01, 'cutweight']= 0
    df.loc[(dR(df.l2_eta, df.l2_phi, df.l1_eta, df.l1_phi)) <= 0.01, 'cutweight']= 0
    df.loc[(dR(df.l0_eta, df.l0_phi, df.l2_eta, df.l2_phi)) <= 0.01, 'cutweight']= 0
    
    ## Met lep angle cut > 2.5
    df.loc[abs(map_phi(df.phi_3l) - map_phi(df.met_phi)) < 2.5, 'cutweight']= 0

    ## Met specific cut
    ## For SFOS--1
    df.loc[(df.SFOS == 1) & (df.met_pt < 45), 'cutweight']= 0
    ## For SFOS--2
    df.loc[(df.SFOS == 2) & (df.met_pt < 55), 'cutweight']= 0

    ## SF Mass for SFOS--0
    df.loc[(df.SFOS == 0) & (((df.l0_l1_isEl % 2 == 0) & (df.l0_l1_m < 20)) | ((df.l1_l2_isEl % 2 == 0) & (df.l1_l2_m < 20)) | ((df.l2_l0_isEl % 2 == 0) & (df.l2_l0_m < 20))), 'cutweight']= 0

    ## Z veto
    ## for SFOS--0
    df.loc[(df.SFOS == 0) 
    & ((df.l0_l1_isEl == 2) & ((df.l0_l1_m > 75) & (df.l0_l1_m < 105))) 
    & ((df.l1_l2_isEl == 2) & ((df.l1_l2_m > 75) & (df.l1_l2_m < 105))) 
    & ((df.l2_l0_isEl == 2) & ((df.l2_l0_m > 75) & (df.l2_l0_m < 105))), 'cutweight']= 0
    ## for SFOS--1
    df.loc[(df.SFOS == 1) &
    ( (((df.l0_l1_isEl % 2 == 0) & (df.l0_l1_c == 0)) & ((df.l0_l1_m > 55) & (df.l0_l1_m < 110))) 
    | (((df.l1_l2_isEl % 2 == 0) & (df.l1_l2_c == 0)) & ((df.l1_l2_m > 55) & (df.l1_l2_m < 110))) 
    | (((df.l2_l0_isEl % 2 == 0) & (df.l2_l0_c == 0)) & ((df.l2_l0_m > 55) & (df.l2_l0_m < 110)))
    ), 'cutweight']= 0
    ## for SFOS--2
    df.loc[(df.SFOS == 2) &
    ( (((df.l0_l1_isEl % 2 == 0) & (df.l0_l1_c == 0)) & ((df.l0_l1_m > 70) & (df.l0_l1_m < 110))) 
    | (((df.l1_l2_isEl % 2 == 0) & (df.l1_l2_c == 0)) & ((df.l1_l2_m > 70) & (df.l1_l2_m < 110))) 
    | (((df.l2_l0_isEl % 2 == 0) & (df.l2_l0_c == 0)) & ((df.l2_l0_m > 70) & (df.l2_l0_m < 110)))
    ), 'cutweight']= 0

    ## Inclusive jet veto; could be ignored
    #df.loc[(df.Njet <= 1), 'cutweight']= 0

    ## Check total weight
    final_weight = df.cutweight.sum()
    final_weight_SFOS0 = df[df.SFOS == 0].cutweight.sum()
    final_weight_SFOS1 = df[df.SFOS == 1].cutweight.sum()
    final_weight_SFOS2 = df[df.SFOS == 2].cutweight.sum()

    ## Get the output
    print(original_weight, final_weight)
    print("Original: {:.3f} Final {:.3f} Eff {:.3f}".format(original_weight, final_weight, (final_weight/ original_weight) * 100))
    print("SFOS0 Original: {:.3f} Final {:.3f} Eff {:.3f}".format(original_weight_SFOS0, final_weight_SFOS0, (final_weight_SFOS0/ (original_weight_SFOS0 + 0.0001)) * 100))
    print("SFOS1 Original: {:.3f} Final {:.3f} Eff {:.3f}".format(original_weight_SFOS1, final_weight_SFOS1, (final_weight_SFOS1/ (original_weight_SFOS1 + 0.0001)) * 100))
    print("SFOS2 Original: {:.3f} Final {:.3f} Eff {:.3f}".format(original_weight_SFOS2, final_weight_SFOS2, (final_weight_SFOS2/ (original_weight_SFOS2 + 0.0001)) * 100))




def map_phi(phi):
    while (phi >= np.pi) is True:
        phi -= np.pi
    while (phi < np.pi) is True:
        phi += np.pi
    return phi

def dR(eta1, phi1, eta2, phi2):
    phi1 = map_phi(phi1)
    phi2 = map_phi(phi2)
    return np.sqrt((eta1-eta2) ** 2 + (phi1-phi2) ** 2)

#####################################
if __name__ == '__main__':
    main()