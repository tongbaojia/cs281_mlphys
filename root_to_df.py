import ROOT
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import sys

def get_info(event):
    return [event.runNumber,
            event.lbNumber,
            event.eventNumber,
            event.SFOS,
            event.j0_m,
            event.j0_pt,
            event.j0_eta,
            event.j0_phi,
            event.l0_m,
            event.l0_pt,
            event.l0_eta,
            event.l0_phi,
            event.l0_c,
            event.l0_isEl,
            event.l1_m,
            event.l1_pt,
            event.l1_eta,
            event.l1_phi,
            event.l1_c,
            event.l1_isEl,
            event.l2_m,
            event.l2_pt,
            event.l2_eta,
            event.l2_phi,
            event.l2_c,
            event.l2_isEl,
            event.met_pt,
            event.met_phi,
            event.weight,
            ]

def get_df(file, classification):
    f = ROOT.TFile.Open(file, "read")
    t = f.Get("TinyTree")
    info = [get_info(event) + [classification] for event in tqdm(t)]

    col_names = ["runNumber",
            "lbNumber",
            "eventNumber",
            "SFOS",
            "j0_m",
            "j0_pt",
            "j0_eta",
            "j0_phi",
            "l0_m",
            "l0_pt",
            "l0_eta",
            "l0_phi",
            "l0_c",
            "l0_isEl",
            "l1_m",
            "l1_pt",
            "l1_eta",
            "l1_phi",
            "l1_c",
            "l1_isEl",
            "l2_m",
            "l2_pt",
            "l2_eta",
            "l2_phi",
            "l2_c",
            "l2_isEl",
            "met_pt",
            "met_phi",
            "weight",
            "is_sig"
            ]

    df = pd.DataFrame(info)
    df.columns = col_names
    return df

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Convert signal and background root files into one dataframe (csv).')

    parser.add_argument('-s', '--signal', type=str, required=True,
                        help='Signal file.')
    parser.add_argument('-b', '--background', type=str, required=True,
                        help='Background file.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file.')

    options = parser.parse_args(argv)
    return options

def merge_and_save(sig_df, bkg_df, output):
    print 'saving to ' + output + '...'
    df = pd.concat([sig_df, bkg_df], axis=0)
    df['internal_id'] = df.index
    df.index = range(len(df)) 
    df.to_csv(output, index=False)   
    print output + ' created!'

if __name__ == '__main__':
    options = parse_arguments(sys.argv[1:])

    sig_df = get_df(options.signal, classification=1)
    bkg_df = get_df(options.background, classification=0)

    merge_and_save(sig_df, bkg_df, options.output)








