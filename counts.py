import os, glob
import argparse
import json
import numpy as np

import uproot

def count_weighted(filenames):
    sumw = 0
    for fi in filenames:
        print(fi)
        ff = uproot.open(fi)
        #bl = ff.get("nanoAOD/Runs")
        bl = ff.get("Runs")
        sumw += bl.array("genEventSumw").sum()
    return sumw


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get weights (counted) for config file')
    parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()


    filenames = None
    if not args.filelist is None:
        filenames = [l.strip() for l in open(args.filelist).readlines()]
    else:
        filenames = args.filenames

    genWeight = count_weighted(filenames)
    print("Number of events (weighted)", genWeight)
