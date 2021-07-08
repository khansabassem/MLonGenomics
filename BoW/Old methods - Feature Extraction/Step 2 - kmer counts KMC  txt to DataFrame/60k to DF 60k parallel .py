#!/usr/bin/env python
# coding: utf-8
#     ''''
#     Notice: THIS SCRIPT ONLY WORKS FROM COMMAND-LINE/IDE
#     IT DOESN'T WORK WITH IPYTHON_NB
#     ''''
#     ''''
#    This script takes the output of KMC3 kmer counter and transforms it to a format compatible with ML training.
#    Input format two columns: [kmer] [count]
#                                AAAA   2
#                                AAAC   0
#                                AAAG   4
#                                AAAT   3
#                                AACA   0
#                                ..    ..
#                                TTTT   2

#    Output format: DataFrame
# 	                            gaac gaat .. cgaa cgat
#                           0   2	6	 ..  1 	   1
#                           1   5	0	 ..  4     5
#                           ..  ..  ..   ..  ..    ..
#                       59999   2   4    ..  0     5
#     ''''
from __future__ import division
import itertools
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from typing import Sequence
from IPython.display import display
import numpy as np
import pandas as pd
import time
#parallel
import psutil
import os
from pathlib import Path
import glob
import natsort
from ctypes import c_int32
import gc
import sys
import threading
pbar = tqdm()
pd.options.display.max_rows = 5
pd.options.display.max_columns = 10

# Create all features given a k-mer length
def generate_all_kmers(k: int, alphabet="CGAT") -> Sequence[str]:
    """generate all polymers of length k
    Args:
        k: length of the Kmer
        alphabet: base unit of the kmer, [C, G, A, T] for DNA
    """

    def _generate_all_kmers(partial_kmers: Sequence[str]):
        if len(partial_kmers[0]) == k:
            return partial_kmers
        return _generate_all_kmers(
            partial_kmers=list(
                itertools.chain.from_iterable(
                    [partial_kmer + n for n in alphabet]
                    for partial_kmer in partial_kmers
                )
            )
        )

    return _generate_all_kmers(alphabet)

# Load KMC kmer count output from disk files and transform into dataframe
def get_files(directory,pattern):
    '''
    Get the files of a directory, this returns the files in sorted order,
    Sort is very important as it assures DataFrame order
    corresponding to the original data order
    '''
    # pl = glob.glob(directory+r"\*.txt")
    files = natsort.natsorted(os.listdir(directory))
    # for p in files:
    #     display(r"\n" + p)
    for path in files:
        yield directory + r"\\" + path

def process_file(filename):
    #     ''''
    #     Reads a txt file, returns a dataframe
    #     ''''
    global dt, features
    dt = pd.DataFrame(columns=features)
    sample = pd.read_fwf(filename,sep=" ", header=None).T
    new_header = sample.iloc[0] #grab the first row for the header
    sample = sample[1:] #take the data less the header row
    sample.columns = new_header #set the header row as the df header
    dt = dt.append(sample, ignore_index=True)  #APPEND Sample to df DataSet
    pbar.update(1)
    return dt

def pd_wrapper(batch,processes=-1):
    # Decide how many proccesses will be created, -1 for all threads
    num_cpus = processes
    if processes <=0:
            num_cpus = psutil.cpu_count(logical=False)
    print('files:%s,  procs:%s'%(len(batch),num_cpus))
    # Create the pool
    pool = Pool(processes=num_cpus)
    start = time.time()
    result = []

    for dfs in tqdm(pool.imap(process_file, batch, chunksize = 10),total=len(batch)):
        result.append(dfs)

    # Concat dataframes to one dataframe
    data = pd.concat(result, ignore_index=True)
    end = time.time()
    print("\nCompleted in: %s sec"%(end - start))
    return data

k = 7
features = []
features = generate_all_kmers(k, alphabet="ACGT")
countsPath = r"D:\DataSet\MULTI\bow" +"\\" + str(k) + r"mer"
# global finalDF
finalDF = pd.DataFrame(columns=features)
files = []

if __name__ == '__main__':
    # Get files based on pattern
    for file in get_files(directory=countsPath, pattern="*.txt"):
        # sum_size = sum_size + os.path.getsize(file)
        files.append(file)
    #reshape the array list into np array of six batches, each batch on a row
    files = np.array(files).reshape(6, 10000)

    for j in range(5,6):

        #free memory
        # n = gc.collect()
        # print("Number of unreachable objects collected by GC:", n)
        start = time.time()
        # global finalDF, change [processes=int] to precise the nb of threads, -1 for all available threads
        finalDF = pd_wrapper(files[j],processes=8)
        end = time.time()
        # total time taken
        print(f"Runtime of dataframe number : "+ str(j) + f"creation is {end - start} secs")
        pd.set_option("max_r", 50)
        pd.set_option("max_colwidth", 40)
        display("Final DF of kmers\n", finalDF)
        # Write DF into a feather binary format  for faster loading time
        finalDF.reset_index(drop = True).to_feather(r"D:\DataSet\MULTI\bow\10k-df-k"+str(k)+"part-"+ str(j)+".feather")
