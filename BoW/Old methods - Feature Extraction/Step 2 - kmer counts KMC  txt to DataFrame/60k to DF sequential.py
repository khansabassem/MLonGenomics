#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from pandas import DataFrame
# import matplotlib.pyplot as plt
# from multiprocessing import Pool

from typing import Sequence
import itertools
import pytest
# ! pip install pyspark
from IPython.display import display
# import dask.dataframe as dd

#parallel
import psutil
import time
import os
from pathlib import Path


# ### Create all features given a k-mer length

# In[8]:


#Define Kmer size 
k=6
features = []

def generate_all_kmers(k: int, alphabet="CGAT") -> Sequence[str]:
    """generate all polymers of length k
    Args:
        k: length of the polymer
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

features = generate_all_kmers(k, alphabet="ACGT")


# In[9]:


len(features)

# fileNames = []

# for i in range(0, 60000):
#     s = 'kmers-' + str(k) +'-seqNb-'+ str(i) + '.txt'
#     fileNames.append(s)
# # print(fileNames)
pd.options.display.max_rows = 5
pd.options.display.max_columns = 10
# ### Import kmer count files and transform into dataframe

# In[38]:


import time
k=6
df = pd.DataFrame(columns=features)
display(df)
# countsPath = r"D:\DataSet\MULTI\bow\\" + str(k)+"mer"
countsPath = r"D:\DataSet\MULTI\bow\6mer\600"

start = time.time()
# i = 0
for i in range(0, 600):
    sample = pd.read_fwf(countsPath + r'\kmers-' + str(k) +'-seqNb-'+ str(i) + '.txt',sep=" ", header=None).T
    new_header = sample.iloc[0] #grab the first row for the header
    sample = sample[1:] #take the data less the header row
    sample.columns = new_header #set the header row as the df header
#     print(sample)
    df= df.append(sample, ignore_index=True)  #APPEND Sample to df DataSet
    display(df)

    
end = time.time()
# total time taken
print(f"Runtime of the program is {end - start} secs")


#     display(sample)
display(df)
# df.head
#
# sequential 600 rows  Runtime of the program is 956.6224915981293 secs
# Parallel   600 rows  Runtime of the program is 173.62739634513855 secs


# In[ ]:




# 600 rows sequential Runtime of the program is 956.6224915981293 secs

# In[1]:


# df.to_feather(r"D:\DataSet\MULTI\bow\df-k"+str(k)+".feather")


# In[2]:


# f.to_pickle(r"D:\DataSet\MULTI\100-6mers-DF")


# ### merge sample into single dataframe  using Dask
# ## Result: This approach will not work as Dask doesn't support inserting series with missing values (unlike pandas, which fills Nan for them).
# ### *ValueError: Length mismatch: Expected axis has 1455 elements, new values have 3072 elements

# In[3]:



# test = dd.read_csv(r'D:\DataSet\MULTI\test bow\raw count\kmers-6-seqNb-*.txt', sep="\t", header=0, assume_missing=True)#, names=features)#, header= 0)# header=[:][0])
# res=test.compute()
# display(res)#,res3,res4, res5)


# ## Transpose: This code transposes files, prerequisite for parallel approach.

# In[ ]:


# for i in range(10):
#     arr = np.genfromtxt(r"D:\DataSet\MULTI\test bow\raw count\kmers-6-seqNb-"+str(i)+".txt", dtype='U').T
#     arr
#     np.savetxt(r"D:\DataSet\MULTI\test bow\raw count\kmers-6-seqNb-"+str(i)+".txt", arr, delimiter="\t",fmt='%s')


# In[ ]:




