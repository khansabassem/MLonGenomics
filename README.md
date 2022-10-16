# MLonGenomics
Exploratory research, in the context of Masters graduation project.  University of Applied Sciences of Western Switzerland (HES-SO MSE). Masters of Science in Engineering program. Complex Information Systems. 
#Master Thesis
[https://github.com/khansabassem/MLonGenomics/blob/main/Bassem%20El%20Khansaa_MSE_TM_SP21_00382d09a2fcb6be2167243348dd0bf5-1.pdf|link to Thesis]

# Techniques
Genomic sequence classification with two approaches:
1. NLP/BoW Bag of Words approach with classical machine learning algorithms
2. Deep Learning approach using LSTM/RNN models.

# Data: 
The data is artifitially generated, it consists of 60k DNA sequences, with length between 4999 and 14999 characters. Characters are {A,C,G,T}. 
The same techniques can be applied to any sort of genomic sequences given you have a sufficient computation power.


# Three types of scripts:
1.	Jupiter notebooks ipython format.
2.	Python scripts (for early, parallelized kmer counting methods developed using libraries that are incompatible with Jupiter notebooks).
3.	Bash scripts used to
a.	Run KMC kmer counter.
b.	Optionally delete intermediary files resulting from the three preprocessing phases. 
# Directories:
There are Four main directories:
1.	BoW: contains notebooks  & scripts of the first ML approach
2.	LSTM: contains notebooks of the LSTM approach
3.	LSTM on KMERS: contains notebooks of the LSTM approach
4.	Data: contains the raw sequences data.
