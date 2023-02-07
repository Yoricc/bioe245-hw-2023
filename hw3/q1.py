"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Helen Sakharova
Instructor: Liana Lareau
Assignment 3
"""

import numpy as np
from collections import Counter # might be helpful
from sklearn.preprocessing import normalize

# Constant used in this exercise
# Represents the width of the motif
width = 8

# Helper dictionary for indexing
nt_dict = {"A":0, "C":1, "G":2, "T":3}

# Extract strings of sequences from the fasta file
with open('motif-regions.fa', 'r') as f:
    #ignore lines with '>', remove '\n' from end of line
    sequences = [seq.split('\n')[0] for seq in f.readlines() if seq[0]!='>']

#Fill in all the functions below according to specifications
#Some structure is provided inside the functions to help: feel free to modify it 
#TODO statements mark where something is missing

def init_p(l, w, seqs, nt_dict):
    '''
    This function initializes weight matrix p_0, which represents our 
    initial guess for the profile of the motif. The dimensions of p_0
    are (4, w+1), where w is the width of the motif. (The first column
    of p_0 represents the background probability of each nucleotide, while 
    each subsequent column represents the probability of seeing each 
    nucleotide at that position in the motif.)

    The weight matrix p_0 is initialized by considering all possible motif 
    starting positions in every sequence and counting the number of times 
    each nucleotide appears in each position of a motif. The background 
    probability of each nucleotide is initialized as uniform. The weight matrix 
    p_0 is then normalized so that each column sums to 1.

    Inputs
    - l: The length of the sequence (assumes all sequences are the same length)
    - w: The width of the motif
    - seqs: A list of strings representing the sequences 
    - nt_dict: A dictionary mapping each nucleotide to its index

    Outputs
    - p_0: A matrix of size (4, w+1) representing the motif profile
    '''
    p_0 = np.zeros((4, w+1))

    # set a uniform background probability for each nucleotide
    for i in nt_dict.keys():
        p_0[nt_dict[i]][0] = ... #TODO

    # count nucleotides in all possible motifs for each sequence
    for sequence in seqs:
        # i in range(l-w+1) -> all possible motif starting positions
        for i in range(l-w+1):
            # j in range(w) -> position within the motif
            for j in range(w):
                # Fill in p_0
                ... #TODO
                
    # normalize columns to sum to 1
    p_0 = normalize(p_0, axis = 0, norm = 'l1')
    return p_0



def update_locations(l, w, p, seqs, nt_dict):
    '''
    This function updates z, a matrix representing the expected location of
    the motif in each sequence. 

    Inputs
    - l: The length of the sequence (assumes all sequences are the same length)
    - w: The width of the motif
    - p: A matrix of size (4, w+1) representing the motif profile
    - seqs: A list of strings representing the sequences 
    - nt_dict: A dictionary mapping each nucleotide to its index

    Outputs
    - z: A matrix of size(len(seqs), l-w+1) representing the expected location
            of the motif in each sequence. Z_ij == the probability that the motif
            starts at position j in sequence i. 
    '''
    z = np.zeros((len(seqs), l-w+1))

    #consider each sequence i
    for i, sequence in enumerate(seqs):
        #consider each possible starting position for the motif j
        for j in range(l-w+1):
            #a list of conditional probabilities
            p_vals = [] 
            #consider the nucleotide at each position in the sequence
            for position, nt in enumerate(sequence):
                #HINT: what is the probability of seeing nt at this position,
                #given that the motif begins at position j?
                #HINT: multiply each probability by 10 to avoid underflow errors (see pdf)
                ... #TODO
            #update Z_ij
            z[i][j] = ... #TODO
    
    # Normalize z_t so that the sum of each row is equal to 1
    z = normalize(z, axis = 1, norm = 'l1')
    return z



def update_locations_E_or_M():
    '''
    Is update_locations the expectation or maximization step of the EM algorithm?

    Have this function return 'E' for expectation or 'M' for maximization.
    '''
    return 'E' or return 'M' # TODO


def update_profile(l, w, z, seqs, nt_dict):
    '''
    This function updates the motif profile p, based on an updated matrix z of the
    expected locations of the motif in each sequence.

    Note: Use a pseudo-count of 1

    Inputs
    - l: The length of the sequence (assumes all sequences are the same length)
    - w: The width of the motif
    - z: A matrix of size(len(seqs), l-w+1) representing the expected location
            of the motif in each sequence. Z_ij == the probability that the motif
            starts at position j in sequence i.
    - seqs: A list of strings representing the sequences 
    - nt_dict: A dictionary mapping each nucleotide to its index

    Outputs
    - p: A matrix of size (4, w+1) representing the motif profile
    '''
    p = np.zeros((4, w+1))
    n = np.zeros((4, w+1))

    #Fill in n for k > 0
    for k in range(1, w+1):
        for nt in nt_dict.keys():
            sum_z = 0 #sum of relevant probabilities Z_ij
            for i, sequence in enumerate(seqs):
                #Add relevant probabilities Z_ij to sum_z
                ... #TODO
            #update n
            ... #TODO

    #Fill in n for k == 0
    ... #TODO

    #use n to calculate p
    #Pseudo-count = 1
    ... #TODO

    return p


def update_profile_E_or_M():
    '''
    Is update_profile the expectation or maximization step of the EM algorithm?

    Have this function return 'E' for expectation or 'M' for maximization.
    '''
    return 'E' or return 'M' # TODO

def run_EM(w, seqs, nt_dict, epsilon=2**-64):
    '''
    This function runs the Expectation Maximization algorithm for motif-finding.

    Hint: Use the previously defined functions init_p, update_locations, and update_motif

    Inputs
    - w: the width of the motif
    - seqs: A list of (same-length) strings representing the sequences
    - nt_dict: A dictionary mapping each nucleotide to its index
    - epsilon: A small value. Terminate the algorithm if the difference between p_t 
            and p_prev is smaller than epsilon for each cell in the matrix.

    Outputs
    - p_t: A matrix of size (4, w+1) representing the motif profile at the end of
            the EM algorithm.
    - z_t: A matrix of size(len(seqs), l-w+1) representing the expected location
            of the motif in each sequence at the end of the EM algorithm. 
    '''
    l = len(seqs[0]) #length of the sequence
    no_change = False #terminate algorithm when there is no change

    #initialize p
    ... #TODO
    while not no_change:
        #update z_t
        ... #TODO
        
        #update p_t
        ... #TODO

        #stop loop if the difference between p_t and p_prev is small enough
        ... #TODO
        
        #update p_prev
        ... #TODO

    return p_t, z_t



# Let's try out your code on a real example! Do not modify this segment!
if __name__ == '__main__':
    print('Running EM motif finder')
    p_final, z_final = run_EM(width, sequences, nt_dict, epsilon=0.0001)
    print('\nFinal p:')
    print(p_final)
    assert(p_final.shape == (4, width+1)), "dimensions of p_final are incorrect"
    assert(np.all(np.sum(p_final, axis=0) == 1.)), "columns of p_final do not sum to 1"
    print('\nFinal z:')
    print(z_final)
    assert(z_final.shape == (len(sequences), len(sequences[0])-width+1)), "dimensions of z_t are incorrect"
    assert(np.all(np.sum(z_final, axis=1) == 1.)), "rows of z_t do not sum to 1"
    # Note - if the rows of z don't add up to 1, you might be encountering an underflow error
    # try scaling the conditional probabilities you are multiplying together by a constant value to avoid this
    print('\nConsensus motif:')
    nt_lookup = {nt_dict[nt]:nt for nt in 'ACGT'}
    best_motif = "".join([nt_lookup[x] for x in np.argmax(p_final[1:], axis=0)])
    print(best_motif)
    print('\nPositions of motifs in sequence:')
    motif_indices = np.argmax(z_final, axis=1)
    print(motif_indices)
    print('\nMotif in each sequence:')
    print(len(sequences[0]))
    print(np.all([len(s)==len(sequences[0]) for s in sequences]))
    motifs = [sequences[i][pos:pos+width] for i,pos in enumerate(motif_indices)]
    print(motifs)

