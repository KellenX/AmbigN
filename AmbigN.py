#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'AmbigN is a classifier for IUPACD ambiguous DNA letters.'

__author__ = 'Kellen Xu'
__version__ = '0.1.0'


if __name__=='__main__':
    print("Welcome to use AmbigN.")

# useful imports
import logomaker as lm
import numpy as np
import pandas as pd

#==============================================================================================================================
# define legal ambiguous letters
try: 
    from Bio.Data.IUPACData import ambiguous_dna_values
    dna_vals = ambiguous_dna_values
    del dna_vals["X"]  # don't use letter "X"    
except ModuleNotFoundError: 
    print("No module named 'Bio'. Using saved values of ambiguous_dna_values.")
    dna_vals = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
                'M': 'AC', 'R': 'AG', 'W': 'AT', 'S': 'CG', 'Y': 'CT', 'K': 'GT',
                'V': 'ACG', 'H': 'ACT', 'D': 'AGT', 'B': 'CGT',
                'N': 'ACGT'
               }
except:
    print("Something else than Biopython went wrong.")

dna_letters = "".join(list(dna_vals.keys()))
dna_keys = dict([(v, k) for k, v in dna_vals.items()])  # swap key-value pairs
dna_keys["ACGT"] = "N"

#==============================================================================================================================
# my ambiguous DNA letter classifier, "ACGT"
def digitizeNt(s):
    if not s in dna_letters: raise ValueError('The arg is not a DNA letter.')
    coordinate = [0,0,0,0]
    S = dna_vals[s]
    quant = 1.0 / len(S)
    if "A" in S: coordinate[0] = quant
    if "C" in S: coordinate[1] = quant
    if "G" in S: coordinate[2] = quant
    if "T" in S: coordinate[3] = quant
    return np.array(coordinate)

def digitizeNseq(Seq):
    coordinates = [digitizeNt(s) for s in Seq]
    return np.array(coordinates)

Nt_dig_array = digitizeNseq(dna_letters)

def normalize4(Fr):
    "Unity-based normalization of four non-negative numbers."
    Fr = np.array(Fr)
    Sum = sum(abs(Fr))
    return np.array((0,0,0,0)) if Sum == 0 else Fr/Sum
        
def cloest_ambigous_letter(Fr, Top_N = 4, show_distance_difference = False):
    "Fined the top N representive ambiguous letter from ACGT frequency at a position."
    coordinate = normalize4(Fr)
    compare_array = coordinate - Nt_dig_array
    norm = np.sum(compare_array**2, axis = 1)**0.5
    topNnorm = sorted(norm)[0:Top_N]

    pos = np.empty(0)
    for val in sorted(set(topNnorm)):
        pos = np.append(pos, np.where(norm == val)[0])
    
    Amb = [dna_letters[i] for i in pos.astype(int)]
    if not show_distance_difference:
        return pos
    else:
        norm = np.array([sum(np.square(coordinate - Nt_dig_array[dna_letters.find(i)])) for i in Amb])
        norm_diff = norm - norm[0]
        return Amb, norm_diff

#==============================================================================================================================
# Lower case letters means positive value/effect.
X_letters = dna_letters + dna_letters.lower()
nt_dig_array = Nt_dig_array * -1  # Nt_dig_array = digitizeNseq(dna_letters)
Xt_dig_array = np.concatenate((Nt_dig_array, nt_dig_array))

def digitizeXt(s):
    if not s in X_letters: raise ValueError('The arg is not a DNA+ letter.')
    coordinate = Xt_dig_array[X_letters.find(s)]
    return coordinate

def digitizeXseq(Seq):
    coordinates = [digitizeXt(s) for s in Seq]
    return np.array(coordinates)

# normalize negative input
def normalabs4(Fr):
    "Unity-based normalization of any four numbers."
    Fr = np.array(Fr)
    Sum = sum(abs(Fr))
    return np.array((0,0,0,0)) if Sum == 0 else Fr/Sum

def cloest_ambigous_abs(Fr, Top_N = 4, show_distance_difference = False):
    "Fined the top N representive ambiguous letter from ACGT frequency at a position."
    coordinate = normalabs4(Fr)  # normalabs4()
    compare_array = coordinate - Xt_dig_array
    norm = np.sum(compare_array**2, axis = 1)**0.5
    topNnorm = sorted(norm)[0:Top_N]

    pos = np.empty(0)
    for val in sorted(set(topNnorm)):
        pos = np.append(pos, np.where(norm == val)[0])
    
    Amb = [X_letters[i] for i in pos.astype(int)]
    if not show_distance_difference:
        return pos
    else:
        norm = np.array([sum(np.square(coordinate - Xt_dig_array[X_letters.find(i)])) for i in Amb])
        norm_diff = norm - norm[0]
        return Amb, norm_diff
    
#==============================================================================================================================
# Fined the closest ambiguous letter from ACGT frequency at a position
# for non-negative input only
# Without oupputing distance

def coordinates_checker(matrix):
    # Check data type
    if not isinstance(matrix, np.ndarray):
        if not isinstance(matrix, pd.core.frame.DataFrame):
            raise TypeError("Input is neither an Numpy ndarray nor a Pandas DataFrame.")  
    # Check data shape
    if matrix.shape[1] != 4:
        raise ValueError("Column number of the input matrix isn't 4.")
    pass


def first_ambigous_letter(coordinate):
    "coordinate must be normalized."
    compare_array = coordinate - Nt_dig_array
    dist2 = np.sum(np.square(compare_array), axis = 1)  # square of norm
    Amb = dna_letters[dist2.argmin()]
    return Amb

def first_ambigous_seq(Frs, input_normalized = False):
    """
    Take an N by 4 matrix.  4 digits along the 2nd axis correspond to 'A'. 'C', 'G' and 'T/U'.
    """
    coordinates_checker(Frs)  # check input matrix
    
    if input_normalized:
        coordinates = Frs
    else:
        coordinates = np.apply_along_axis(normalize4, 1, Frs)  # normalize input matrix
    Ambs = np.apply_along_axis(first_ambigous_letter, 1, coordinates)
    seq = "".join(Ambs)
    return seq

# Output distance values
def first_ambdist_letter(coordinate):
    "coordinate must be normalized."
    compare_array = coordinate - Nt_dig_array
    dist2 = np.sum(np.square(compare_array), axis = 1)  # square of norm
    index = dist2.argmin()
    Amb = dna_letters[index]
    return Amb, dist2[index]

def first_ambdist_seq(Frs, input_normalized = False):
    """
    Take an N by 4 matrix.  4 digits along the 2nd axis correspond to 'A'. 'C', 'G' and 'T/U'.
    Return the proximal ambiguous sequence as a string and the distances as an Numpy array.
    """
    coordinates_checker(Frs)  # check input matrix
    
    if input_normalized:
        coordinates = Frs
    else:
        coordinates = np.apply_along_axis(normalize4, 1, Frs)  # normalize input matrix
    tmp_arr = np.apply_along_axis(first_ambdist_letter, 1, coordinates)
    Ambs = tmp_arr[:,0].astype('U1')  # store as unicode string
    seq = "".join(Ambs) 
    dist2s = tmp_arr[:,1].astype(float)
    norms = np.sqrt(dist2s)  # turn square back to norm
    return seq, norms

# for both negative and positive input
# Without oupputing distance
def posneg_ambigous_letter(coordinate):
    "coordinate must be normalized."
    compare_array = coordinate - Xt_dig_array
    dist2 = np.sum(np.square(compare_array), axis = 1)  # square of norm
    Amb = X_letters[dist2.argmin()]
    return Amb

def posneg_ambigous_seq(Frs, input_normalized = False):
    """
    Take an N by 4 matrix.  4 digits along the 2nd axis correspond to 'A'. 'C', 'G' and 'T/U'.
    """
    coordinates_checker(Frs)  # check input matrix

    if input_normalized:
        coordinates = Frs
    else:
        coordinates = np.apply_along_axis(normalabs4, 1, Frs)  # normalize input matrix
    Ambs = np.apply_along_axis(posneg_ambigous_letter, 1, coordinates)
    seq = "".join(Ambs)
    return seq

# Output distance values
def posneg_ambdist_letter(coordinate):
    "coordinate must be normalized."
    compare_array = coordinate - Xt_dig_array
    dist2 = np.sum(np.square(compare_array), axis = 1)  # square of norm
    index = dist2.argmin()
    Amb = X_letters[index]
    return Amb, dist2[index]

def posneg_ambdist_seq(Frs, input_normalized = False):
    """
    Take an N by 4 matrix.  4 digits along the 2nd axis correspond to 'A'. 'C', 'G' and 'T/U'.
    Return the proximal ambiguous sequence as a string and the distances as an Numpy array.
    """
    coordinates_checker(Frs)  # check input matrix

    if input_normalized:
        coordinates = Frs
    else:
        coordinates = np.apply_along_axis(normalabs4, 1, Frs)  # normalize input matrix
    tmp_arr = np.apply_along_axis(posneg_ambdist_letter, 1, coordinates)
    Ambs = tmp_arr[:,0].astype('U1')  # store as unicode string
    seq = "".join(Ambs) 
    dist2s = tmp_arr[:,1].astype(float)
    norms = np.sqrt(dist2s)  # turn square back to norm
    return seq, norms

#==============================================================================================================================
# Intersection (inner) and union (outer) of two ambighous DNA sequences
# Only accept dna_letters

def DNAnt_letter_checker(a,b):
    a = a.upper(); b = b.upper()
    if not a in dna_letters: 
        raise ValueError('The first arg is not a DNA letter.')
    elif not b in dna_letters: 
        raise ValueError('The second arg is not a DNA letter.')
    else:
        return True

def DNAseq_letter_checker(dna1, dna2):
    "Return False if any input sequences contains illegal DNA letter. Only accept letters in Bio.Data.IUPACData.ambiguous_dna_letter ."
    dna1 = dna1.upper(); dna2 = dna2.upper()
    if not all(i in dna_letters for i in dna1): 
        raise ValueError('The dna1 contains illegal DNA letter.')
    elif not all(i in dna_letters for i in dna2): 
        raise ValueError('The dna2 contains illegal DNA letter.')
    else:
        return True

def dna_inner(a,b):
    "Only accept letters in Bio.Data.IUPACData.ambiguous_dna_letter ."
    inner = set(dna_vals[a]).intersection(set(dna_vals[b]))
    if len(inner) == 0:
        return "X"
    else:
        return dna_keys["".join(sorted(list(inner)))]

def dna_outer(a,b):
    "Only accept letters in Bio.Data.IUPACData.ambiguous_dna_letter ."
    outer = set(dna_vals[a]).union(set(dna_vals[b]))
    return dna_keys["".join(sorted(list(outer)))]

amb_dna_inner_dic, amb_dna_outer_dic = {}, {}  # create Cayley tables
for i in dna_letters:
    for j in dna_letters:
        amb_dna_inner_dic[i+j] = dna_inner(i,j)
        amb_dna_outer_dic[i+j] = dna_outer(i,j)
        
def amb_seq_inner(dna1, dna2):
    "Args dna1 and dna2 must be of equal length."
    inner = ""
    for a, b in zip(list(dna1), list(dna2)):
        inner = inner + amb_dna_inner_dic[a+b]
    return inner

def amb_seq_outer(dna1, dna2):
    "Args dna1 and dna2 must be of equal length."
    outer = ""
    for a, b in zip(list(dna1), list(dna2)):
        outer = outer + amb_dna_outer_dic[a+b]
    return outer
#==============================================================================================================================
#==============================================================================================================================
   