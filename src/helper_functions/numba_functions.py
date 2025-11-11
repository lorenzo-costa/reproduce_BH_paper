from numba import jit
import numpy as np

@jit(nopython=True)
def compute_power(rejected, true_values):
    correct_rejections = 0
    total_non_nulls = 0
    n = len(rejected)
    for i in range(n):
        if true_values[i] != 0:
            total_non_nulls += 1
            if rejected[i] == 1:
                correct_rejections += 1

    return correct_rejections / total_non_nulls if total_non_nulls > 0 else 0.0
    
    
@jit(nopython=True)
def compute_fdr(rejected, true_values):
    false_positives = 0
    total_rejections = 0
    
    for i in range(len(rejected)):
        if rejected[i] == 1: 
            total_rejections += 1
            if true_values[i] == 0:
                false_positives +=1
            
    return false_positives / total_rejections if total_rejections > 0 else 0.0

@jit(nopython=True)
def compute_true_rejections(rejected, true_values):
    true_rej = 0.0
    for i in range(len(rejected)):
        if (rejected[i] == 1 ) and (true_values[i] != 0):
            true_rej += 1
    return true_rej

@jit(nopython=True)
def compute_total_rejections(rejected):
    # numba should be able to handle np.sum nefficiently by default but it does 
    # not hurt to spell loop out
    tot_rej = 0.0
    for i in range(len(rejected)):
        if rejected[i] == 1:
            tot_rej += 1
    return tot_rej
