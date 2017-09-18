'''
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import numpy as np

               
def simple_lrp(R, input_tensor, biases, weights):
 
    Z = np.expand_dims(weights, 0) * np.expand_dims(input_tensor, -1)
    Zs = np.expand_dims(np.sum(Z, 1), 1) + np.expand_dims(np.expand_dims(biases, 0), 0)
    stabilizer = 1e-8*(np.where(np.greater_equal(Zs,0), np.ones_like(Zs, dtype=np.float32), np.ones_like(Zs, dtype=np.float32)*-1))
    Zs += stabilizer
                
    return np.sum((Z / Zs) * np.expand_dims(R, 1),2)

    
def flat_lrp(R, input_tensor, biases, weights):
    '''
    distribute relevance for each output evenly to the output neurons' receptive fields.
    note that for fully connected layers, this results in a uniform lower layer relevance map.
    '''
    Z = np.ones_like(np.expand_dims(weights, 0))
    Zs = np.sum(Z, 1, keepdims=True) 
    return np.sum((Z / Zs) * np.expand_dims(R, 1),2)
                         
def ww_lrp(R, input_tensor, biases, weights):
    '''
    LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
    '''
    Z = np.square( np.expand_dims(weights,0) )
    Zs = np.expand_dims( np.sum(Z, 1), 1)
    return np.sum((Z / Zs) * np.expand_dims(R, 1),2)
        
def epsilon_lrp(R, input_tensor, biases, weights, epsilon):
    '''
    LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
    '''
    Z = np.expand_dims(weights, 0) * np.expand_dims(input_tensor, -1)
    Zs = np.expand_dims(np.sum(Z, 1), 1) + np.expand_dims(np.expand_dims(biases, 0), 0)
    Zs += epsilon * np.where(np.greater_equal(Zs,0), np.ones_like(Zs)*-1, np.ones_like(Zs))

    return np.sum((Z / Zs) * np.expand_dims(R, 1),2)

def alphabeta_lrp(R, input_tensor, biases, weights ,alpha):
    '''
    LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
    '''
    beta = 1 - alpha
    Z = np.expand_dims(weights, 0) * np.expand_dims(input_tensor, -1)

    if not alpha == 0:
        Zp = np.where(np.greater(Z,0),Z, np.zeros_like(Z))
        term2 = np.expand_dims(np.expand_dims(np.where(np.greater(biases,0),biases, np.zeros_like(biases)), 0 ), 0)
        term1 = np.expand_dims( np.sum(Zp, 1), 1)
        Zsp = term1 + term2
        Ralpha = alpha * np.sum((Zp / Zsp) * np.expand_dims(R, 1),2)
    else:
        Ralpha = 0

    if not beta == 0:
        Zn = np.where(np.less(Z,0),Z, np.zeros_like(Z))
        term2 = np.expand_dims(np.expand_dims(np.where(np.less(biases,0),biases, np.zeros_like(biases)), 0 ), 0)
        term1 = np.expand_dims( np.sum(Zn, 1), 1)
        Zsp = term1 + term2
        Rbeta = beta * np.sum((Zn / Zsp) * np.expand_dims(R, 1),2)
    else:
        Rbeta = 0

    return Ralpha + Rbeta
