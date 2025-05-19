# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:25:07 2024

@author: donat.adams
"""

import numpy as np
# symbolisches Rechnen mit u,v,x,y,z
#from sympy import *
import sympy as sym
import scipy.linalg as sp

# eigene Funktionen
def eliminate(Aa_in, tolerance=np.finfo(float).eps*10., fix=False, verbos=0):
    # eliminates first row
    # assumes Aa is np.array, As.shape>(1,1)
    Aa = Aa_in
    Nn = len(Aa)
    # Mm = len(Aa[0,:])
    if (Nn < 2):
        return Aa
    else:
        if not fix:
            prof = np.argsort(np.abs(Aa_in[:, 0]))
            Aa = Aa[prof[::-1]]
        if np.abs(Aa[0, 0]) > tolerance:
            el = np.eye(Nn)
            el[0:Nn, 0] = -Aa[:, 0] / Aa[0, 0]
            el[0, 0] = 1.0 / Aa[0, 0]
            if (verbos > 50):
                print('Aa \n', Aa)
                print('el \n', el)
                print('pr \n', np.matmul(el, Aa))
            return np.matmul(el, Aa)
        else:
            return Aa


def FirstNonZero(lis):
    return next((i for i, x in enumerate(lis) if x), len(lis)-1)


def SortRows(Aa):
    inx = np.array(list(map(FirstNonZero, Aa)))
    #print('inx: ',inx,inx.argsort())
    return Aa[inx.argsort()]


def mrref(Aa_in, verbos=0):
    Aa = Aa_in*1.0
    Nn = len(Aa)
    kklist = np.arange(0, Nn - 1)
    #print('kklist', kklist)
    for kk in kklist:
        Aa[kk:, kk:] = eliminate(Aa[kk:, kk:], verbos=verbos-1)
    Aa = SortRows(Aa)
    Aa = np.flipud(Aa)
    # for kk in kklist:
    for kkh in kklist:
        kk = FirstNonZero(Aa[kkh, :])
        Aa[kkh::, kk::] = eliminate(Aa[kkh::, kk::], fix=True, verbos=verbos-1)
    return np.flipud(Aa)


def mnull(Aa_in,leps=np.finfo(float).eps*10,verbos=0):
    Aa=mrref(Aa_in) 
    Aa=Aa[list(map(np.linalg.norm,Aa  ))>leps] # extract non-zero linies
    mpiv=np.array(Aa[0]*0,dtype=bool)
    jj=0 # setup mask, indicating pivot-variables
    for ro in Aa>leps:
        for x in ro[jj:]:
            if x:
                mpiv[jj]=True
                jj=jj+1
                break
    
    jj=0 ; la=Aa[:,mpiv] ; veo=[]
    for jj in  np.argwhere(mpiv==False): 
        ve=np.linalg.lstsq(la, -Aa[:,jj],rcond=None)[0]
        vel=np.zeros((len(mpiv)))
        vel[mpiv]=ve[:,0] ; vel[jj]=1
        veo.append(vel)
 
    opt=np.array(veo).T
    if (verbos>10):
        print(Aa.shape,opt.shape)
        print('Test: ',np.matmul(Aa,opt))
    return opt 


# Test mrref
# Aa=np.array([[2,2,1,4],[1,9,2,3]])
# print('eliminate: \n',eliminate(Aa,fix=True,verbos=0))
# Aa=np.array([[3,2,1,4],[1,9,2,3],[1,6,6,0]])
# Alos=mrref(Aa,verbos=0)
# print('mrref:',np.matmul(Aa[:,:3],Alos[:,-1])-Aa[:,-1])
#  init_printing(use_unicode=True)
# Aae = np.array(
#     [[3, 3, -8, 6, 0, 14], [1, 1, -4, 2, 1, 3], [5, 5, -20, 10, 3, 13]])
# Aae = np.array([[3, 3, -8, 6, 0, 14], [1, 1, -4, 2, 1, 3]])
# Aaes = mrref(Aae)
# print(Aaes)
# print(np.matmul(Aae[:, :-1], [10, 0, 2, 0, 1]))
# riv = mnull(Aae)
# print(riv)
# Aae = np.array([[2, 1, -2, 7], [1, 8, -4, 20], [-3, 6, 0, 6]])
# Aaes = mrref(Aae)
# print(Aaes)
# print(np.matmul(Aae[:,:-1],[10,0,2,0,1]))