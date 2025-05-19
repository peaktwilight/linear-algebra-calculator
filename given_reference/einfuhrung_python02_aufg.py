# -*- coding: utf-8 -*-
"""
Created on HS2024
@author: dadams
"""
import os
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname) 
from core import mnull, mrref

# %% Geben Sie folgende Elemente der Matrix aus
# befehle np.array( ...).T, np.array(...).shape 
Aae = np.array([[4, 4, -16, 4], [1, 0, -4, 2], [5, 2, -20, 8]])
# Aae transponiert
print(Aae.T)
# Dimensionen der Matrix
nmA=Aae.shape   ; print('shape: ',nmA)
# das Element mit dem Index 1,3
print('Aae:\n', Aae)
print('Aae[1,3]: (python)', Aae[1,3])
print('Aae[1,3] (Tafel):', Aae[0,2])
# die erste zeile
print(Aae[0])
# die erste spalte
print(Aae[:,0])
# die letzte spalte
print(Aae.T[-1],' oder ',Aae[:,-1])
# Koeffizientenmatrix, d.h. Spalten 1 bis 3 
print(Aae[:,0:3])

# %% in Anlehnung an BSP 8.6  Schnittmengen qualitativ: Inhomogene LGS  EM75PE
# Bestimmen Sie die Schnittmengen der Ebenen E_1,  E_2 und E_3
# Befehle: mrref, mnull
# a)
# 2 x  +y  -2 z =  -1
#  x  +8 y  -4 z =  10
#  6 x  -y  +18 z =  81      # Schnittpunkt 2,3,4
Aae = np.array([[2, 1,-2,-1], [1,8,-4, 10], [6,-1,18,81]])
print(Aae)
Aaes=mrref(Aae)
print(Aaes)
print('Schnittpunkt: ' ,Aaes[:,-1])

# %% b)
# 2 x  +y  -2 z =  7
#  x  +8 y  -4 z =  20
#  -3 x  +6 y  +0 =  6     # Schnittgerade durch 4, 3, 3
Aae = np.array([[2, 1, -2, 7], [1, 8, -4, 20], [-3, 6, 0, 6]])
print(Aae)
Aaes=mrref(Aae)
print(Aaes)
print('Schnittpunkt: ' ,Aaes[:,-1])
riv=mnull(Aae[:,0:3]) # wir lösen homogene Gleichung
print('Richtungsv.: ' ,riv)
# Schnittgerade: [2.4 2.2 0. ]+s*[1.  , 0.5 , 1.25]

# %% c)
# 4 x  +18 y  -12 z =   62
#  2 x  +9 y  -6 z =   29
#  12 x  -3 y  +4 z =   67    # Parallele Ebenen
Aae = np.array([[4, 18, -12, 62], [2, 9, -6, 29], [12, -3, 4, 67]])
Aaes=mrref(Aae)
# das LGS ist inkonsistent 0*z=1
# das LGS hat keine Lösung

# %% d)
#  2 x  +y  -2 z =  9
#  6 x  -y  +18 z =  41
#  24 x  +8 y  +0 = 110    # Kamin
Aae = np.array([[2, 1, -2, 9], [6,-1,18,41], [24,8,0,110]])
Aaes=mrref(Aae)
print(Aaes)
# das LGS ist inkonsistent 0*z=1
# das LGS hat keine Lösung

#%% e)
#  0  +9 y  +8 z =  40
#  8 x  +9 y  +6 z =  56
#  10 x  +0  +6 z =  -14    # Schnittpunkt 1,8,-4
Aae = np.array([[0,9,8,40], [8,9,6,56], [10,0,6,-14]])

#%% f)
# -14 x  +2 y  +18 z =   78
#  -7 x  +y  +9 z =  -41
#  0  +4 y  +8 z =   9     # Parallele Ebenen 
Aae = np.array([[-14,2,18,78],[-7,1,9,-41],[0,4,8,9]])

# %% g)
# -7 x  +y  +9 z =  -31
#  0  +4 y  +8 z =  -12
#  14 x  +2 y  -10 z =   50   # Schnittgerade durch
Aae = np.array([[-7, 1, 9, -31], [0, 4, 8, -12], [14, 2, -10, 50]])



#%% h)
#  4 x  +3 y  -5 z =   18
#  -3 x  +3 y  +10 z =   5
#  5 x  +9 y  +0  = 36
Aae = np.array([[4,3,-5,18],[-3,3,10,5],[5,9,0,36]])

#%% i)
#   4 x  +18 y  -12 z =  62
#  2 x  +9 y  -6 z =  31
#  x  -3 y  +4 z =  2
Aae = np.array([[4,18,-12,64],[2,9,-6,31],[1,-3,4,2]])


# %% in Anlehnung an: BSP 8.6 Schnittmengen qualitativ: Homogene LGS  GL27L6
# Unten finden Sie Ebenen E1, E2 und E3
# Geben Sie die Schnittmenge der Ebenen an?
#%% a)
#  0   +9 y   +8 z  =   0
#  8 x +9 y   +6 z  =   0
#  10 x +0   +6 z  =   0
Aa = np.array([[0,9,8],[8,9,6],[10,0,6]])

#%% b)
#  -7 x   +y   +9 z  =   0
#  0   +4 y   +8 z  =   0
#  14 x   +2 y   -10 z  =   0
Aa = np.array([[-7,1,9],[0,4,8],[14,2,-10]])


#%%c)
#  -14 x   +2 y   +18 z  =   0
#  -7 x   +y   +9 z  =   0
#  0   +4 y   +8 z  =   0
Aa = np.array([[-14,2,18],[-7,1,9],[0,4,8]])

#%% d)
#  2 x  +  y   -2 z  =   0
#  x    +8 y   -4 z  =   0
#  -3 x +6 y   +0 =  0
Aa = np.array([[2,1,-2],[1,8,-4],[-3,6,0]])

# %% 8.22 Freie Variablen und Pivot-Variablen 863440
# Bestimmen Sie  freie Variablen und Pivot-Variablen und die Lösung des LGS.
# # a)
#    0 + 0  +2 z  +0   + 0 =  4
#    x + y  -4 z  +u   + 0 =  2
#  3 x + 3y -12 z +3 u + v =  7
Aae = np.array([[0,0,2,0,0,4],[1,1,-4,1,0,2],[3,3,-12,3,1,7]])


# %% b)
# 3 x  +3 y  -8 z   +6u   +0  = 14
#   x  +y   -4 z    +2u   +v = 3
# 5 x  +5 y  -20 z  +10u + 3 v = 13

Aae = np.array([[3, 3, -8, 6, 0, 14], [1, 1, -4, 2, 1, 3], [5, 5, -20, 10, 3, 13]])



# %% 9.7 Summe von Matrizen, Multiplikation mit Skalar   WEWKTK
A = np.array([[0, 1], [2, 3], [4, 2]])
B = np.array([[1, 2, 1], [2, 0, 3]])
C = np.array([[-8, 0, -4], [4, 6, 3], [0, -3, -1]])
D = np.array([[0, -4, -3], [7, 9, 2], [-1, 0, -5]])
E = np.array([-7, 0, 2])
F = np.array([0, 7, -6])
# Berechnen Sie folgenden Ausdrücke. Manche existieren nicht, wieso?
# Befehle: + - .*
# a) A + B.T

# b) 3 * B

# c) E + F

# d) E + F.T

# e) B * F

# f) B * F.T

# g) E * E

# h) F * E

# i) C * E

# j) C.T * C

# k) C * F


# %% Produkt von zwei Matrizen  CKZ2V1
A = [[0, 4, 2], [2, -1, 5], [6, 0, -3]]
B = [[4, 1, 9], [5, 4, 2], [-3, 5, 2]]
C = [[5, 6], [5, 1]]
D = [[-1, 3], [5, 0]]
E = [[7, -2, 4, 0], [3, 3, 0, 1]]
F = [[0, 9], [2, 0], [2, 4], [0, -2]]
# Berechne die Produkte
# Befehle: np.matmul, np.array
# a) A*B

# b) C *D

# c) E *F

# d) E *C


# %% Rechenregeln + Indexverschiebung für Summen  2W2RMM
# Berechnen Sie die Summen , Befehle :,
# np.arange np.sum np.power np.exp, np.log
# a) sum_{i=0}^{15}(5+3i)

# b) sum_{i=0}^{9} (50-5i)

# c) sum_{i=5}^{10}   (i*(i+1))/2

# d) sum_{i=0}^{12} (1+(i+3)^2)

# e) sum_{j=1}^18 3^(j-10)

# f) sum_{i=2}^8 (-5)^i

# %% BSP 9.30 Matrix mal Vektor = Einsetzen in Koeffizientenmatrix   XGQMA7
# Handelt es sich bei den Lösungen unten um partikuläre Lösungen, homogene Lösungen oder gar keine Lösung?                     \\
# Befehle: np.matmul np.array.T

# a)             x  y  z  = const , erweiterte Koeffizientenmatrix
Aae = np.array([[4, 4, -16, 4], 
                [1, 0, -4, 2], 
                [5, 2, -20, 8]])
 

u = np.array([2, -1, 0])

v = [0, 2, 1]

w = [4, 0, 1]


# %% b)          x  y  z  = const , erweiterte Koeffizientenmatrix
Aae = np.array([[3, 5, -1, 7], 
                [-1, 5, -6, 0]])

u = np.array([-25/20, 19/20, 1])

v = np.array([35/20, 7/20, 0])

w = np.array([1/20, 26/20, 1])



# %% in Anlehnung an BSP 7.25 Hessesche Normalenform 6ZXEAL

P = [10, 4, -12]
Q = [11.4, 0, 1.2]
R = [1,-20, -71]
S = [1, 8, 25]
lisp = np.array([P, Q, R, S])
# Ebene E ([ x, y,z]-[ 5,1, -2])* [4, 0 ,-3]=0
# Ebene F ([ x, y,z]-[ 1 ,1 ,1])* [4, 0 ,-3]=0
# Befehle: np.array, np.linalg.norm, np.abs, list, map
# a) Erstelle eine Funktion absE(x,y,z), die den Abstand eines Punktes [x,y,z] zu E berechnet
# Befehle: np.array, np.linalg.norm, np.abs
def absE(X):

    return hh

# b) Berechne damit den Abstand der Punkte P ... S zu E
# Befehle:  list, map

# c) Erstelle eine Funktion absF(X), die den Abstand eines Punktes X zu F berechnet

def absF(X):

    return hh

# d) Berechne damit den Abstand der Punkte P ... S zu E


# e) Abstand von E und F zum Ursprung?
