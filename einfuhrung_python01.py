# -*- coding: utf-8 -*-
"""
Created on HS2022
@author: dadams
"""

import numpy as np
# symbolisches Rechnen mit u,v,x,y,z
#from sympy import *
import sympy as sym
import scipy.linalg as sp


# eigene Funktionen
def eliminate(Aa_in,tolerance=np.finfo(float).eps*10.,fix=False,verbos=0):
    # eliminates first row
    # assumes Aa is np.array, As.shape>(1,1)
    Aa=Aa_in
    Nn = len(Aa)
    # Mm = len(Aa[0,:])
    if (Nn<2):
        return Aa
    else:
        if not fix:
            prof = np.argsort(np.abs(Aa_in[:, 0]))
            Aa = Aa[prof[::-1]]
        if np.abs(Aa[0, 0])>tolerance:
            el = np.eye(Nn)
            el[0:Nn,0] = -Aa[:, 0] / Aa[0, 0]
            el[0, 0] = 1.0 / Aa[0, 0]
            if (verbos>50):
                print('Aa \n', Aa)
                print('el \n', el)
                print('pr \n', np.matmul(el, Aa))
            return np.matmul(el, Aa)
        else:
            return Aa

def mrref(Aa_in,verbos=0):
    Aa=Aa_in*1.0
    Nn=len(Aa) ; kklist=np.arange(0, Nn - 1)
    for kk in kklist:
        Aa[kk:, kk:] = eliminate(Aa[kk:, kk:],verbos=verbos-1)
    Aa = np.flipud(Aa)
    for kk in kklist:
        Aa[kk::, Nn-kk-1::] = eliminate(Aa[kk::, Nn-kk-1::],fix=True,verbos=verbos-1)
    return  np.flipud(Aa)


# Test mrref
# Aa=np.array([[2,2,1,4],[1,9,2,3]])
# print('eliminate: \n',eliminate(Aa,fix=True,verbos=0))
# Aa=np.array([[3,2,1,4],[1,9,2,3],[1,6,6,0]])
# Alos=mrref(Aa,verbos=0)
# print(np.matmul(Aa[:,:3],Alos[:,-1])-Aa[:,-1])
#  init_printing(use_unicode=True)


#%% Polarkoordinaten zu kartesischen Koordinaten  NNHCXF 
# Geben Sie die kartesischen Koordinaten der Vektoren an. 
# [vx vy]=r*[cos(phi) sin(phi)]
# Befehle: np.array, np.cos, np.sin, print
# a)
r=5.0;phi=216.9 /360*np.pi*2 # in Grad
v=r*np.array([np.cos(phi), np.sin(phi)]) ; print(v)
#    -3.9984
#    -3.0021
#%% b)
r=13.0;phi=-0.4 # in Bogenmass
w=r*np.array([np.cos(phi), np.sin(phi)]); print(w)
#    11.9738
#    -5.0624

#%% Normierung eines Vektors 5VRS99 
#1) Berechnen Sie die Komponenten von u=AB
#2) Berechnen Sie die Komponenten von v=u*1/|u|
#3) Berechnen Sie |v|
# Befehle: np.array, np.linalg
 A=np.array([ 3 , 4]) ; B=np.array([ 6, 0])
 u=B-A ; print('u ', u)
#      3
#     -4
v=u*1/np.linalg.norm(u) ; print('v ', v)
#     0.6000
#    -0.8000
langev=np.linalg.norm(v) ; print('langev: ',langev)
#     1
  

#%% Richtung und Länge  SWI49N
#Geben sie die kartesischen Koordinten der folgenden Vektoren an:
# Befehle: np.linalg.norm
# a) Länge 10
af=[8,-0.5] 
an=af/np.linalg.norm(af)
a=10*an ; print(a)
# 9.9805   
# -0.6238

#%% b) Länge 5 
bs=[-33,56];
bn=bs/np.linalg.norm(bs)
b=10*bn ; print(b)


#%% Schatten, spitzer/stumpfer Zwischenwinkel   QHZIHW, JBARLL 
# 1) Berechnen Sie die Länge des Schattens von b auf a 
# 2) Zwischenwinkel  0<phi<90° (spitz) oder 90<phi<180° (stumpf)?
# 3) Geben Sie den Schatten von b auf a  als Vektor an.
# Befehle: np.matmul, np.linalg.norm
#a)  
a=np.array([3,-4 ])/5;b=[12,-1.0]
# 1)  
s=np.matmul(a,b)/np.linalg.norm(a)
print(s)
# 8
# 2) 8>0, also ist  0<phi<90°
# 3)  
svect=np.matmul(a,b)/np.matmul(a,a)*np.array(a)
print(svect)
#    4.8000
#    -6.400

#%% b)
a=[4 3];b=[ 12 -1]
#c)
a=[3 -4];b=[4  3]
#d)
a=[7 -24];b=[6.84 5.12]


#%% Skalarprodukt, Orthogonalität  891584 
# Bestimme die Vektoren in der Liste, die zu v orthogonal sind.
# Befehle: np.matmul
v=[ 1, 5, 2 ];
a=[   263, -35 ,-44]; b=[-121  ,15 , -48 ]; c=[71 ,5 ,-48 ];
print(np.matmul(v,a))
# 0 senkrecht
print(np.matmul(v,b))
# -142 nicht senkrecht
print(np.matmul(v,c))
# 0 senkrecht

#%% Winkel zwischen Vektoren 520784 
# Berechne den Winkel zwischen den Vektoren a und b.
# Befehle: np.arccos, np.matmul, np.linalg.norm, np.pi
# a)
a=[1, 1, -1 ];b=[ 1, -1, 1 ]
wib=np.arccos(np.matmul(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
print(wib)
# 1.9106 Bogenmass
# b) Geben Sie den Winkel in Grad an
wid=wib/(2*np.pi)*360 ; print(wid)
# 109.4712 Grad
#d)  Definieren Sie eine Funktion winkel(a,b), die den Winkel zwischen zwei Vektoren berechnet
def winkel(a,b):
    return np.arccos(np.matmul(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
print(winkel(a,b))

#% Berechnen Sie nun den Winkel zwischen a und b in Grad
a=[ 1, 1];b=[ 1, -1  ]
print(winkel(a,b)/(2*np.pi)*360)
#90

#%% Vektorprodukt in Orthonormalbasis   BT8J1D  
# Berechnen Sie die Vektorprodukte
# Befehle: np.cross
#%%a)
a=[-2 ,0 ,0] ; b=[0, 9 ,8] 
c=np.cross(a,b) ; print(c)
#    0
#     16
#    -18

#%% b)    
a=[1,5,0 ]; b=[ 0,7,0 ]
c=np.cross(a,b) ; print(c)
# 0 0 7 

#%% Fläche Dreieck   62FVCH 
#Berechne die Fläche des Dreiecks mit den Ecken A, B und C 
# Befehle: np.cross, np.linalg.norm(
#a)
A=np.array([  0 ,4 ,2 ]); B=[0 , 8 , 5 ]; C=[ 0 , 8, -1 ];
b=C-A
c=B-A
Fg=np.cross(b,c)
F=np.linalg.norm(Fg)/2.0 ; print(F)
#12

#%% Abstand Punkt-Gerade   CJ1IXZ  
# Abstand zwischen Gerade X=A+la*v und Punkt B
# Befehle: cross, norm, abs
#a)
A=np.array([3,0,0]);v=[2,0,0];B=[5,10, 0]
h=np.linalg.norm(np.cross(v,B-A))/np.linalg.norm(v)
print(h)
# 10 

#%% Kollinear   RIMDII 
#Bestimme ob die Vektoren kollinear sind, indem du die erste Komponente eliminierst.
# Befehle: mrref
#a)
u=[-3,2];v=[ 2, -3];
mat= np.array([u , v])# enthält beide Vektoren
#  -3     2
#  2    -3
print('mat: ',mat)
mats=mrref(mat) # Gauss-Elimination durchführen
print('mats: ',mats)
# Keine Nullzeile also nicht kollinear

#%%b)
u=[-3,2];v=[6,-4];
mat= np.array([u , v])# enthält beide Vektoren
#  -3     2
#  2    -3
print('mat: ',mat)
mats=mrref(mat) # Gauss-Elimination durchführen
print('mats: ',mats)
# Nullzeile also kollinear


#%% Gauss-Verfahren: Gleichungen lösen  K9C5RL 
#Bestimmen Sie  die Dreiecksform mit dem Gaussverfahren.
# Lösen Sie dann das Gleichungssystem durch Einsetzen von unten nach oben.
# Befehle: np.matmul
#   x  - 4 y - 2 z =  -25  
#      - 3 y + 6 z  = -18
# 7 x - 13 y - 4 z =  -85

#   x     y     z  =   d 
mat=np.array([[  1 ,   - 4 , -2  , -25 ], 
        [ 0 ,   - 3 ,  6  , -18    ],
       [ 7  ,  - 13 , -4  , -85]])
mats=mrref(mat)
print(mats)
#      x     y     z  =  d 
#      1     0     0    -1
#      0     1     0     6
#      0     0     1     0
# Lösung x=-1 ; y=6 ; z=0


#%% Komplanare Vektoren ACTUPR 
#Entscheide, ob die Vektoren komplanar sind. 
#Falls ja: Welche Linearkombination ergibt den Nullvektor $\vect{0}$?
# Befehle: np.zeros, np.array([a,b,c]).T , np.eye, mrref
#a)

# print(mat)
# #   3     2     0     1     0     0
# #   0     4     3     0     1     0
# #   3    10     6     0     0     1
# mats=mrref(mat)
# [[ 1. 0. -0.5  0.41666667 0. -0.08333333]
#  [ 0. 1. 0.75 -0.125     0. 0.125 ]
#  [ 0. 0. 0.    0.5       1. -0.5 ]]
# # Die Vektoren sind linear abhängig
# # Die Linearkombination ist 0= 0.5*a + 1*b-0.5c
a=[3 ,2 ,0] ; b=[0, 4 ,3]; c=[3, 10, 6]
mat=np.zeros([3,6])
mat[:,:3]=np.array([a,b,c])
mat[:,3:]=np.eye(3)
print('mat:\n',mat)
mats=mrref(mat)
print('mats: ',mats)
# Nullzeile also kollinear


#%% b)
u=[3 ,0 ,-1] ; v=[ 0 , 4, 3] ; w=[15, -4 ,-7]
mat=np.zeros([3,6])
mat[:,:3]=np.array([u,v,w])
mat[:,3:]=np.eye(3)
print('mat:\n',mat)
mats=mrref(mat)
print('mats: ',mats)
# Die Vektoren sind linear unabhängig

#%% c)
a=[3, 10 6] ; b=[ 6 ,0 ,-3] ; c=[6 ,7 ,3]

#d)
u=[3 ,8, 5] ; v=[ 6, -4, -5] ; w=[6 ,12 ,7]


#%% Lösung eines LGS   L3YGQD  
# Überprüfen Sie, ob die angegebenen Lösungen die linearen Gleichungssysteme erfüllen. 
# Befehle: np.matmul
#a)        5 x + 2 y   = 65
u=[ 11 ,5];v=[ 7 ,14] ; w=[15 ,-5] 
L1= [5 ,2] 
print(np.matmul(L1,u)) #=5*11+ 2*5
#    65: u ist Lösung
print(np.matmul(L1,v))
#   63: v ist keine Lösung
print(np.matmul(L1,w))
#   65: w ist Lösung

#%% b) 5 x + 2 y  =5 
#   3 x + 2 y  =7 
mat=[[ 5 , 2 ],[      3 , 2  ]]
u=[ 0 ,0] ; v=[ -1, 5] ; w=[ 5 ,-1]
print(np.matmul(mat,u))
#      0
#      0
# keine Lösung

print(np.matmul(mat,v))
#      5
#      7
# ist Lösung

print(np.matmul(mat,w))
#     23
#     13
# keine Lösung 


#%% c) 5 x + 2 y   +z   =4 
#    x  + 3 y   +2z  =-1 
u=[ 3 , -18 ,25] ; v=[ 1 ,0 ,-1 ] ; w =[5 ,9, -1]
mat=[[ 5 , 2 , 1],[   1 ,3,2  ]];
print(np.matmul(mat,u))
#     4
#     -1
# Ist Lösung

print(np.matmul(mat,v))
#      4
#     -1
# ist Lösung

print(np.matmul(mat,w))
#     42
#     30
# ist keine Lösung

#%% d) -2 x        +z =-7 
#      x    +y  +2z=34 
#      x        + z=17
u=[0,1,17] ; v=[1,0,-1] ; w =[8,8,9]
  
#%% Gleichungen mit Vektoren ITFSBL
#Bestimme x, so dass folgende Gleichungen erfüllt sind.

# a) u-v-x=x+v+x
# b) u-v-x=4x-v+x
# Befehle: sym.symboles, sym.solve
u, v , x, y, z = sym.symbols('u v x y z ')
sola=sym.solve(u-v-x-(x+v+x),x);print('sola: ',sola)
solb=sym.solve(u-v-x-(4*x-v+x),x);print('solb: ',solb)

#%% 
u=np.array([1 , 5 ] ); v=np.array([3 , -4]);

losa=u/3 - (2*v)/3  ;print('losa: ',losa)
#   -1.6667 ,   4.3333 

losb=u/6 ;print('losb: ',losb)
# 0.1667, 0.8333
