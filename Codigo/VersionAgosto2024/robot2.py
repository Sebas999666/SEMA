# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:19:50 2023

@author: Sebastian Valero
"""
#Librerias
import numpy as np
import math
from scipy.spatial.transform import Rotation

B=np.radians(-80.43)
C=np.radians(-180.75)
H=np.radians(-64.62)
M1=np.radians(-23.46)
M2=np.radians(87.73)
M3=np.radians(-78.81)
q=np.array([B,C,H,M1,M2,M3])

# Lista para guardar la pose del robot 

Pose=[]
Matrix=[]
TT=[]

###############################################################
# Funciones
 #Matriz de Transformacion
def Matriz__T(thetaa,a,d,alpha):
    T= np.array ([[np.cos(thetaa), -np.cos(alpha)*np.sin(thetaa), np.sin(alpha)*np.sin(thetaa), a*np.cos(thetaa)], 
               [np.sin(thetaa), np.cos(alpha)*np.cos(thetaa), -np.sin(alpha)*np.cos(thetaa), a*np.sin(thetaa)],
               [0, np.sin(alpha), np.cos(alpha), d],
               [0,0,0,1]])
    return T

 #Devuelve la matriz de transformacion 
def pos(joint,q):
    global TT
    #Parametros DH
    #a[m],d[m], alpha[rad]
    DH= np.array ([[0,  0.1273,  math.pi/2], 
                    [-0.612,  0, 0],
                    [-0.5723,  0,  0],
                    [0,  0.163941,  math.pi/2],
                    [0,  0.1157,  -math.pi/2],
                    [0,  0.0922,  0]])
    a=0
    d=0
    alpha=0
    thetas=0
    if(joint == 1):
         a= DH[0][0]
         d= DH[0][1]
         alpha= DH[0][2]
         thetas=q[0]
    elif(joint == 2):
         a= DH[1][0]
         d= DH[1][1]
         alpha= DH[1][2]
         thetas=q[1]
    elif(joint == 3):
         a= DH[2][0]
         d= DH[2][1]
         alpha= DH[2][2]
         thetas=q[2]
    elif(joint == 4):
         a= DH[3][0]
         d= DH[3][1]
         alpha= DH[3][2]
         thetas=q[3]
    elif(joint == 5):
         a= DH[4][0]
         d= DH[4][1]
         alpha= DH[4][2]
         thetas=q[4]
    elif(joint == 6):
         a= DH[5][0]
         d= DH[5][1]
         alpha= DH[5][2]
         thetas=q[5]
    #else:
         #print("joint fuera de rango")
         
    TT= Matriz__T(thetas,a,d,alpha)
    return TT
    #T_lim = np.around(TT, decimals=4)

# Genera la matriz de Transformacion entre dos puntos, ej 0T6
def  T_entre(num1, num2,q):
    global Pose
    matrices = []# Guardo cada una de las matrices de cada junta
    
    for i in range(num1, num2 + 1):
        TT = pos(i,q)#Cada una de las matrices para una posicion unica
        matrices.append(TT) 
        
    #print("matriz",matrices)
    resultado = matrices[0]
    for i in range(1, len(matrices)):
        resultado = np.dot(resultado,matrices[i])
        path = resultado [0:3, 3]#x,y,z de la matriz resultante
        Pose.append(path)#Coordenadas de las articulaciones en base del robot
    return resultado

def posit(q):
    global Pose
    Pose = []
    # Aplica la funcion de arriba
    for j in range(0,5):
        T_entre(0, j + 1,q)

    return Pose

# Ejecuta la funcion principal
b= T_entre(0,6,q)
# print("Pose",Pose)
# print("Resultado",b)
    
T_esquina1 = np.array([[1,0,0,-0.06],[0,1,0,-0.12],[0,0,1,0.2],[0,0,0,1]])

resultado1=b@T_esquina1
resultado2=T_esquina1@b 

# print(resultado1)
# print("JA")
# Tarea hallar bien los angulos de rotacion


#print(resultado_final)
#resultado= T_entre(num1, num2,q)
#mat_rot=resultado[:3, :3]
# Halla los angulkos de Euler
#angulos_rotacion = np.linalg.euler_angles_from_matrix(mat_rot, 'zyx')


#rotacion = Rotation.from_matrix(mat_rot)

# Obtener los ángulos de rotación
#angulos_rotacion = rotacion.as_euler('xyz')
#print(angulos_rotacion)

# Obtiene los angulos notacion vector rotacion
#vector_rotacion = rotacion.as_rotvec()

#print(vector_rotacion)

def ang(vector):
    transformation_matrix= []
    rotation_obj = R.from_matrix(transformation_matrix[:3, :3])
    mat_rot= rotation_obj.as_matrix()
    #mat_rot= Rotation.from_euler('zyx', vector_rad, degrees=False)
    rotacion = Rotation.from_matrix(mat_rot)
    # Obtener los ángulos de rotación
    angulos_rotacion = rotacion.as_euler('zyx')
    #print("angulos_rotacion",angulos_rotacion)

    vector_rotacion = rotacion.as_rotvec()

    #print("vector_rotacion",vector_rotacion)
    return vector_rotacion

    

    