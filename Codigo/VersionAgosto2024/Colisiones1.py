# LAS COLISIONES SE DAN EN CENTIMETROS
import numpy as np
import math
from robot2 import*

from UR10IK import invKine
import cmath
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
###############################################################################
# Parametros
###############################################################################

mat=np.matrix
d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])#ur10 mm
a = mat([0 ,-0.612 ,-0.5723 ,0 ,0 ,0])#ur10 mm
alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0 ]) # ur10
volumenEmpacado = 0
volumenPromedio = 0
num_cajas_total= 0
error = 1e-4

HOLI = 0

###############################################################################
# Rotaciones como entendemos
###############################################################################

def AH(n,th):
    global mat, a, d, alph
    T_a = mat(np.identity(4), copy=False)
    T_a[0,3] = a[0,n-1]
    T_d = mat(np.identity(4), copy=False)
    T_d[2,3] = d[0,n-1]
    Rzt = mat([[cos(th[n-1]), -sin(th[n-1]), 0 ,0],
                [sin(th[n-1]), cos(th[n-1]), 0, 0],
                [0           ,0            , 1, 0],
                [0           ,0           ,  0, 1]],copy=False)
    Rxa = mat([[1, 0,                 0,                  0],
                [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
                [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
                [0, 0,                 0,                  1]],copy=False)
    A_i = T_d * Rzt * T_a * Rxa
    return A_i
def HTrans(th):  
    A_1=AH( 1,th)
    A_2=AH( 2,th)
    A_3=AH( 3,th)
    A_4=AH( 4,th)
    A_5=AH( 5,th)
    A_6=AH( 6,th)
        
    T_06=A_1*A_2*A_3*A_4*A_5*A_6

    return T_06

###########################################################pos_ff [####################
# Clases
###############################################################################

class ParteCilindro: # Tiene que estar en las coordenadas del pallet

    # Constructor

    def __init__(self, diametro, p1, p2, pallet): # diametro debe estar en centimetros, p1 y p2 deben ser en metros desde la base del robot y pallet es en metros desde la base delk robot
        toleranciaAltura = 0.01 / 2.0
        radio = diametro / 2.0

        # Convertir los puntos a coordenadas del pallet 0

        p1[0] -= pallet[0]
        p1[1] -= pallet[1]
        p1[2] -= pallet[2]
        p2[0] -= pallet[0]
        p2[1] -= pallet[1]
        p2[2] -= pallet[2]
        p1[0] *= 100
        p1[1] *= 100
        p1[2] *= 100
        p2[0] *= 100
        p2[1] *= 100
        p2[2] *= 100
        p1[0],p1[1] = -p1[0],-p1[1]
        p2[0],p2[1] = -p2[0],-p2[1]
        
        # Se determinan los nuevos p1 y p2 que tengan en cuenta la tolerancia

        vector = [p1[0] - p2[0],p1[1] - p2[1],p1[2] - p2[2]]
        modulo = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        vector[0] /= modulo
        vector[1] /= modulo
        vector[2] /= modulo
        p1 = [p1[i] + vector[i] * toleranciaAltura  for i in range(3)]
        p2 = [p2[i] - vector[i] * toleranciaAltura for i in range(3)]

        # Se determina el plano de una de las tapas del cilindro

        num = 16
        vectorArray = np.array(vector)
        anguloax=math.acos(np.dot(vectorArray,np.array([1,0,0])))
        anguloay=math.acos(np.dot(vectorArray,np.array([0,1,0])))
        anguloaz=math.acos(np.dot(vectorArray,np.array([0,0,1])))
        vect_sel=np.array([0,0,0])
        pi_med= math.pi/2.0
        if (abs(anguloax-pi_med)<=abs(anguloay-pi_med)and abs(anguloax-pi_med)<=abs(anguloaz-pi_med)):
            vect_sel[0]=1
        elif(abs(anguloay-pi_med)<=abs(anguloax-pi_med)and abs(anguloay-pi_med)<=abs(anguloaz-pi_med)):
            vect_sel[1]=1      
        else:
            vect_sel[2]=1           
        vector1= np.cross(vectorArray, vect_sel)
        nV = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2])
        vector1n = vector1
        vector1n[0] /= nV
        vector1n[1] /= nV
        vector1n[2] /= nV
        P1Array = np.array(p1)

        # Generar los puntos de la circunferencia en el plano de una tapa
        
        puntos_circunferencia= vector1n * radio
        angulos = np.linspace(0, 2*np.pi, num+1)[:-1]
        puntos_circunferencia1 = [P1Array + np.dot(self.calcular_matriz_rotacion2(vector, angulo), puntos_circunferencia)  for angulo in angulos]
        
        # Generar los puntos de la otra tapa
            
        P2Array = np.array(p2)
        puntos_circunferencia2 = [P2Array + np.dot(self.calcular_matriz_rotacion2(vector, angulo), puntos_circunferencia)  for angulo in angulos]
        todosPuntos = []
        todosPuntos.extend(puntos_circunferencia1)
        todosPuntos.extend(puntos_circunferencia2)
        self.esquinas = todosPuntos

        # Determinar el bounding box

        self.x1 = self.esquinas[0][0]
        self.x2 = self.esquinas[0][0]
        self.y1 = self.esquinas[0][1]
        self.y2 = self.esquinas[0][1]
        self.z1 = self.esquinas[0][2]
        self.z2 = self.esquinas[0][2]
        for p in self.esquinas:
            self.x1 = min(self.x1, p[0])
            self.x2 = max(self.x2, p[0])
            self.y1 = min(self.y1, p[1])
            self.y2 = max(self.y2, p[1])
            self.z1 = min(self.z1, p[2])
            self.z2 = max(self.z2, p[2])

    def calcular_matriz_rotacion(self,axis, angle): #Determina la matriz de rotacion en relacion a un eje y a un angulo, esto sirve para rotar puntos alrededor de ese eje
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        rotation_matrix = np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                                        [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s],
                                        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c)]])
        return rotation_matrix
    

    def calcular_matriz_rotacion2(self,axis, theta):
        axis = np.asarray(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
class ParteCaja:

    # Constructor

    def __init__(self, esq1, esq2, q, pallet): # Las esquinas estan en sistema de coordenadas del gripper en metros

        esquinas0 = np.array([[esq1[0] / 100.0, esq1[1] / 100.0, esq1[2] / 100.0,1],
                             [esq1[0] / 100.0, esq2[1] / 100.0, esq1[2] / 100.0, 1],
                             [esq2[0] / 100.0, esq2[1] / 100.0, esq1[2] / 100.0, 1],
                             [esq2[0] / 100.0, esq1[1] / 100.0, esq1[2] / 100.0, 1],
                             [esq1[0] / 100.0, esq1[1] / 100.0, esq2[2] / 100.0, 1],
                             [esq1[0] / 100.0, esq2[1] / 100.0, esq2[2] / 100.0, 1],
                             [esq2[0] / 100.0, esq2[1] / 100.0, esq2[2] / 100.0, 1],
                             [esq2[0] / 100.0, esq1[1] / 100.0, esq2[2] / 100.0, 1]])
        self.esquinas = []
        M0T6 = HTrans(q)
        for i in range(8):
            punto = np.eye(4)
            punto[:,3] = np.array([esquinas0[i,0], esquinas0[i,1], esquinas0[i,2], 1])
            resu = (M0T6@punto)[:,3].tolist()
            
            self.esquinas.append([resu[0][0], resu[1][0], resu[2][0]])  
        # print("q",np.rad2deg(np.array(q)))
        # print("M0T6",M0T6)
        # print("resu",resu)
        # Convertir a coordenadas del pallet
            
        for e in self.esquinas:
            e[0] = pallet[0] - e[0]
            e[1] = pallet[1] - e[1]
            e[2] -= pallet[2]
            e[0] *= 100
            e[1] *= 100
            e[2] *= 100

        # Determinar el bounding box

        self.x1 = self.esquinas[0][0]
        self.x2 = self.esquinas[0][0]
        self.y1 = self.esquinas[0][1]
        self.y2 = self.esquinas[0][1]
        self.z1 = self.esquinas[0][2]
        self.z2 = self.esquinas[0][2]
        for p in self.esquinas:
            self.x1 = min(self.x1, p[0])
            self.x2 = max(self.x2, p[0])
            self.y1 = min(self.y1, p[1])
            self.y2 = max(self.y2, p[1])
            self.z1 = min(self.z1, p[2])
            self.z2 = max(self.z2, p[2])
class Plano:

    #Constructor

    def __init__(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        p12 = [x2 - x1, y2 - y1, z2 - z1]
        p13 = [x3 - x1, y3 - y1, z3 - z1]
        self.A = p12[1] * p13[2] - p12[2] * p13[1]
        self.B = -(p12[0] * p13[2] - p12[2] * p13[0])
        self.C = p12[0] * p13[1] - p12[1] * p13[0]
        n = (self.A * self.A + self.B * self.B + self.C * self.C) ** 0.5
        if n > 1e-4:
            self.A /= n
            self.B /= n
            self.C /= n
            self.D = -x1 * self.A - y1 * self.B - z1 * self.C
            self.sirve = True
        else:
            self.A = 0
            self.B = 0
            self.C = 0
            self.D = 0
            self.sirve = False

    #Métodos

    def distancia(self, v):
        return self.A * v[0] + self.B * v[1] + self.C * v[2] + self.D
    def neg(self):
        A = -A
        B = -B
        C = -C
        D = -D
    def centrar(self, centro):
        if self.distancia(centro) < 0:
            self.neg()
class Aux:

    #Constructor

    def __init__(self, g, l, v1, v2):
        self.g = g
        self.l = l
        self.v1 = v1
        self.v2 = v2

    # Método
        
    def puntoLejano(self):
        ind = 0
        d = self.l.distancia(self.g[0])
        for i,v in enumerate(self.g):
            val = self.l.distancia(v)
            if val < d:
                d = val
                ind = i
        return ind

###############################################################################
# Clases Patron empaquetamiento
###############################################################################

class CajaEmpacada:

    # Cosntructor

    def __init__(self, esq1, esq2):
        self.volumen = (esq2[0] - esq1[0]) /100 * (esq2[1] - esq1[1]) /100 *(esq2[2] - esq1[2])/100
        self.esquinas = [esq1,
                        [esq1[0], esq2[1], esq1[2]],
                        [esq2[0], esq2[1], esq1[2]],
                        [esq2[0], esq1[1], esq1[2]],
                        [esq1[0], esq1[1], esq2[2]],
                        [esq1[0], esq2[1], esq2[2]],
                        esq2,
                        [esq2[0], esq1[1], esq2[2]]] # Esquinas en sistema de coordenadas de packing en centimetros

        # Bounding box

        self.x1 = self.esquinas[0][0]
        self.x2 = self.esquinas[0][0]
        self.y1 = self.esquinas[0][1]
        self.y2 = self.esquinas[0][1]
        self.z1 = self.esquinas[0][2]
        self.z2 = self.esquinas[0][2]
        for p in self.esquinas:
            self.x1 = min(self.x1, p[0])
            self.x2 = max(self.x2, p[0])
            self.y1 = min(self.y1, p[1])
            self.y2 = max(self.y2, p[1])
            self.z1 = min(self.z1, p[2])
            self.z2 = max(self.z2, p[2])
CajasEmpacadas = []

###############################################################################
# Funciones
###############################################################################

def translate_packing_corners(msg,pallet):

    ## Pallet es arreglo 

    temp = msg.split(';')

    decode=np.array(temp[1].split(','), dtype='int32')*10  #milimetros

    inf = decode[0:3]
    sup = decode[3:6]    

    inf[0] = pallet[0]- inf[0]
    inf[1] = pallet[1]- inf[1]
    inf[2] = pallet[2]+ inf[2]   

    sup [0] = pallet[0]- sup [0]
    sup [1] = pallet[1]- sup [1]
    sup [2] = pallet[2]+ sup [2]   
    
    sup_m = sup/1000 #metros
    inf_m = inf/1000 #metros

    # retornamos en metros.     
    return inf_m, sup_m
def determinarColision(cajasPallet, partesRobot):  # Colision entre partes y cajas empacadas
    global tolerancia
    hayColision = False
    for r in partesRobot:
        for c in cajasPallet:
            hayColision = True

            # Robot puntos Cajas planos
            for cp in c.planos:
                hayColision = False
                for rp in r.puntos:
                    if cp.distancia(rp) < -tolerancia:
                        #print(cp.distancia(rp))
                        hayColision = True
                        break
                if not(hayColision):
                    break
                
            # Robot planos Cajas puntos
            if hayColision:
                for rp in r.planos:
                    hayColision = False
                    for cp in c.esquinas:
                        if rp.distancia(cp) < -tolerancia:
                            #print(rp.distancia(cp))
                            hayColision = True
                            break
                    if not(hayColision):
                        break
            
            #Verificar si hay colision
            if hayColision:
                #print("colision entre la caja ubicada con esquinas")
                #print(c.esquinas)
                #print("Colisiona con la parte del robot tipo", r.tipo, "Con los siguientes puntos")
                #print(r.puntos)
                break
        if hayColision:
            break
    return hayColision

###############################################################################
# Codigo
###############################################################################

def igual(v1, v2, i):
    global error
    return abs(v1[i] - v2[i]) < error
def menor(v1, v2, i):
    global error
    return v1[i] + error < v2[i]
def DifMinkowsky(v1, v2):
        cx = 0
        cy = 0
        cz = 0
        v = []
        for i in v1:
            for j in v2:
                v.append([i[0] - j[0], i[1] - j[1], i[2] - j[2]])
                cx += v[-1][0]
                cy += v[-1][1]
                cz += v[-1][2]
        cx /= len(v)
        cy /= len(v)
        cz /= len(v)
        
        # Determinar los puntos del convex hull
                
        # print(v)        
        ch = ConvexHull(np.array(v))
        planes = ch.equations
        # print("planes")
        # print(planes)
        # Obtener la componente D de los planos

        planesD = []
        for p in planes:
            n = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
            if (p[0] * cx + p[1] * cy + p[2] * cz + p[3]) / n > 0:
                planesD.append(p[3] / n)
            else:
                planesD.append(-p[3] / n)

        # Determinar si está el origen dentro de la difMink

        toleranciaOverlapping = 0.5
        for d in planesD:
            if d < toleranciaOverlapping:
                return True
        return False
def Colisiona(parte, Cemp):
    if not(parte.x1 >= Cemp.x2 or parte.x2 <= Cemp.x1 or parte.y1 >= Cemp.y2 or parte.y2 <= Cemp.y1 or parte.z1 >= Cemp.z2 or parte.z2 <= Cemp.z1):
        return DifMinkowsky(parte.esquinas, Cemp.esquinas)
    return False
def HayColisiones(q,tamCaja,pallet): # El pallet está en mm
    global CajasEmpacadas, HOLI
    return False
    HOLI +=1
    print("Holi = ", HOLI)
    pallet0 = [0,0,0]
    pallet0[0] = pallet[0]/1000
    pallet0[1] = pallet[1]/1000
    pallet0[2] = pallet[2]/1000


    # Camara
    # Camara = ParteCaja([-15,-3,19],[-6,-13,16], q, pallet0)
    Camara = ParteCaja([-15,-3,17],[-6,-13,12], q, pallet0)
    print("Camara ", Camara.x1, " ", Camara.x2, " ", Camara.y1, " ", Camara.y2, " ", Camara.z1, " ", Camara.z2)
    # Verificar colisiones con la cámara
    
    for c in CajasEmpacadas:
        if Colisiona(c, Camara):
            return True
    
    # Sigue con las demas partes

    poses = posit(q)
    Partes = []

    # Cilindro grande del punto 2 a 3
    P1 = poses[1].tolist()
    P2 = poses[2].tolist()
    Partes.append(ParteCilindro(13, P1, P2, pallet0))

    # Cilindo pequeno 1 del punto 3 a 4
    P1 = poses[2].tolist()
    P2 = poses[3].tolist()
    Partes.append(ParteCilindro(11, P1, P2, pallet0))

    # Cilindo pequeno 2 del punto 4 a 5
    P1 = poses[3].tolist()
    P2 = poses[4].tolist()
    Partes.append(ParteCilindro(11, P1, P2, pallet0))

    # Cilindo pequeno del punto 5 a 6
    P1 = poses[4].tolist()
    P2 = poses[5].tolist()
    Partes.append(ParteCilindro(1, P1, P2, pallet0))

    # Gripper 
    Partes.append(ParteCaja([-15,-8,21],[15,8,18],q , pallet0))

    # Caja
    vals = tamCaja.split(",")
    tolCaja = 2
    valx = (float(vals[0]) - tolCaja) / 2.0
    valy = (float(vals[1]) - tolCaja) / 2.0
    Partes.append(ParteCaja([-valx, -valy, 19],[valx, valy, 19 + float(vals[2]) - 7], q, pallet0))

    #Verificación de colisiones
    for p in Partes:
        for c in CajasEmpacadas:
            if Colisiona(c, p):
                return True
    return False
def AdicionarCajaVirtual(esq1, esq2): # Las esquinas están en cm en coordenadas del pallet
    global CajasEmpacadas, volumenPromedio, volumenEmpacado,num_cajas_total
    CajasEmpacadas.append(CajaEmpacada(esq1, esq2))
    num_cajas_total= len(CajasEmpacadas)
    volumenEmpacado += CajasEmpacadas[-1].volumen
    volumenPromedio += (CajasEmpacadas[-1].volumen - volumenPromedio) / num_cajas_total
def GetVolumenEmpacado():
    global volumenEmpacado
    return volumenEmpacado
def GetVolumenPromedio():
    global volumenPromedio
    return volumenPromedio
def get_num_cajas():
    global num_cajas_total

    return num_cajas_total

###############################################################################
# Pruebas para borrar
###############################################################################

myq = [-1.3298,-2.43,-1.48,-0.79,1.56,0.32]
punto = np.array([14,6,0,1])
M0T6 = HTrans(myq)
#print("M0T6 = ", M0T6)

rot = Rotation.from_euler('xyz', [1.268, 1.2542, 1.2096], degrees=False)

# Obtener la matriz de transformación homogénea
matriz_transformacion = np.eye(4)
matriz_transformacion[:3, :3] = rot.as_matrix()
matriz_transformacion[:3, 3] = [-0.00325, -0.00067, 0.189]

M1T7 = np.dot(M0T6, matriz_transformacion)
matriz_4x4 = np.array([[1, 0, 0, punto[0]],   
                    [0, 1, 0, punto[1]],
                    [0, 0, 1, punto[2]],
                    [0, 0, 0, 1]])

#Aqui toca multiplicar la matrix por la de translacion
resu_partial= np.dot(M0T6, punto)
resu = [resu_partial[0,0], resu_partial[0,1], resu_partial[0,2]]
#print("resu = ",resu)