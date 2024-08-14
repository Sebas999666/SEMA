#---------------------------------------------------------------------------------------
# Librerías
#---------------------------------------------------------------------------------------

import copy
import sys
import math
import socket
from functools import cmp_to_key

#---------------------------------------------------------------------------------------
# Variables globales
#---------------------------------------------------------------------------------------

r_juntarEspacios = 2 # 0: no juntar, 1: si juntar, 2: si juntar y expandir
r_estabilidad = 1 # 1: estabilidad parcial, 2: full support
r_esquinas = 2 # 0: 1 esquina, 1: 2 esquinas, 2: 4 esquinas
r_conocimiento = True # 0: no hay conocimiento, 1: se conocen tipos de caja
rx = 0 # 1: x se intercambia con z, 2: 1 y luego se gira 90� en z, 3: 1 y 2 
ry = 0 # 1: y se intercambia con z, 2: 1 y luego se gira 90� en z, 3: 1 y 2 
rz = 3 # 1: z se intercambia con z, 2: 1 y luego se gira 90� en z, 3: 1 y 2
tLx = 10 #Contenedor x
tLy = 10 #Contenedor y
tLz = 10 #Contenedor z

error = 0.0001
errorArea = 0
errorSoporte = 0.5

direccionIP = ""
puertoServidor = 0
r_conocimiento = True

#---------------------------------------------------------------------------------------
# Clases
#---------------------------------------------------------------------------------------

class Vec2:

    #Constructor

    def __init__(self, p_x, p_y):
        self.x = p_x
        self.y = p_y

    #Métodos

    def distanciaP(self, otrox, otroy):
        return abs(self.x - otrox) + abs(self.y - otroy)

    #Operadores

    def __eq__(self, otro):
        return self.x == otro.x and self.y == otro.y
    def __ne__(self, otro):
        return self.x != otro.x or self.y != otro.y
    def __lt__(self, otro):
        if self.x == otro.x:
            return self.y < otro.y
        return self.x < otro.x
class Linea:

    #Constructor

    def __init__(self, x1, y1, x2, y2):
        self.A = y1 - y2
        self.B = x2 - x1
        n = (self.A * self.A + self.B * self.B) ** 0.5
        if n > error:
            self.A /= n
            self.B /= n
            self.C = -x1 * self.A - y1 * self.B
        else:
            self.A = 0
            self.B = 0
            self.C = 0

    #Métodos

    def distancia(self, vx, vy):
        return self.A * vx + self.B * vy + self.C
    def neg(self):
        self.A = -self.A
        self.B = -self.B
        self.C = -self.C
class Caja:

    #Constructor

    def __init__(self, p_dx, p_dy, p_dz):
        self.dx = p_dx
        self.dy = p_dy
        self.dz = p_dz

    #Operadores
    
    def __eq__(self, otro):
        return self.dx == otro.dx and self.dy == otro.dy and self.dz == otro.dz
class TipoCaja:

    #Constructor

    def __init__(self, p_id, p_x0, p_y0, p_z0, p_masa):
        self.id = p_id
        self.volumen = p_x0 * p_y0 * p_z0
        self.masa = p_masa

        #Creación de las cajas

        self.cajas = []
        if rx == 1 or rx == 3:
            self.cajas.append(Caja(p_z0, p_y0, p_x0))
        if rx == 2 or rx == 3:
            self.cajas.append(Caja(p_y0, p_z0, p_x0))
        if ry == 1 or ry == 3:
            self.cajas.append(Caja(p_x0, p_z0, p_y0))
        if ry == 2 or ry == 3:
            self.cajas.append(Caja(p_z0, p_x0, p_y0))
        if rz == 1 or rz == 3:
            self.cajas.append(Caja(p_x0, p_y0, p_z0))
        if rz == 2 or rz == 3:
            self.cajas.append(Caja(p_y0, p_x0, p_z0))

        #Cajas únicas

        for i in range(len(self.cajas) - 1):
            ci = self.cajas[i]
            for j in range(len(self.cajas) - 1, i, -1):
                if ci == self.cajas[j]:
                    del self.cajas[j]
        
        #Mínimas dimensiones

        if (r_conocimiento):
            self.minDx = self.cajas[0].dx
            self.minDy = self.cajas[0].dy
            self.minDz = self.cajas[0].dz
            for i in range(1, len(self.cajas)):
                self.minDx = min(self.minDx, self.cajas[i].dx)
                self.minDy = min(self.minDy, self.cajas[i].dy)
                self.minDz = min(self.minDz, self.cajas[i].dz)
class EspacioMaximal:

    #Constructor

    def __init__(self, p_x1, p_y1, p_z1, p_x2, p_y2, p_z2):
        self.x1 = p_x1
        self.x2 = p_x2
        self.dx = p_x2 - p_x1
        self.y1 = p_y1
        self.y2 = p_y2
        self.dy = p_y2 - p_y1
        self.z1 = p_z1
        self.z2 = p_z2
        self.dz = p_z2 - p_z1
        self.vol = self.dx * self.dy * self.dz
        self.cambio = True
        self.indCajas = []
class Soporte:

    #Constructor

    def __init__(self, p_indPB, p_area, cx, cy):
        self.indPB = p_indPB
        self.area = p_area
        self.centrox = cx
        self.centroy = cy
        self.deltaMasa = 0
        self.masaSoporte = 0

    #Métodos

    def ActualizarSoporte(self, new_masa):
        self.deltaMasa = new_masa - self.masaSoporte
        self.masaSoporte = new_masa
class PolAux:

    #Constructor

    def __init__(self, p_grupo, p_linea, p_v1, p_v2):
        self.grupo = p_grupo.copy()
        self.linea = p_linea
        self.v1 = p_v1
        self.v2 = p_v2
    
    # Metodos

    def DeterminarAlejado(self):
        dAlejado = 0
        resp = 0
        for i, v in enumerate(self.grupo):
            d = self.linea.distancia(v.x, v.y)
            if d < dAlejado:
                dAlejado = d
                resp = i
        return resp
class Poligono:

    #Constructor

    def __init__(self, points):

        #Puntos únicos

        points.sort()
        pointsu = [points[0]]
        for i in range(1, len(points)):
            if points[i] != pointsu[-1]:
                pointsu.append(points[i])
        
        #Dos vértices extremos

        ch = [pointsu[0], pointsu[-1]]
        self.xmin = ch[0].x
        self.xmax = ch[1].x
        l12 = Linea(ch[0].x, ch[0].y, ch[1].x, ch[1].y)

        #Se dividen los puntos

        g1 = []
        g2 = []
        for v in pointsu:
            d = l12.distancia(v.x, v.y)
            if d < -error:
                g1.append(v)
            elif d > error:
                g2.append(v)

        #Se crea la lista de PalAux

        lista = []
        if len(g1) > 0:
            lista.append(PolAux(g1, l12, ch[0], ch[1]))
        if len(g2) > 0:
            l21 = Linea(ch[1].x, ch[1].y, ch[0].x, ch[0].y)
            lista.append(PolAux(g2, l21, ch[1], ch[0]))
        while len(lista) > 0:
            copia = copy.deepcopy(lista)
            lista.clear()
            for p in copia:
                ind = p.DeterminarAlejado()
                ch.append(p.grupo[ind])
                grupo1 = []
                grupo2 = []
                l1 = Linea(p.v1.x, p.v1.y, ch[-1].x, ch[-1].y)
                l2 = Linea(ch[-1].x, ch[-1].y, p.v2.x, p.v2.y)
                for v in p.grupo:
                    if l1.distancia(v.x, v.y) < -error:
                        grupo1.append(v)
                    elif l2.distancia(v.x, v.y) < -error:
                        grupo2.append(v)
                if len(grupo1) > 0:
                    lista.append(PolAux(grupo1, l1, p.v1, ch[-1]))
                if len(grupo2) > 0:
                    lista.append(PolAux(grupo2, l2, ch[-1], p.v2))
        
        # Lineas

        self.lineas = []
        self.ymin = ch[-1].y
        self.ymax = self.ymin
        for i in range(len(ch) - 1):
            v1 = ch[i]
            self.ymin = min(self.ymin, v1.y)
            self.ymax = max(self.ymax, v1.y)
            for j in range(i + 1, len(ch)):
                v2 = ch[j]
                maxD = v1.distanciaP(v2.x, v2.y)
                l = Linea(v1.x, v1.y, v2.x, v2.y)
                sirve = True
                for k in range(len(ch) - 1, -1, -1):
                    v3 = ch[k]
                    d = l.distancia(v3.x, v3.y)
                    if d < -error:
                        sirve = False
                        break
                    if k > j:
                        if abs(d) < error: #colineal
                            d = v1.distanciaP(v3.x, v3.y)
                            if maxD < d:
                                maxD = d
                                v2 = v3
                                ch[j] = v2
                            del ch[k]
                if sirve:
                    self.lineas.append(l)
                    ch[i + 1], ch[j] = ch[j], ch[i + 1]
                    break
        self.lineas.append(Linea(ch[len(ch) - 1].x, ch[len(ch) - 1].y, ch[0].x, ch[0].y))
    
    # Métodos

    def pundoDentroPoligono(self, cx, cy):
        if self.xmin <= cx and cx < self.xmax:
            if self.ymin <= cy and cy < self.ymax:
                for l in self.lineas:
                    if l.distancia(cx, cy) < errorSoporte:
                        return False
                return True
        return False
class CajaEmpacada:

    #Constructor

    def __init__(self, p_ind, p_tc, p_c, p_e, p_nEsquina, pbs):
        self.ind = p_ind
        self.id = p_tc.id
        self.volumen = p_tc.volumen
        self.masa = p_tc.masa
        self.dx = p_c.dx
        self.dy = p_c.dy
        self.dz = p_c.dz
        bfx = p_e.dx - p_c.dx
        bfy = p_e.dy - p_c.dy
        self.pf = 0
        if bfx < error:
            self.pf += 1
        if bfy < error:
            self.pf += 1
        self.esquina = p_nEsquina
        self.bf = p_e.dx - self.dx
        self.bf = min(self.bf,  p_e.dy - self.dy)
        if p_nEsquina == 1:
            self.x1 = p_e.x1
            self.x2 = self.x1 + self.dx
            self.y1 = p_e.y1
            self.y2 = self.y1 + self.dy
        elif p_nEsquina == 2:
            self.x2 = p_e.x2
            self.x1 = self.x2 - self.dx
            self.y1 = p_e.y1
            self.y2 = self.y1 + self.dy
        elif p_nEsquina == 3:
            self.x1 = p_e.x1
            self.x2 = self.x1 + self.dx
            self.y2 = p_e.y2
            self.y1 = self.y2 - self.dy
        elif p_nEsquina == 4:
            self.x2 = p_e.x2
            self.x1 = self.x2 - self.dx
            self.y2 = p_e.y2
            self.y1 = self.y2 - self.dy
        self.z1 = p_e.z1
        self.z2 = self.z1 + self.dz
        self.qEM = math.floor(p_e.dx / self.dx) * math.floor(p_e.dy / self.dy)
        if self.z1 == 0:
            self.soportada = True
        else:
            self.cmx = (self.x2 + self.x1) / 2.0
            self.cmy = (self.y2 + self.y1) / 2.0

            #Determinar soportes

            intersecciones = []
            self.soportes = []
            for ind in p_e.indCajas:
                p = pbs[ind]
                if not(self.x1 >= p.x2 or p.x1 >= self.x2):
                    if not(self.y1 >= p.y2 or p.y1 >= self.y2):
                        misx = [self.x1, p.x1, self.x2, p.x2]
                        misx.sort()
                        misy = [self.y1, p.y1, self.y2, p.y2]
                        misy.sort()
                        intersecciones.append(Vec2(misx[1], misy[1]))
                        intersecciones.append(Vec2(misx[2], misy[1]))
                        intersecciones.append(Vec2(misx[1], misy[2]))
                        intersecciones.append(Vec2(misx[2], misy[2]))
                        self.soportes.append(Soporte(ind, (misx[2] - misx[1]) * (misy[2] - misy[1]), (misx[1] + misx[2]) / 2.0, (misy[1] + misy[2]) / 2.0))
            
            #Determinar polígono

            if len(intersecciones) > 0:
                self.poligono = Poligono(intersecciones)
                if self.poligono.pundoDentroPoligono(self.cmx, self.cmy):
                    self.masaTotal = self.masa
                    self.ActualizarSoportes()
                else:
                    self.soportada = False
            else:
                self.soportada = False
    
    #Métodos

    def GaussianElimination(self, A, resp):
        
        # Obtener matriz triangular

        nF = len(A[0]) - 1
        j = 0
        for a_it in range(nF):
            
            # Encontrar el máximo valor absoluto de la columna hacia abajo

            a_max = max(range(a_it, len(A)), key=lambda i: abs(A[i][j]))
            valMax = abs(A[a_max][j])
            if valMax < errorArea:
                A = A[:j]
                return

            valMax = A[a_max][j]

            # Intercambiar fila actual con la de máximo valor

            if a_max != a_it:
                A[a_max], A[a_it] = A[a_it], A[a_max]

            # Dividir las siguientes filas y columnas según un factor para tener una matriz triangular

            for a_it2 in range(a_it + 1, len(A)):
                factor = A[a_it2][j] / valMax
                if abs(factor) < errorArea:
                    continue
                for a_it22, a_it12 in zip(A[a_it2][j+1:], A[a_it][j+1:]):
                    a_it22 -= factor * a_it12

            # Fila del pivote

            for a_it12 in A[a_it][j+1:]:
                a_it12 /= valMax

            j += 1

        A = A[:nF]

        # Obtener respuestas a partir de la matriz triangular

        resp = [0] * nF
        for a_it in reversed(range(len(A))):
            resp[a_it] = A[a_it][-1]
            for a_it2, r_it2 in zip(A[a_it][:-1], resp):
                resp[a_it] -= r_it2 * a_it2
    def GaussianElimination2(self, A, resp, nvalidas):
        
        # Recalcular filas nuevas (las viejas se dejan como están)

        for a_it in A[nvalidas:]:
            j = 0
            for a_it1 in A[:nvalidas]:
                factor = a_it[j]
                if abs(factor) < errorArea:
                    continue
                a_it12 = a_it1[j + 1:]
                for a_it2, a_it12_val in zip(a_it[j + 1:], a_it12):
                    a_it2 -= factor * a_it12_val

        # Seguir haciendo la matriz triangular de la misma manera pero para abajo

        nF = len(A[0]) - 1
        j = nvalidas
        for a_it in A[nvalidas:nF]:
            
            # Encontrar el máximo valor absoluto de la columna hacia abajo

            a_max = max(range(len(A)), key=lambda x: abs(A[x][j]), default=None)
            if a_max is None or abs(A[a_max][j]) < errorArea:
                print("Matriz Singular")
                resp = [100000000] * nF
                return
            valMax = A[a_max][j]

            # Intercambiar fila actual con la de máximo valor

            if a_max != j:
                A[a_max], A[j] = A[j], A[a_max]

            # Dividir las siguientes filas y columnas según un factor para tener una matriz triangular

            for a_it2 in A[j + 1:]:
                factor = a_it2[j] / valMax
                if abs(factor) < errorArea:
                    continue
                a_it12 = A[j][j + 1:]
                for a_it22, a_it12_val in zip(a_it2[j + 1:], a_it12):
                    a_it22 -= factor * a_it12_val

            # Fila del pivote

            for a_it12 in A[j + 1:]:
                a_it12 /= valMax

        A = A[:nF]

        # Obtener respuestas a partir de la matriz triangular

        resp = [0] * nF
        for a_it in reversed(range(len(A))):
            resp[a_it] = A[a_it][-1]
            for a_it2, r_it2 in zip(A[a_it][:-1], resp):
                resp[a_it] -= r_it2 * a_it2
    def ObtenerEcuacionesSoporte(self, resp): 
        A = []
        A1 = []
        A2 = []
        A3 = [1] * len(self.soportes)
        for s1 in self.soportes:
            A4 = []
            A5 = []
            for s2 in self.soportes:
                A4.append(s1.centrox - s2.centrox)
                A5.append(s1.centroy - s2.centroy)
            A4.append((s1.centrox - self.cmx) * self.masaTotal)
            A5.append((s1.centroy - self.cmy) * self.masaTotal)
            A.append(A4)
            A.append(A5)
            A1.append(self.cmx - s1.centrox)
            A2.append(self.cmy - s1.centroy)
        A1.append(0)
        A2.append(0)
        A3.append(self.masaTotal)
        A.append(A1)
        A.append(A2)
        A.append(A3)
        self.GaussianElimination(A, resp)
        if len(A) < len(A[0]) - 1:
            n_validas = len(A)
            n_Columnas = len(A[0])
            for s1, i in zip(self.soportes, range(len(self.soportes))):
                for s2, j in zip(self.soportes[(i + 1):], range(i + 1, len(self.soportes))):
                    A.append([0] * n_Columnas)
                    A[-1][i] = 1.0 / s1.area
                    A[-1][j] = -1.0 / s1.area
            self.GaussianElimination2(A, resp, n_validas)
    def ActualizarSoportes(self):
        resp = []
        self.ObtenerEcuacionesSoporte(resp)
        for s, r in zip(self.soportes, resp):
            s.masaSoporte = r
        self.soportada = True
    def centromasaEnPoligono(self):
        self.soportada = self.poligono.pundoDentroPoligono(self.cmx, self.cmy)
class ContAux:

    #Constructor

    def __init__(self, e, c):
        self.espacios = copy.deepcopy(e)
        self.ActualizarEspaciosMaximales(c)
        self.JuntarEspaciosMaximales()
        self.DeterminarMetrica()

    #Métodos

    def JuntarEspaciosMaximales(self):

        #Eliminar espacios contenidos

        for i in range(len(self.espacios) - 2, -1, -1):
            e1 = self.espacios[i]
            for j in range(len(self.espacios) - 1, i, -1):
                e2 = self.espacios[j]

                #Ver si e1 contiene a e2

                if e1.z1 <= e2.z1 and e2.z2 <= e1.z2:
                    if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                        if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                            del self.espacios[j]
                            continue
                            
                #Ver si e2 contiene a e1

                if e2.z1 <= e1.z1 and e1.z2 <= e2.z2:
                    if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                        if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                            del self.espacios[i]
                            break

        if r_juntarEspacios == 1: #Juntar
            seguir_juntando = True
            while seguir_juntando:
                seguir_juntando = False
                i = 0
                while i < len(self.espacios):
                    e1 = self.espacios[i]
                    j = len(self.espacios) - 1
                    while j > i:
                        e2 = self.espacios[j]

                        # Verificar si se pueden juntar

                        if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                            if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                    self.espacios[i].x1 = min(e1.x1, e2.x1)
                                    self.espacios[i].x2 = max(e1.x2, e2.x2)
                                    self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                    del self.espacios[j]
                                    seguir_juntando = True
                            elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                    self.espacios[i].y1 = min(e1.y1, e2.y1)
                                    self.espacios[i].y2 = max(e1.y2, e2.y2)
                                    self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                    del self.espacios[j]
                                    seguir_juntando = True
                        j -= 1
                    i += 1
        elif r_juntarEspacios == 2: #Juntar y expandir
            if r_estabilidad == 1:
                seguir_juntando = True
                while seguir_juntando:
                    seguir_juntando = False
                    i = 0
                    while i < len(self.espacios):
                        e1 = self.espacios[i]
                        j = len(self.espacios) - 1
                        while j > i:
                            e2 = self.espacios[j]

                            # Verificar si se pueden juntar

                            if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                                if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                    if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                        self.espacios[i].x1 = min(e1.x1, e2.x1)
                                        self.espacios[i].x2 = max(e1.x2, e2.x2)
                                        self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                        del self.espacios[j]
                                        seguir_juntando = True
                                elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                    if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                        self.espacios[i].y1 = min(e1.y1, e2.y1)
                                        self.espacios[i].y2 = max(e1.y2, e2.y2)
                                        self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                        del self.espacios[j]
                                        seguir_juntando = True

                            #Ver si e2 se puede expandir con e1

                            if e1.z1 <= e2.z1 and e2.z2 <= e1.z2:
                                if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                                    if not (e1.y1 > e2.y2 or e2.y1 > e1.y2):
                                        my = min(e1.y1, e2.y1)
                                        My = max(e1.y2, e2.y2)
                                        if e2.y1 != my or e2.y2 != My:
                                            self.espacios[j].y1 = my
                                            self.espacios[j].y2 = My
                                            self.espacios[j].dy = My - my
                                            seguir_juntando = True
                                if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                                    if not (e1.x1 > e2.x2 or e2.x1 > e1.x2):
                                        mx = min(e1.x1, e2.x1)
                                        Mx = max(e1.x2, e2.x2)
                                        if e2.x1 != mx or e2.x2 != Mx:
                                            self.espacios[j].x1 = mx
                                            self.espacios[j].x2 = Mx
                                            self.espacios[j].dx = Mx - mx
                                            seguir_juntando = True

                            #Ver si e1 se puede expandir con e2

                            if e2.z1 <= e1.z1 and e1.z2 <= e2.z2:
                                if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                                    if not (e2.y1 > e1.y2 or e1.y1 > e2.y2):
                                        my = min(e2.y1, e1.y1)
                                        My = max(e2.y2, e1.y2)
                                        if e1.y1 != my or e1.y2 != My:
                                            self.espacios[i].y1 = my
                                            self.espacios[i].y2 = My
                                            self.espacios[i].dy = My - my
                                            seguir_juntando = True
                                if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                                    if not (e2.x1 > e1.x2 or e1.x1 > e2.x2):
                                        mx = min(e2.x1, e1.x1)
                                        Mx = max(e2.x2, e1.x2)
                                        if e1.x1 != mx or e1.x2 != Mx:
                                            self.espacios[i].x1 = mx
                                            self.espacios[i].x2 = Mx
                                            self.espacios[i].dx = Mx - mx
                                            seguir_juntando = True
                            j -= 1
                        i += 1
            elif r_estabilidad == 2:
                seguir_juntando = True
                while seguir_juntando:
                    seguir_juntando = False
                    i = 0
                    while i < len(self.espacios):
                        e1 = self.espacios[i]
                        j = len(self.espacios) - 1
                        while j > i:
                            e2 = self.espacios[j]

                            # Verificar si se pueden juntar

                            if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                                if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                    if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                        self.espacios[i].x1 = min(e1.x1, e2.x1)
                                        self.espacios[i].x2 = max(e1.x2, e2.x2)
                                        self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                        del self.espacios[j]
                                        seguir_juntando = True
                                elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                    if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                        self.espacios[i].y1 = min(e1.y1, e2.y1)
                                        self.espacios[i].y2 = max(e1.y2, e2.y2)
                                        self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                        del self.espacios[j]
                                        seguir_juntando = True

                                #Ver si e2 se puede expandir con e1

                                if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                                    if not (e1.y1 > e2.y2 or e2.y1 > e1.y2):
                                        my = min(e1.y1, e2.y1)
                                        My = max(e1.y2, e2.y2)
                                        if e2.y1 != my or e2.y2 != My:
                                            self.espacios[j].y1 = my
                                            self.espacios[j].y2 = My
                                            self.espacios[j].dy = My - my
                                            seguir_juntando = True
                                if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                                    if not (e1.x1 > e2.x2 or e2.x1 > e1.x2):
                                        mx = min(e1.x1, e2.x1)
                                        Mx = max(e1.x2, e2.x2)
                                        if e2.x1 != mx or e2.x2 != Mx:
                                            self.espacios[j].x1 = mx
                                            self.espacios[j].x2 = Mx
                                            self.espacios[j].dx = Mx - mx
                                            seguir_juntando = True

                                #Ver si e1 se puede expandir con e2

                                if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                                    if not (e2.y1 > e1.y2 or e1.y1 > e2.y2):
                                        my = min(e2.y1, e1.y1)
                                        My = max(e2.y2, e1.y2)
                                        if e1.y1 != my or e1.y2 != My:
                                            self.espacios[i].y1 = my
                                            self.espacios[i].y2 = My
                                            self.espacios[i].dy = My - my
                                            seguir_juntando = True
                                if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                                    if not (e2.x1 > e1.x2 or e1.x1 > e2.x2):
                                        mx = min(e2.x1, e1.x1)
                                        Mx = max(e2.x2, e1.x2)
                                        if e1.x1 != mx or e1.x2 != Mx:
                                            self.espacios[i].x1 = mx
                                            self.espacios[i].x2 = Mx
                                            self.espacios[i].dx = Mx - mx
                                            seguir_juntando = True
                            j -= 1
                        i += 1
    
    def ActualizarEspaciosMaximales(self, c):
        if r_estabilidad == 1:
            i = len(self.espacios) - 1
            while i >= 0:
                e = self.espacios[i]
                if not (e.x1 >= c.x2 or e.x2 <= c.x1 or e.y1 >= c.y2 or e.y2 <= c.y1 or e.z1 >= c.z2):

                    # Espacios en x
                    if c.x1 > e.x1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, c.x1, e.y2, e.z2))
                    if e.x2 > c.x2:
                        self.espacios.append(EspacioMaximal(c.x2, e.y1, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en y
                    if c.y1 > e.y1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, e.x2, c.y1, e.z2))
                    if e.y2 > c.y2:
                        self.espacios.append(EspacioMaximal(e.x1, c.y2, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en z
                    if e.z2 > c.z2:
                        self.espacios[i] = EspacioMaximal(e.x1, e.y1, c.z2, e.x2, e.y2, e.z2)
                    else:
                        del self.espacios[i]
                i -= 1
        elif r_estabilidad == 2:
            i = len(self.espacios) - 1
            while i >= 0:
                e = self.espacios[i]
                if not (e.x1 >= c.x2 or e.x2 <= c.x1 or e.y1 >= c.y2 or e.y2 <= c.y1 or e.z1 >= c.z2):

                    # Espacios en x
                    if c.x1 > e.x1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, c.x1, e.y2, e.z2))
                    if e.x2 > c.x2:
                        self.espacios.append(EspacioMaximal(c.x2, e.y1, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en y
                    if c.y1 > e.y1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, e.x2, c.y1, e.z2))
                    if e.y2 > c.y2:
                        self.espacios.append(EspacioMaximal(e.x1, c.y2, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en z
                    if e.z2 > c.z2:
                        e = EspacioMaximal(c.x1, c.y1, c.z2, c.x2, c.y2, e.z2)
                    else:
                        del self.espacios[i]
                i -= 1
    def DeterminarMetrica(self):
        self.metrica = 0
        misz = []
        self.nz = 0
        for e in self.espacios:
            self.metrica += e.vol
            
            # Verificar que la z del espacio actual no esté

            agregar = True
            for z in misz:
                if z == e.z1:
                    agregar = False
                    break
            if agregar:
                misz.append(e.z1)
                self.nz += 1
        self.metrica /= len(self.espacios)
def OrdenClave(c):
    return (c.z1, -c.pf, c.nzEM, -c.qEM, c.bf, -c.VPEM, -c.y2)
class Contenedor:

    #Constructor

    def __init__(self, p_lx, p_ly, p_lz):
        self.Lx = p_lx
        self.Ly = p_ly
        self.Lz = p_lz
        self.espacios = [EspacioMaximal(0, 0, 0, p_lx, p_ly, p_lz)]
        self.cajas = []
        self.volumen = p_lx * p_ly * p_lz
        self.utilizacion = 0.0
        self.volumenCargado = 0.0
        self.maxZ = 0
        self.minDx = 0
        self.minDy = 0
        self.minDz = 0
        self.tiposCaja = []
        self.opciones = []

    #Métodos

    #MinD

    def SetTiposCajas(self, p_tc):
        self.tiposCaja = p_tc
        self.ActualizarMinD()
    def ActualizarMinD(self):
        if r_conocimiento:
            self.minDx = self.tiposCaja[0].minDx
            self.minDy = self.tiposCaja[0].minDy
            self.minDz = self.tiposCaja[0].minDz
            for t in self.tiposCaja[1:]:
                self.minDx = min(self.minDx, t.minDx)
                self.minDy = min(self.minDy, t.minDy)
                self.minDz = min(self.minDz, t.minDz)

    #Espacios Maximales

    def JuntarEspaciosMaximales(self):

        #Eliminar espacios contenidos

        for i in range(len(self.espacios) - 2, -1, -1):
            e1 = self.espacios[i]
            for j in range(len(self.espacios) - 1, i, -1):
                e2 = self.espacios[j]

                #Ver si e1 contiene a e2

                if e1.z1 <= e2.z1 and e2.z2 <= e1.z2:
                    if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                        if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                            del self.espacios[j]
                            continue
                            
                #Ver si e2 contiene a e1

                if e2.z1 <= e1.z1 and e1.z2 <= e2.z2:
                    if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                        if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                            del self.espacios[i]
                            break

        if r_juntarEspacios == 1: #Juntar
            seguir_juntando = True
            while seguir_juntando:
                seguir_juntando = False
                i = 0
                while i < len(self.espacios):
                    e1 = self.espacios[i]
                    j = len(self.espacios) - 1
                    while j > i:
                        e2 = self.espacios[j]

                        # Verificar si se pueden juntar

                        if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                            if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                    self.espacios[i].x1 = min(e1.x1, e2.x1)
                                    self.espacios[i].x2 = max(e1.x2, e2.x2)
                                    self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                    del self.espacios[j]
                                    seguir_juntando = True
                            elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                    self.espacios[i].y1 = min(e1.y1, e2.y1)
                                    self.espacios[i].y2 = max(e1.y2, e2.y2)
                                    self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                    del self.espacios[j]
                                    seguir_juntando = True
                        j -= 1
                    i += 1
        elif r_juntarEspacios == 2: #Juntar y expandir
            if r_estabilidad == 1:
                seguir_juntando = True
                while seguir_juntando:
                    seguir_juntando = False
                    i = 0
                    while i < len(self.espacios):
                        e1 = self.espacios[i]
                        j = len(self.espacios) - 1
                        while j > i:
                            e2 = self.espacios[j]

                            # Verificar si se pueden juntar

                            if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                                if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                    if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                        self.espacios[i].x1 = min(e1.x1, e2.x1)
                                        self.espacios[i].x2 = max(e1.x2, e2.x2)
                                        self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                        del self.espacios[j]
                                        seguir_juntando = True
                                elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                    if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                        self.espacios[i].y1 = min(e1.y1, e2.y1)
                                        self.espacios[i].y2 = max(e1.y2, e2.y2)
                                        self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                        del self.espacios[j]
                                        seguir_juntando = True

                            #Ver si e2 se puede expandir con e1

                            if e1.z1 <= e2.z1 and e2.z2 <= e1.z2:
                                if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                                    if not (e1.y1 > e2.y2 or e2.y1 > e1.y2):
                                        my = min(e1.y1, e2.y1)
                                        My = max(e1.y2, e2.y2)
                                        if e2.y1 != my or e2.y2 != My:
                                            self.espacios[j].y1 = my
                                            self.espacios[j].y2 = My
                                            self.espacios[j].dy = My - my
                                            seguir_juntando = True
                                if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                                    if not (e1.x1 > e2.x2 or e2.x1 > e1.x2):
                                        mx = min(e1.x1, e2.x1)
                                        Mx = max(e1.x2, e2.x2)
                                        if e2.x1 != mx or e2.x2 != Mx:
                                            self.espacios[j].x1 = mx
                                            self.espacios[j].x2 = Mx
                                            self.espacios[j].dx = Mx - mx
                                            seguir_juntando = True

                            #Ver si e1 se puede expandir con e2

                            if e2.z1 <= e1.z1 and e1.z2 <= e2.z2:
                                if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                                    if not (e2.y1 > e1.y2 or e1.y1 > e2.y2):
                                        my = min(e2.y1, e1.y1)
                                        My = max(e2.y2, e1.y2)
                                        if e1.y1 != my or e1.y2 != My:
                                            self.espacios[i].y1 = my
                                            self.espacios[i].y2 = My
                                            self.espacios[i].dy = My - my
                                            seguir_juntando = True
                                if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                                    if not (e2.x1 > e1.x2 or e1.x1 > e2.x2):
                                        mx = min(e2.x1, e1.x1)
                                        Mx = max(e2.x2, e1.x2)
                                        if e1.x1 != mx or e1.x2 != Mx:
                                            self.espacios[i].x1 = mx
                                            self.espacios[i].x2 = Mx
                                            self.espacios[i].dx = Mx - mx
                                            seguir_juntando = True
                            j -= 1
                        i += 1
            elif r_estabilidad == 2:
                seguir_juntando = True
                while seguir_juntando:
                    seguir_juntando = False
                    i = 0
                    while i < len(self.espacios):
                        e1 = self.espacios[i]
                        j = len(self.espacios) - 1
                        while j > i:
                            e2 = self.espacios[j]

                            # Verificar si se pueden juntar

                            if e1.z1 == e2.z1 and e1.z2 == e2.z2:
                                if e1.y1 == e2.y1 and e1.y2 == e2.y2:
                                    if not (e1.x1 > e2.x2 or e1.x2 < e2.x1):
                                        self.espacios[i].x1 = min(e1.x1, e2.x1)
                                        self.espacios[i].x2 = max(e1.x2, e2.x2)
                                        self.espacios[i].dx = self.espacios[i].x2 - self.espacios[i].x1
                                        del self.espacios[j]
                                        seguir_juntando = True
                                elif e1.x1 == e2.x1 and e1.x2 == e2.x2:
                                    if not (e1.y1 > e2.y2 or e1.y2 < e2.y1):
                                        self.espacios[i].y1 = min(e1.y1, e2.y1)
                                        self.espacios[i].y2 = max(e1.y2, e2.y2)
                                        self.espacios[i].dy = self.espacios[i].y2 - self.espacios[i].y1
                                        del self.espacios[j]
                                        seguir_juntando = True

                                #Ver si e2 se puede expandir con e1

                                if e1.x1 <= e2.x1 and e2.x2 <= e1.x2:
                                    if not (e1.y1 > e2.y2 or e2.y1 > e1.y2):
                                        my = min(e1.y1, e2.y1)
                                        My = max(e1.y2, e2.y2)
                                        if e2.y1 != my or e2.y2 != My:
                                            self.espacios[j].y1 = my
                                            self.espacios[j].y2 = My
                                            self.espacios[j].dy = My - my
                                            seguir_juntando = True
                                if e1.y1 <= e2.y1 and e2.y2 <= e1.y2:
                                    if not (e1.x1 > e2.x2 or e2.x1 > e1.x2):
                                        mx = min(e1.x1, e2.x1)
                                        Mx = max(e1.x2, e2.x2)
                                        if e2.x1 != mx or e2.x2 != Mx:
                                            self.espacios[j].x1 = mx
                                            self.espacios[j].x2 = Mx
                                            self.espacios[j].dx = Mx - mx
                                            seguir_juntando = True

                                #Ver si e1 se puede expandir con e2

                                if e2.x1 <= e1.x1 and e1.x2 <= e2.x2:
                                    if not (e2.y1 > e1.y2 or e1.y1 > e2.y2):
                                        my = min(e2.y1, e1.y1)
                                        My = max(e2.y2, e1.y2)
                                        if e1.y1 != my or e1.y2 != My:
                                            self.espacios[i].y1 = my
                                            self.espacios[i].y2 = My
                                            self.espacios[i].dy = My - my
                                            seguir_juntando = True
                                if e2.y1 <= e1.y1 and e1.y2 <= e2.y2:
                                    if not (e2.x1 > e1.x2 or e1.x1 > e2.x2):
                                        mx = min(e2.x1, e1.x1)
                                        Mx = max(e2.x2, e1.x2)
                                        if e1.x1 != mx or e1.x2 != Mx:
                                            self.espacios[i].x1 = mx
                                            self.espacios[i].x2 = Mx
                                            self.espacios[i].dx = Mx - mx
                                            seguir_juntando = True
                            j -= 1
                        i += 1
    def ActualizarEspaciosMaximales(self, c):
        if r_estabilidad == 1:
            i = len(self.espacios) - 1
            while i >= 0:
                e = self.espacios[i]
                if not (e.x1 >= c.x2 or e.x2 <= c.x1 or e.y1 >= c.y2 or e.y2 <= c.y1 or e.z1 >= c.z2):

                    # Espacios en x
                    if c.x1 > e.x1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, c.x1, e.y2, e.z2))
                    if e.x2 > c.x2:
                        self.espacios.append(EspacioMaximal(c.x2, e.y1, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en y
                    if c.y1 > e.y1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, e.x2, c.y1, e.z2))
                    if e.y2 > c.y2:
                        self.espacios.append(EspacioMaximal(e.x1, c.y2, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en z
                    if e.z2 > c.z2:
                        self.espacios[i] = EspacioMaximal(e.x1, e.y1, c.z2, e.x2, e.y2, e.z2)
                    else:
                        del self.espacios[i]
                i -= 1
        elif r_estabilidad == 2:
            i = len(self.espacios) - 1
            while i >= 0:
                e = self.espacios[i]
                if not (e.x1 >= c.x2 or e.x2 <= c.x1 or e.y1 >= c.y2 or e.y2 <= c.y1 or e.z1 >= c.z2):

                    # Espacios en x
                    if c.x1 > e.x1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, c.x1, e.y2, e.z2))
                    if e.x2 > c.x2:
                        self.espacios.append(EspacioMaximal(c.x2, e.y1, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en y
                    if c.y1 > e.y1:
                        self.espacios.append(EspacioMaximal(e.x1, e.y1, e.z1, e.x2, c.y1, e.z2))
                    if e.y2 > c.y2:
                        self.espacios.append(EspacioMaximal(e.x1, c.y2, e.z1, e.x2, e.y2, e.z2))

                    # Espacios en z
                    if e.z2 > c.z2:
                        e = EspacioMaximal(c.x1, c.y1, c.z2, c.x2, c.y2, e.z2)
                    else:
                        del self.espacios[i]
                i -= 1
    def EliminarEspaciosMaximales_MinDxMinDyMinDz(self):
        if r_conocimiento == 1:
            i = len(self.espacios) - 1
            while i >= 0:
                e = self.espacios[i]
                if e.dx < self.minDx or e.dy < self.minDy or e.dz < self.minDz:
                    del self.espacios[i]
                i -= 1
    def ActualizarIndicesCajasDeEspacios(self):
        if r_juntarEspacios == 0:
            for e in self.espacios:
                if e.z1 == 0:
                    continue 
                e.indCajas.clear()
                for i, p_it in enumerate(self.cajas):
                    if p_it.z2 == e.z1:
                        if not (p_it.x1 >= e.x2 or e.x1 >= p_it.x2):
                            if not (p_it.y1 >= e.y2 or e.y1 >= p_it.y2):
                                e.indCajas.append(i)
                                break
        else:
            for e in self.espacios:
                if e.z1 == 0:
                    continue
                e.indCajas.clear()
                for i, p_it in enumerate(self.cajas):
                    if p_it.z2 == e.z1:
                        if not (p_it.x1 >= e.x2 or e.x1 >= p_it.x2):
                            if not (p_it.y1 >= e.y2 or e.y1 >= p_it.y2):
                                e.indCajas.append(i)
        
    #Restricciones

    def Opciones_Tamanio(self, tc):
        n = 4  # Número de esquinas para soporte completo
        self.opciones.clear()
        for c_it in tc.cajas:
            for e_it in self.espacios:
                if c_it.dx <= e_it.dx and c_it.dy <= e_it.dy and c_it.dz <= e_it.dz:
                    for i in range(1, n+1):
                        c = CajaEmpacada(len(self.cajas), tc, c_it, e_it, i, self.cajas)
                        if c.soportada:
                            self.opciones.append(c)

        # Eliminar opciones iguales

        self.opciones.sort(key=lambda c: (c.x1, c.y1, c.z1, c.x2, c.y2, c.z2))
        i = 0
        while i < len(self.opciones) - 1:
            ci = self.opciones[i]
            j = i + 1
            while j < len(self.opciones):
                cj = self.opciones[j]
                if (ci.x1, ci.y1, ci.z1, ci.x2, ci.y2, ci.z2) == (cj.x1, cj.y1, cj.z1, cj.x2, cj.y2, cj.z2):
                    del self.opciones[j]
                    j -= 1
                else:
                    break
                j += 1
            i += 1

        # Calcular metricas de las opcioens válidas

        for c in self.opciones:
            miCont = ContAux(self.espacios, c)
            c.nzEM = miCont.nz # número de z1 de los espacios maximales
            c.VPEM = miCont.metrica # volumen promedio del espacio maximal
    def Priorizacion1(self):
        self.opciones.sort(key=OrdenClave)

    #Empacar

    def ActualizarArbol(self):
        arbol = [self.cajas[-1]]
        ini = 0
        indUltimo = 0
        miz1 = arbol[0].z1
        while arbol[indUltimo].z1 > 0:
            for s in arbol[indUltimo].soportes:
                s_pb = self.cajas[s.indPB]
                yaEsta = False
                for a_pb in arbol[indUltimo:]:
                    if s_pb.ind == a_pb.ind:
                        yaEsta = True
                        break
                if not yaEsta:
                    arbol.append(copy.deepcopy(s_pb))

            # Ordenar árbol de mayor a menor z1

            for i in range(indUltimo + 1, len(arbol) - 1):
                for j in range(i + 1, len(arbol)):
                    if arbol[j].z1 > arbol[i].z1:
                        arbol[i], arbol[j] = arbol[j], arbol[i]

            # Verificar si se actualizan los soportes

            indUltimo += 1
            zSiguiente = arbol[indUltimo].z1
            if zSiguiente != miz1:
                for i in range(ini, indUltimo):
                    pb_i = arbol[i]
                    pb_i.centromasaEnPoligono()
                    pb_i.ActualizarSoportes()
                ini = indUltimo
                miz1 = zSiguiente
    def Empacar(self, pos_Opcion):
        self.cajas.append(self.opciones[pos_Opcion])
        self.maxZ = max(self.maxZ, self.cajas[-1].z2)
        self.volumenCargado += self.cajas[-1].volumen
        self.utilizacion = self.volumenCargado / self.volumen
        self.ActualizarEspaciosMaximales(self.cajas[-1])
        self.JuntarEspaciosMaximales()
        self.EliminarEspaciosMaximales_MinDxMinDyMinDz()
        self.ActualizarIndicesCajasDeEspacios()
        self.ActualizarArbol()

    #Algoritmo constructivo

    def DeterminarOpciones(self, tc):
        self.Opciones_Tamanio(tc)
        # print(len(self.opciones))  # Comentario: descomenta si deseas imprimir el tamaño de opciones
        if len(self.opciones) > 0:
            self.Priorizacion1()
            # if random.randint(0, 99) < 50:  # Comentario: descomenta si deseas simular rand() % 100 < 50
            #     return 0
            return 1
        return -1
contenedores = []

#---------------------------------------------------------------------------------------
# Main 
#---------------------------------------------------------------------------------------

def Packing(p_ip, p_puerto, p_Lx, p_Ly, p_Lz, p_rz):
    
    # Parámetros

    tLx = p_Lx
    tLy = p_Ly
    tLz = p_Lz
    rz = p_rz
    puertoServidor = p_puerto
    direccionIP = p_ip

    # Mensaje de los parámetros
    
    if not (rz == 1 or rz ==2 or rz ==3):
        print("Packing: mal rotación rz")

    # Crear un socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock == socket.error:
        print("Error al crear el socket")
        sys.exit(0)

    # Conectar el socket al servidor

    server_addr = (direccionIP, puertoServidor)
    try:
        sock.connect(server_addr)
    except socket.error as e:
        print("Error al conectar con el servidor:", e)
        sock.close()
        sys.exit(0)

    contenedores.append(Contenedor(tLx, tLy, tLz))

    # Mensaje de saludo

    mensaje_saludo = "Packing;" + str(tLx) + ";" + str(tLy) + ";" + str(tLz)
    sock.sendall(mensaje_saludo.encode())

    # Ciclo infinito

    primerMensajeTamaniosCajas = False
    if r_conocimiento:
        primerMensajeTamaniosCajas = True
    while True:

        # Recibir datos del servidor

        buffer = sock.recv(1024)
        if not buffer:
            print("Error al recibir datos del servidor")
            continue

        # Procesar mensaje del servidor

        mensaje_in = buffer.decode()
        print("Datos recibidos: ", mensaje_in)
        if mensaje_in == "end":
            print("END")
            break
        elif mensaje_in:

            # Analizar mensaje de entrada

            if primerMensajeTamaniosCajas:
                primerMensajeTamaniosCajas = False
                tiposCajas = []
                lineas = mensaje_in.split(';')
                for linea in lineas:
                    val1, val2, val3 = map(int, linea.split(','))
                    tiposCajas.append(TipoCaja(len(contenedores[-1].cajas), val1, val2, val3, 0))
                contenedores[-1].SetTiposCajas(tiposCajas)
            else:
                if "," in mensaje_in:
                    val1, val2, val3 = map(float, mensaje_in.split(','))
                    val1, val2, val3 = int(val1), int(val2), int(val3)
                    tc = TipoCaja(len(contenedores[-1].cajas), val1, val2, val3, 0)
                    mensaje_out = str(contenedores[-1].maxZ) + ";"
                    miRespuesta = contenedores[-1].DeterminarOpciones(tc)
                    if miRespuesta == 1:  # Si hay opciones
                        for o_it in contenedores[-1].opciones:
                            mensaje_out += str(o_it.x1) + "," + str(o_it.y1) + "," + str(o_it.z1) + "," + str(o_it.x2) + "," + str(o_it.y2) + "," + str(o_it.z2) + ";"
                    elif miRespuesta == 0:  # Dejar pasar la caja
                        mensaje_out = "0"
                    elif miRespuesta == -1:  # No hay opciones
                        mensaje_out = "-1"
                    print("Datos enviados:", mensaje_out)
                    if sock.sendall(mensaje_out.encode()) == socket.error:
                        print("Error al enviar mensaje")
                else:
                    contenedores[-1].Empacar(int(mensaje_in))

    # Cerrar el socket

    sock.close()
def PackingNoConexion():
    global tLx, tLy, tLz, rz

    # Parámetros

    tLx = 48
    tLy = 54
    tLz = 100
    rz = 3

    # Crear contenedor

    contenedores.append(Contenedor(tLx, tLy, tLz))

    # Crear secuencias
    
    g = [21, 31, 16] #6
    m = [17, 20, 13] #9
    p = [13, 16, 13] #10
    mistipos=[]
    mistipos.append(TipoCaja(0, p[0], p[1], p[2], 0))
    mistipos.append(TipoCaja(1, m[0], m[1], m[2], 0))
    mistipos.append(TipoCaja(2, g[0], g[1], g[2], 0))
    contenedores[-1].SetTiposCajas(mistipos)
    secuencia = [g, g, m, p, p, m, m, p, m]

    # Ciclo

    for s in secuencia:
        tc = TipoCaja(len(contenedores[-1].cajas), s[0], s[1], s[2], 0)
        miRespuesta = contenedores[-1].DeterminarOpciones(tc)
        if miRespuesta == 1:
            contenedores[-1].Empacar(0)

#PackingNoConexion()

###########################################################################
# Parámetros
def intento():
    tLx = 33
    tLy = 40
    tLz = 100
    rz = 3
    puertoServidor = 2004
    direccionIP = "192.168.0.108"

    # Mensaje de los parámetros

    if not (rz == 1 or rz ==2 or rz ==3):
        print("Packing: mal rotación rz")

    # Crear un socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock == socket.error:
        print("Error al crear el socket")
        sys.exit(0)

    # Conectar el socket al servidor

    server_addr = (direccionIP, puertoServidor)
    try:
        sock.connect(server_addr)
    except socket.error as e:
        print("Error al conectar con el servidor:", e)
        sock.close()
        sys.exit(0)

    contenedores.append(Contenedor(tLx, tLy, tLz))

    # Mensaje de saludo

    mensaje_saludo = "Packing;" + str(tLx) + ";" + str(tLy) + ";" + str(tLz)
    sock.sendall(mensaje_saludo.encode())

    # Ciclo infinito

    primerMensajeTamaniosCajas = False
    if r_conocimiento:
        primerMensajeTamaniosCajas = True
    while True:

        # Recibir datos del servidor

        buffer = sock.recv(1024)
        if not buffer:
            #print("Error al recibir datos del servidor")
            continue

        # Procesar mensaje del servidor

        mensaje_in = buffer.decode()
        print("Datos recibidos: ", mensaje_in)
        if mensaje_in == "end":
            print("END")
            break
        elif mensaje_in:

            # Analizar mensaje de entrada

            if primerMensajeTamaniosCajas:
                primerMensajeTamaniosCajas = False
                tiposCajas = []
                lineas = mensaje_in.split(';')
                for linea in lineas:
                    val1, val2, val3 = map(int, linea.split(','))
                    tiposCajas.append(TipoCaja(len(contenedores[-1].cajas), val1, val2, val3, 0))
                contenedores[-1].SetTiposCajas(tiposCajas)
            else:
                if "," in mensaje_in:
                    val1, val2, val3 = map(float, mensaje_in.split(','))
                    val1, val2, val3 = int(val1), int(val2), int(val3)
                    tc = TipoCaja(len(contenedores[-1].cajas), val1, val2, val3, 0)
                    mensaje_out = str(contenedores[-1].maxZ) + ";"
                    miRespuesta = contenedores[-1].DeterminarOpciones(tc)
                    if miRespuesta == 1:  # Si hay opciones
                        for o_it in contenedores[-1].opciones:
                            mensaje_out += str(o_it.x1) + "," + str(o_it.y1) + "," + str(o_it.z1) + "," + str(o_it.x2) + "," + str(o_it.y2) + "," + str(o_it.z2) + ";"
                    elif miRespuesta == 0:  # Dejar pasar la caja
                        mensaje_out = "0"
                    elif miRespuesta == -1:  # No hay opciones
                        mensaje_out = "-1"
                    print("Datos enviados:", mensaje_out)
                    if sock.sendall(mensaje_out.encode()) == socket.error:
                        print("Error al enviar mensaje")
                else:
                    contenedores[-1].Empacar(int(mensaje_in))

    # Cerrar el socket

    sock.close()
# END