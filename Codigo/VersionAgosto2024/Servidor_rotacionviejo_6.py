#Nueva trayectoria con Blend radious
# Funciona con vel lenta
# Librerias
#import psutil
import time 
import socket
import numpy as np
from _thread import *
import math 
import rtde_control
import rtde_receive
import rtde_io
import threading
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDERceive 
from scipy.spatial.transform import Rotation
from Colisiones1 import *
from UR10IK import invKine, HTrans
from datetime import datetime
from SEMAPython3 import Packing
# from functionMultisize import Vision
from subprocess import Popen, PIPE,run
##############################################

# getting the current date and time
current_datetime = datetime.now()

#############################################
           
# Connections


rtde_c = rtde_control.RTDEControlInterface("192.168.0.104")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.104")
rtde_i = rtde_io.RTDEIOInterface("192.168.0.104")

ServerSideSocket = socket.socket()
ServerSideSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
host = '192.168.0.108'
port = 2004
ThreadCount = 0
try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print('Socket is listening..')
ServerSideSocket.listen(5)

#HostLife
CleanUp=False

#ClientConditions
visionReady=False
trajectoryReady=False
packingReady=False

#Flags
RequestPacking=False
ReceivedPacking=False

# Shared Variables
robotQ=np.zeros(6)
msg_packing=" "
posicionEscogida = 0
msg_vision= " "
last_vision=" "
send_message = " "
largo= 21#22
ancho= 31#32
alto= 15#15
tamCaja=str(largo)+","+str(ancho)+","+str(alto)

# Environment
#tipos de caja que se le envian a Packing
boxSizes=np.array([[20,30,15],[16,19,12],[12,15,12]])
#boxSizes=np.array([[20,30,15]])
calibrationArea=21400
leeway=1
areas=boxSizes[:,0]*boxSizes[:,1]*(60.6+boxSizes[:,2]-boxSizes[0,2])/60.6
pixelAreas=calibrationArea*areas/areas[0]
boxSizes[:,:2]= boxSizes[:,:2]+leeway
TC = ""

tamArrayNumeric=boxSizes.tolist()#[[20,30,15],[15,20,12],[12,15,12]]
tamArray=""

for tam in tamArrayNumeric:  
    tamx=[str(x) for x in tam]  
    x_str = ",".join(tamx)
    tamArray=tamArray+x_str+';'

tamArray=tamArray[:-1] 

#############################################
# Parameters
frecuencia=0.8 #0.1--10 Hz
altura_z_pallet = -815
#pallet = [190,-484,aaa]
pallet = [150,-550,altura_z_pallet]# milimetros

girar180 = False
buf = False
bandera=True
T_pallet = []
volPallet = 0
#h_conveyor = -356/1000
h_conveyor = -356/1000
wait_pose=[-0.0372, -1.4364 ,-1.7959,-1.4827, 1.5708, 1.5708]
wait_pose_xyz=[0.597,-0.179,0.4,3.14,0,0] #calcular las coordenadas cartesianas

#
rotx = False
#TC = "Grande"
pay = rtde_r.getPayload()
############################################

printing=False

# Results list
lista_p = [None]*31
Torques= []
posiciones_articulares=[]
#Matriz
Torquess= np.zeros((6, 6))
nombreArchivoLista = "Lista_p.txt"
current_date_time = current_datetime.strftime("%m-%d-%Y, %H-%M-%S")
nombreArchivoLista="Lista_p "+current_date_time+".txt"
ArchivoTorque="Torques"+current_date_time+".txt"

####################################################################################
#Functions

def altura_palet(x,y):
   #z = (0.01*x*x) + (0.02*y*y) + (0.02*x*y) + (-0.01*x) + (0.04*y) + (-0.81)
    z = -(0.0259*x) + (0.0088*y) - 822.0376 #+ 739#para dejar caer
    return z

# Function calculates the Euler angles from the rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convierte ángulos de Euler (RPY) a la matriz de rotación.
    
    Parámetros:
    - roll: Ángulo de roll en radianes.
    - pitch: Ángulo de pitch en radianes.
    - yaw: Ángulo de yaw en radianes.
    
    Devuelve:
    - Matriz de rotación 3x3.
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R_x = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])
    
    R_y = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])
    
    R_z = np.array([[cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R 

#Function transforms Euler angles into rotation vector notation (rot vec)
def deg2rotvec(deg,etapa, normalize=True):
    z_e = 0
    x_e = 0
    y_e = 0
    #print("etapa",etapa)
    if(etapa  == "v"): # cajas grandes y medianas
        #z_e, x_e, y_e=np.radians(deg),np.pi,0
        #z_e, x_e, y_e = (1.580+np.radians(deg)),-1.49, 0.033
        z_e, x_e, y_e = (1.549+np.radians(deg)),-1.503, 0.023
    elif(etapa == "peqv"): # cajas pequenias
        z_e, x_e, y_e = (2.369+np.radians(deg)),-1.497, 0.022
    elif(etapa == "p"): # cjas grandes y medianas
        z_e, x_e, y_e = (1.548+np.radians(deg)),-1.483, 0.025
    elif(etapa == "peqp"): # cajas pequenias
        z_e, x_e, y_e = (-0.709+np.radians(deg)),-1.542, 0.067
    
    else:
        print("no esta entrando")
    
    # Assuming the angles are in radians.
    c1 = np.cos(z_e/2)
    s1 = np.sin(z_e/2)
    c2 = np.cos(x_e/2)
    s2 = np.sin(x_e/2)
    c3 = np.cos(y_e/2)
    s3 = np.sin(y_e/2)
    c1c2 = c1*c2
    s1s2 = s1*s2
    w = c1c2*c3 - s1s2*s3
    x = c1c2*s3 + s1s2*c3
    y = s1*c2*c3 + c1*s2*s3
    z = c1*s2*c3 - s1*c2*s3
    angle = 2 * np.arccos(w)
    if normalize:
        norm = x*x+y*y+z*z
        if norm < 0.001:
            # when all euler angles are zero angle =0 so
            # we can set axis to anything to avoid divide by zero
            x = 1
            y = 0
            z = 0
        else:
            norm = np.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm
    rotvec=angle*np.array([z,x,y])
    return rotvec[0],rotvec[1],rotvec[2]

#Define una función que se ejecutara en un hilo separado para obtener las posiciones y los torques articulares
def obtener_torques():
    global frecuencia, Torques, rtde_r, bandera, posiciones_articulares
    while bandera:
        torque_actual=rtde_c.getJointTorques()
        Torques.append(torque_actual)
        q_actual= rtde_r.getActualQ()
        posiciones_articulares.append(q_actual)
        time.sleep(frecuencia)

# Funtion for the wait pose
def move_to_wait_pose(wait_pose):
    global bandera
    bandera=True
    hilo_torque1= threading.Thread(target=obtener_torques)
    hilo_torque1.start()
    rtde_c.moveJ(q=wait_pose, speed = 3, acceleration = 3, asynchronous = False)
    bandera=False
    hilo_torque1.join()
    cog = rtde_r.getPayloadCog()
    rtde_c.setPayload(1.9,cog)
    

move_to_wait_pose(wait_pose)

# Actualizar coordenadas pallet
def translate_packing(msg,pallet):
    #Definir las variables globales
    global ejeY, posicionEscogida, ReceivedPacking, girar180, rotx,TC
    if(msg == "-1"):
        return -1,0
    elif(msg == "0"):#dejar pasar la caja
        return 0,0

    temp = msg.split(';')
    #print("temp",temp)
    #print(msg)

    #Calcula la altura maxima en el palet actual
    z_max = (int(temp[0])/100)+ 0.02 #metros
    
    SiHayColisiones = True
    inf = [0,0,0]
    sup = [0,0,0]
    #Leer el mensaje de packing
    for i in range(1,len(temp)-1):
        pos = temp[i]
        decode=np.array(pos.split(','), dtype='int32')*10  #milimetros
        inf = decode[0:3]#mm
        sup = decode[3:6] #mm
    

        pos_ff= inf+(sup - inf)/2 #mm
        pos_ff[2]= sup[2]
        pos_ff[0] = pallet[0]- pos_ff[0]#mm
        pos_ff[1] = pallet[1]- pos_ff[1]#mm
        zz = altura_palet(pos_ff[0],pos_ff[1])
        print("zz",zz)
        pos_ff[2] = zz + pos_ff[2]   

        print("pos_ff",pos_ff)
        print("pallet",pallet)

        pos_ff_m = pos_ff/1000 #metros
        rx,ry,rz = 0,0,0
        if( (sup - inf)[0] < (sup - inf)[1]):
            ejeY = False
            rotx = False
            if(TC =="Pequena"):
                rx,ry,rz = deg2rotvec(0,"peqp")
                print("debe rotar5")
            else:
                rx,ry,rz = deg2rotvec(0,"p")
                   
            print("sin rotar")    
            #angulo_place=np.array([4.769,-0.182,0.058])
            
        else:
            ejeY = True
            rotx = False
            if(TC =="Pequena"):
                rx,ry,rz = deg2rotvec(90,"peqp")
                print("debe rotar5")
            else:
                rx,ry,rz = deg2rotvec(90,"p")
                print(" rota 1")
            #angulo_place=np.array([2.495,2.307,-2.531])
        angulo_place=np.array([rx,ry,rz])
        pos_ff_mo = np.concatenate((pos_ff_m,angulo_place))
        print("pose_packing",pos_ff_m)
        q_near4 = [-1.4781,-2.0534,-0.8714, -1.7853, 1.5624,0]
        #my_q=inversa(pos_ff_mo,q_near4 )
        qIK= rtde_c.getInverseKinematics(pos_ff_mo,q_near4)
       # print("my_q mio",np.degrees(my_q))
        #print("my_q robot",np.degrees(np.array(qIK)))
        my_q= qIK
        print("my_q",my_q)
        SiHayColisiones = HayColisiones(my_q,tamCaja, pallet)
        
        if (SiHayColisiones): #Aca se intenta girando el gripper
            my_q[5]+= np.radians(180)

            SiHayColisiones = HayColisiones(my_q,tamCaja, pallet)
            if (SiHayColisiones == False):
                if ejeY:
                    rotx = True
                    if(TC =="Pequena"):
                        rx,ry,rz = deg2rotvec(-90,"peqp")
                        print("debe rotar2")
                    else:
                        rx,ry,rz = deg2rotvec(-90,"p")
                    angulo_place=np.array([rx,ry,rz])  
                    print("debe rotar3")
                    #angulo_place=np.array([2.4, -2.432, 2.523])
                else:
                    rotx = True
                    if(TC =="Pequena"):
                        rx,ry,rz = deg2rotvec(-180,"peqp")
                        print("debe rotar4")
                    else:
                        rx,ry,rz = deg2rotvec(-180,"p")
                        print("debe rotar5")
                    angulo_place=np.array([rx,ry,rz])  
                    #angulo_place=np.array([0.092, 2.161, -2.308])
                
                posicionEscogida = i -1
                AdicionarCajaVirtual(inf / 10, sup / 10)
                #print("Posicion Escogida = ",posicionEscogida)
                pos_ff_m = np.concatenate((pos_ff_m,angulo_place)) 
                break
        else:
            posicionEscogida = i -1
            AdicionarCajaVirtual(inf / 10, sup / 10)
            #print("Posicion Escogida = ",posicionEscogida)
            pos_ff_m = np.concatenate((pos_ff_m,angulo_place))
            break
        
        
    # retornamos en metros.     
    #print("HayColisiones = ", SiHayColisiones)
    #ReceivedPacking = True
    if (SiHayColisiones == False):
        lista_p[24] = inf[0]
        #print(type(inf[0]))
        lista_p[25] = inf[1]   
        lista_p[26] = inf[2]
        lista_p[27] = sup[0]
        lista_p[28] = sup[1]
        lista_p[29] = sup[2]
        lista_p[30] = TC
        return z_max, pos_ff_m
    else:
        print("hubo coliciones en todo")
        return -1, 0
    
# Update the coordinates of the box on the Band
def translate_vision(msg):
    decoded= np.array(msg.split(','), dtype='float16')   

    coor_x = decoded[0]
    coor_y = decoded[1]
    coor_z = decoded[2] 
    length = decoded[3]
    width = decoded[4]
    height= decoded[5]
    angle = decoded[6]
    threshold = decoded[7]
    
    return [coor_x, coor_y, coor_z, length, width, height,angle, threshold]
    

# Function that writes the results to a txt
def AgergarLineaTexto():
    global lista_p, nombreArchivoLista
    texto = str(lista_p[0])
    for i in range(1, len(lista_p)):
        texto += " " + str(lista_p[i])
    with open(nombreArchivoLista, 'a') as archivo:
        archivo.write(texto + '\n')

# Function that calculates inverse kinematics
def inversa(pos_deseada,qnear):
    
    theta1 = qnear[0]
    theta2 = qnear[1]
    theta3 = qnear[2]
    theta4 = qnear[3]
    theta5 = qnear[4]
    theta6 = qnear[5]
    
    x=pos_deseada[0]
    y=pos_deseada[1]
    z=pos_deseada[2]
    rx=pos_deseada[3]
    ry=pos_deseada[4]
    rz=pos_deseada[5]
    
    r = Rotation.from_rotvec(np.array([rx, ry, rz]))
    RR = r.as_matrix()
    
    desired_pos = np.array([[RR[0, 0], RR[0, 1], RR[0, 2], x],
                            [RR[1, 0], RR[1, 1], RR[1, 2], y],
                            [RR[2, 0], RR[2, 1], RR[2, 2], z],
                            [0, 0, 0, 1]])
    
    
    th = np.matrix([[theta1], [theta2], [theta3], [theta4], [theta5], [theta6]])
    q00= invKine(desired_pos,th)
    q0=np.squeeze(q00) # Vector de 6 posiciones articulares
    #q1= np.array([[1,0,0, 0],
    #              [0,1,0, 0],
     #             [0,0,1, -0.2],
    #              [0, 0, 0, 1]])
    #q2=q0*q1

    #print("q2",q2)
    # c = [0]
    #location = HTrans(q00.reshape((6,1)),c )
    
    return q0

# Function that builds the trajectory
def pick_and_place_box(pick,place,z_max,hbox,angulo_pick,TC,detectionElapsed):  
    acx=2.0
    velx= 2.0
    blendus= 0.2
    move_to_wait_pose(wait_pose)
    global lista_p, Listax,girar180, volPallet,rotx, pay,buf, bandera
    

    Vol_cajas= GetVolumenEmpacado()
    Vol_prom = GetVolumenPromedio()
    lista_p[19] = Vol_prom 
    lista_p[20] = Vol_cajas 
    lista_p[21] = Vol_cajas / volPallet # Utilization
    num_cj = get_num_cajas()
    lista_p[22] = num_cj 
    
    ##########################################################################################################
    #Parte de Picking
    if(TC == "Pequena"):
        rx,ry,rz = deg2rotvec(-angulo_pick,"peqv")
        #print("rotacion pequena")
    else:
        rx,ry,rz = deg2rotvec(-angulo_pick,"v")
        #print("rotacion grande")

    #Posiciones
    rot=np.array([rx,ry,rz])            
    pick_point = np.concatenate((pick,rot))
    #print("pick_point",pick_point )
    # Save the picking x,y,z coordinates in the dataframe
    lista_p[0]=pick[0]
    lista_p[1]=pick[1]
    lista_p[2]=pick[2]
    # Save the packing x,y,z coordinates in the dataframe
    lista_p[3]=place[0]
    lista_p[4]=place[1]
    lista_p[5]=place[2]
    #Time to identify the box
    lista_p[6]=detectionElapsed
    
    ################################################################
    # Move to upper picking box place. 
    ctrl_point_1 = pick_point.copy()    
    ctrl_point_1[2] = max(pick_point[2]+hbox,h_conveyor+0.32)
    q_near1 = rtde_r.getActualQ()
    ctrl_joint_1= rtde_c.getInverseKinematics(ctrl_point_1,q_near1)
    ctrl_joint_pick= rtde_c.getInverseKinematics(pick_point,q_near1)
    
    #Distancia
    dist_1 = np.linalg.norm(wait_pose_xyz[0:3]- pick)
   
    # Distance to go down
    dist_bajar = dist_1 +  hbox
    lista_p[14]= dist_bajar
    #Movimiento
    velocity = velx
    acceleration = acx
    blend_1 = 0.08#antes 0.02 original
    blend_2 = 0.0# debe ser cero
    blend_3 = 0.08
    path_pose1 = ctrl_joint_1 +[velocity, acceleration, blend_1]
    path_pose2 = ctrl_joint_pick+[velocity, acceleration, blend_2]
    path_pick = [path_pose1, path_pose2]
    
    bandera=True
    hilo_torque2= threading.Thread(target=obtener_torques)
    hilo_torque2.start()
    temp1 = time.time() 
    rtde_c.moveJ(path_pick)
    
    bandera=False
    hilo_torque2.join()
    #Torque
    torq_0 = rtde_c.getJointTorques()
    Torquess[0]= torq_0 
    #print("Torque de pick", torq_0 )
    temp2 = time.time()
    lista_p[7]= temp2 - temp1
    temp1 = temp2
    #rtde_c.stopScript()
    
    #################################################
     # #  # Turn on gripper 
    if(TC=="Grande"):
         cog = rtde_r.getPayloadCog()
         rtde_i.setToolDigitalOut(0,True)
         time.sleep(0.5)
         rtde_i.setToolDigitalOut(1,True)
         time.sleep(0.5)
         rtde_c.setPayload(1.9+0.7,cog)
         time.sleep(0.5)
         
    else:
         rtde_i.setToolDigitalOut(1,True)
         cog = rtde_r.getPayloadCog()
         rtde_c.setPayload(1.9+ 0.34,cog)
         
         time.sleep(0.5)     
     
    #tiempo
    temp2 = time.time()
    lista_p[8]= temp2 - temp1
    temp1 = temp2
    
    #############################################################################################################
    #Trayectoria libre
    #Posiciones
    # climbing distance
    dist_subir =np.linalg.norm(ctrl_point_1 - pick_point)
    alt=h_conveyor+0.2
    
    #punto near pick
    ctrl_point_2 = np.array([0.250, pick_point[1],0.105,2.391,2.316,-2.601])
    q_near2 = [0,-0.8727,-2.6412, -1.2160, 1.5708, 1.5708]
    ctrl_joint_2= rtde_c.getInverseKinematics(ctrl_point_2,q_near2)
    
    #place
    
    place_point = place
    
    #punto near place
    ctrl_point_3 = np.array([place_point[0],-0.268,0.3,place_point[3],place_point[4],place_point[5]])
    q_near3 = [-1.5708,-0.8355,-2.4831, -1.3963, 1.5708, 0]
    ctrl_joint_3= rtde_c.getInverseKinematics(ctrl_point_3,q_near3)
    
    #Punto intermedio
    alt=h_conveyor+0.2
    hh = max(z_max + hbox + pallet[2]/1000.0,h_conveyor+0.52)
    if(hh < alt ):
        hh = alt+ 0.05
    aaa = np.mean(np.array([pick[0],place[0]]))
    bbb = place[1]
    ctrl_point_0 =[aaa,bbb,hh,2.402,2.326,-2.514] 
    q_near0 = rtde_r.getActualQ()
    ctrl_joint_0 = rtde_c.getInverseKinematics(ctrl_point_0,q_near0)
    if(rotx == True):
        ctrl_joint_0[5]=ctrl_joint_0[5]+0.5*(ctrl_joint_2[5]-ctrl_joint_0[5])    
      
    #Place
    ctrl_point_4 = place_point.copy() 
    ctrl_point_4[2] = max(place_point[2] + 0.02, 0.1525)
    q_near4 = [-1.4781,-2.0534,-0.8714, -1.7853, 1.5624, 0]
    ctrl_joint_4= rtde_c.getInverseKinematics(ctrl_point_4,q_near4)
    ctrl_joint_place= rtde_c.getInverseKinematics(place_point,q_near4)
    
    
    #distance 
    dist_transpor =np.linalg.norm(ctrl_point_1 - ctrl_point_2)
    dist_Bpallet =np.linalg.norm(ctrl_point_2 - place_point)
    distancia_parcial= dist_subir+dist_transpor+dist_Bpallet
    lista_p[15]= distancia_parcial
    #Movimiento
    velocity = velx
    acceleration = acx
    lista_p[12]= velocity
    lista_p[13]= acceleration
    blend_3 = blendus
    blend_4 = blendus
    blend_5 = blendus
    blend_6 = 0.0 #debe ser cero
    
    path_pose3 = ctrl_joint_2 +[ velocity, acceleration, blend_3]
    path_pose4 = ctrl_joint_0 +[ velocity, acceleration, blend_4]
    path_pose5 = ctrl_joint_3   +[ velocity, acceleration, blend_5]
    path_pose6 = ctrl_joint_4   +[ velocity, acceleration, blend_6]
    path_pose7 = ctrl_joint_place+[ velocity, acceleration, blend_6]
    path_tray = [ path_pose4, path_pose6]
    bandera=True
    hilo_torque3= threading.Thread(target=obtener_torques)
    hilo_torque3.start()
    temp1 = time.time() 
    rtde_c.moveJ(path_tray)
    temp2 = time.time()
    bandera = False
    hilo_torque3.join()
    lista_p[9]= temp2 - temp1
    temp1 = temp2
    torq_1 = rtde_c.getJointTorques()
    Torquess[1]= torq_1
    rtde_c.moveL(place_point,1.0,0.8,False)
    #print("Torque de tray", torq_1)
    ##############################################################################################################
    #Parte de packing
    #Posiciones
    lista_p[3]=place[0]
    lista_p[4]=place[1]
    lista_p[5]=place[2]
    #Distancia        
    # # Move to pallet destination.
    ctrl_point_5 = place_point.copy() 
    ctrl_point_5[2] = ctrl_point_4[2]  
    ctrl_joint_5= rtde_c.getInverseKinematics(ctrl_point_5,q_near3)
    
    ##############################################################
    # # Turn off the gripper. 
    if(TC=="Grande"):
        rtde_i.setToolDigitalOut(0,False)
        rtde_i.setToolDigitalOut(1,False)
        cog = rtde_r.getPayloadCog()
        rtde_c.setPayload(1.9,cog)
    else:
        rtde_i.setToolDigitalOut(1,False)
        cog = rtde_r.getPayloadCog()
        rtde_c.setPayload(1.9,cog)
    time.sleep(0.4)
    # Time
    temp2 = time.time()
    lista_p[10]= temp2 - temp1
    temp1 = temp2
    #Distance
    dist_Spallet =np.linalg.norm(ctrl_point_5- place_point)
    dist_regreso = np.linalg.norm(wait_pose_xyz[0:3] - ctrl_point_5[0:3])
    lista_p[16]= dist_Spallet+ dist_regreso
    #########################################################    
    #Movimiento
    velocity = velx
    acceleration = acx
    blend_8 = blendus
    blend_9 = blendus
    blend_10 = 0.0
    path_pose8 = ctrl_joint_5+[ velocity, acceleration, blend_8]
    path_pose10 = wait_pose+[velocity, acceleration, blend_10]
    if(ctrl_point_5[2] > 0.4):
        q_near6 = [-1.4781,-2.0534,-0.8714, -1.7853, 1.5624, 0]
        ctrl_point_6 = [0.6323,-0.290,0.1525,2.402,2.326,-2.514]
        ctrl_joint_6 = rtde_c.getInverseKinematics(ctrl_point_6,q_near6)
        path_pose9 = ctrl_joint_6+[velocity, acceleration, blend_9]
        path_regreso = [path_pose8,path_pose9,path_pose10]
    else:
        path_regreso = [path_pose8,path_pose10]
    
    bandera=True
    hilo_torque4= threading.Thread(target=obtener_torques)
    hilo_torque4.start()
    temp1 = time.time() 
    rtde_c.moveJ(path_regreso)
    temp2 = time.time()
    bandera= False
    hilo_torque4.join()
    lista_p[11]= temp2 - temp1
    temp1 = temp2
    torq_2 = rtde_c.getJointTorques()
    Torquess[2]= torq_2
    #print("Torque de regreso", torq_2)
    ###########################################################################
    # Time Sum
    Suma_t= 0
    for i in range(6,12):
        Suma_t+= lista_p[i]    
    lista_p[17]= Suma_t
    Suma_d = 0
    for i in range(14,17):
        Suma_d += lista_p[i]    
    lista_p[18]= Suma_d
    #lista_red = [round(x, 2) for x in lista_p]
    np.save('TorqueV_medx',Torques)
    #print("Torques",Torques)
    print(np.size(Torques))
    np.save('Posiciones Articulares',posiciones_articulares)
    print(np.size(posiciones_articulares))
    #print("posiciones_articulares",posiciones_articulares)
    AgergarLineaTexto()
    #print("detener")
    #time.sleep(2)  

    return "Done"
# Function to turn on the Band
def conveyor_on():
    rtde_i.setStandardDigitalOut(0,False)
    
    
# Function to turn off the Band
def conveyor_off():
    rtde_i.setStandardDigitalOut(0,True)
    

##############################################################################
# Function that manages messages

def multi_threaded_client(connection):
    global msg_vision
    global last_vision
    global msg_packing
    global visionReady
    global trajectoryReady
    global packingReady
    global T_pallet, volPallet
    global RequestPacking
    global ReceivedPacking
    global CleanUp
    global tamCaja
    global tamArray
    global calibrationArea
    global leeway


    global posicionEscogida
    type= connection.recv(1024).decode('utf-8')
    temp1 = type.split(';')
    print("Type is: ",temp1[0])     
    connection.send(str.encode(tamArray))  
    firstCall = True
    while True:
        if temp1[0] == "Packing":
            if (firstCall):
                firstCall = False
                T_pallet=[float(temp1[1]),float(temp1[2]),float(temp1[3]) ]
                volPallet = T_pallet[0] /100 * T_pallet[1] /100 * T_pallet[2] /100
                #msg_packing = connection.recv(1024).decode('utf-8')
                print('T_pallet = ',T_pallet)
                print('volPallet = ', volPallet)
            if RequestPacking:
                
                connection.sendall(str.encode(tamCaja))
                #print("tamano", str.encode(tamCaja))
                msg_packing = ""
                
                msg_packing = connection.recv(1024).decode('utf-8')
                #print("Msg_packing = ",msg_packing)
                RequestPacking =False
            elif ReceivedPacking:
                ReceivedPacking = False
                connection.send(str.encode(str(posicionEscogida)))
                print("Posicion escogida", posicionEscogida)
                Packing_posicioneEscogida = True

        elif type == "vision":
            if (firstCall):
                firstCall = False
                msg= connection.recv(1024)
                connection.send(str.encode(str(calibrationArea)))
                msg= connection.recv(1024)
                connection.send(str.encode(str(leeway)))
                msg= connection.recv(1024)
            else:
                if trajectoryReady:
                    connection.send(str.encode("Ready"))
                    msg_vision == 'no box'
                    msg_vision= connection.recv(1024).decode('utf-8')
                    if not(msg_vision == 'no box' or  msg_vision == '' or  msg_vision == 'vision'):
                        last_vision=msg_vision
                    #print("Box position: ",msg_vision)
                else:
                    connection.sendall(str.encode("Busy"))
                    msg_vision= connection.recv(1024).decode('utf-8')
                    #print("msg_visi6on: ",msg_vision)
            
    
    connection.close()

def pick_place_traj():
    global msg_vision
    global msg_packing
    global send_message
    global ejeY
    global trajectoryReady
    global visionReady
    global RequestPacking
    global Listax
    global pallet
    global Torques
    global tamCaja
    global ThreadCount
    print('VISION:',msg_vision)
    print('PACKING:',msg_packing)
    global h_conveyor,TC
    global ReceivedPacking



    trajectoryReady=True
    visionReady = True
    detectionStart = time.time()
    detectionElapsed = 0
    state = 0 # 0 means initialization, 1 means awaiting boxes, 2 means refining detection,   3 means calculating end points and executing path
    work_vision ="0.89567,-0.30461,0,21,31,15,5.236394799058843,0.2"
    conveyor_off()
    while True: 
        # print('VISION:',msg_vision)
        # print('PACKING',msg_packing)
        # print("state", state)
        if state == 0:
            
            if ThreadCount >1:
                state = 1
                time.sleep(0.5)
                conveyor_on()
        if state == 1:
            ms_vision=str(msg_vision)
            if not (ms_vision == 'no box' or ms_vision == '' or ms_vision == ' ' or  ms_vision == 'vision'):
                #print("El mensaje fue" , ms_vision,ms_vision == 'vision')
                detectionStart = time.time()
                conveyor_off()
                time.sleep(0.75)
                state = 2
        if state == 2:
            temp_3 = time.time()
            detectionElapsed = temp_3 - detectionStart
            #print("detectionElapsed",detectionElapsed)
            
            #print("threshold1")
            threshold=0.2
            if (msg_vision != 'no box'):
                work_vision=last_vision
                coor_x, coor_y, coor_z, length, width, height, angles, threshold = translate_vision(work_vision)
            if threshold<0.1:
                state=3
                trajectoryReady=False
            elif detectionElapsed>5:
                print("Detection timeout!")
                state = 1
                conveyor_on()
            #print("threshold2")
        if state == 3:
            #print("work_vision", work_vision)
            coor_x, coor_y, coor_z, length, width, height, angles, threshold = translate_vision(work_vision)
            pick = np.array([coor_x, coor_y,coor_z])
            largo = length
            ancho= width
            alto = height
            tamCaja = str(largo)+","+str(ancho)+","+str(alto)

            # Cambio de TC

            if (largo == boxSizes[0][0]  and ancho == boxSizes[0][1]  and alto == boxSizes[0][2]):
                TC ="Grande"

            elif (largo == boxSizes[1][0]  and ancho == boxSizes[1][1]  and alto == boxSizes[1][2]):
                TC ="Mediana"

            else:
                TC= "Pequena"

            #print("tamCaja",tamCaja)
            msg_packing = ""
            RequestPacking =True #Banderita para solicitar packing, el cliente de packing la baja
            while RequestPacking:
                time.sleep(0.001)
            
            z_pick = h_conveyor + (alto/100)
            #print('z_pick:',z_pick, alto)
            pick[2] = z_pick # Conect Robot
            print ("Msg packign a traducir ", msg_packing)
            z_max, place = translate_packing(msg_packing,pallet)
            ReceivedPacking = True
            print ("zmax", z_max, "place", place)
            if z_max == -1 and place == 0:
                print ("Termina de palletizar")
                break # Corregir esto es algo temporal, se deberia mandar mensaje a cada thread para parar todo
            elif(z_max == 0 and place == 0):
                state = 0
                trajectoryReady = True
                print ("Deja pasar")
                conveyor_on()
                time.sleep(1)
                pass
            h_box = alto/100

            #print('place:',place)
            angulo_pick = angles
            Torques=[]
            pick_and_place_box(pick,place,z_max,h_box,angulo_pick,TC,detectionElapsed)
            time.sleep(0.5) 
            trajectoryReady=True #Vision empieza a detectar nuevamente
            state = 1
            conveyor_on() 
        
            np.save('TorqueN_med',Torques)

# Data frame titles
with open(nombreArchivoLista, 'w') as archivo:
    archivo.write('xo yo zo xf yf zf Identificar movelbaja PrenderGripper movelSube movejTrasporta movelBaja ApagarGripper Regreso Suma Velo/s Acel/s2 dist_baja dist_sube dist_Tra dist_Bpallet dist_Spallet dist_Regr Vol_Prom Vol_cjs Utilizacion nCajas infx infy infz supx supy supz Tam' + '\n')

conveyor_off()

start_new_thread(Packing,('192.168.0.108',2004,33,40,100,3))
process = Popen(['python3', 'visionCNN.py'])

while True:
    
    if not printing:
        start_new_thread(pick_place_traj,())
        printing=True
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
      

