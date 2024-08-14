#!/usr/bin/env python3
import socket
from ultralytics import YOLO
import cv2
import depthai as dai
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(img):
    
    h,w = img.shape[:2]
    
    pt=0
    pb=0
    pl=0
    pr=0
    
    if h>w:
        pad_left=int((h-w)/2)
        pl=pad_left
        pad_right=h-w-pad_left
        pr=pad_right
        constant= cv2.copyMakeBorder(img,0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)
    else:
        pad_top=int((w-h)/2)
        pt=pad_top
        pad_bottom=w-h-pad_top
        pb=pad_bottom
        constant= cv2.copyMakeBorder(img,pad_top,pad_bottom,0,0,cv2.BORDER_CONSTANT,value=0)

    letterBox=constant.copy()

    tensor_size=256
    
    image = cv2.resize(constant, (tensor_size, tensor_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    image=image.transpose((2, 0, 1))/255

    return image,pt,pl

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def delete_circle(image,coords,r):
    b, a = coords
    h,w = image.shape[:2]
    y,x = np.ogrid[-a:h-a, -b:w-b]
    mask = x*x + y*y <= r*r

    image[mask] = 0

    return image

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(weights='DEFAULT')
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def get_corners(model, crop,border_x,border_y):
    points=[[0,0],[0,0],[0,0],[0,0]]
    
    
    h,w = crop.shape[:2]
    
    if h>2 and w>2:
    
        image,pt,pl = process_image(crop)    
    
        images=torch.from_numpy(np.array([image])).to(torch_device).float()
    
    
        recon_batch = model(images)
    
        pred = (recon_batch[0].cpu().detach().numpy())
    
        img=pred.transpose(1,2,0)
    
        all_corners=np.sum(img,axis=2)
    
    
    
        p0=list(np.unravel_index(all_corners.argmax(), all_corners.shape))[::-1]
        all_corners=delete_circle(all_corners,p0,25)
    
        p1=list(np.unravel_index(all_corners.argmax(), all_corners.shape))[::-1]
        all_corners=delete_circle(all_corners,p1,25)
    
        p2=list(np.unravel_index(all_corners.argmax(), all_corners.shape))[::-1]
        all_corners=delete_circle(all_corners,p2,25)
    
        p3=list(np.unravel_index(all_corners.argmax(), all_corners.shape))[::-1]
        all_corners=delete_circle(all_corners,p3,25)
        
        points=np.array([p0,p1,p2,p3])
        midpoint=np.mean(points,axis=0)
        relative_points=points-midpoint
        
        angles=np.arctan2(relative_points[:,0],relative_points[:,1])
        indices=np.argsort(angles)
        
        points=points[indices]
        
        if h>w:
            points[:,0]=(points[:,0])*h/w-pl*256/w
        else:
            points[:,1]=(points[:,1])*w/h-pt*256/h
    
    points=(points*np.array([w/256,h/256]))
    points[:,0]+=border_x
    points[:,1]+=border_y
    
    points=points.astype(np.int32).reshape((-1, 1, 2))
    
    return points

def get_pixel_data(im, corners,pixelAreas):
    
    coords=np.reshape(corners,(4,2))
    area = cv2.contourArea(coords)
    areaCheck=np.abs(1-area/pixelAreas)
    size=np.argmin(areaCheck)
    
    cnt=coords
    cathetus=(cnt[1]-cnt[0])
    cathetus2=(cnt[2]-cnt[1])

    if (np.dot(cathetus2,cathetus2)>np.dot(cathetus,cathetus)):
        cathetus=cathetus2
    
    angle=np.degrees(np.arctan2(cathetus[0], cathetus[1]))

    if angle>=0:
        angle=(90-angle)+0
    else:
        angle=-(angle+90)+0
        
    center=np.mean(cnt, axis=0).astype(int)
    #offsets
    mmpp=1.595
    offsetX=362
    offsetY=-693

    pixelX=center[1]
    pixelY=center[0]
  
    coordX=pixelX*mmpp+offsetX
    coordY=pixelY*mmpp+offsetY
    # print('angle','{:02.2f}'.format(angle),'X','{:03.2f}'.format(coordX),'Y','{:03.2f}'.format(coordY),size)
    cv2.circle(im, (int(center[0]),int(center[1])), radius=5, color=(0, 0, 255), thickness=-1)
    
    return im,pixelX,pixelY,coordX,coordY,angle,size
    
    

model = ResNetUNet(4).to(torch_device)
model.load_state_dict(torch.load('UNet3.mod'))
model.eval()

detector = YOLO('boxes.pt',verbose=False) 

print

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
camRgb.setIspScale(1,8)
xoutRgb = pipeline.create(dai.node.XLinkOut)

controlIn = pipeline.create(dai.node.XLinkIn)
configIn = pipeline.create(dai.node.XLinkIn)
ispOut = pipeline.create(dai.node.XLinkOut)

controlIn.setStreamName('control')
configIn.setStreamName('config')
ispOut.setStreamName('isp')

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(526, 390)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)
controlIn.out.link(camRgb.inputControl)
configIn.out.link(camRgb.inputConfig)
camRgb.isp.link(ispOut.input)


# Connect to device and start pipeline
msg="no box"
boxSize=[0.32,0.22,0,15]
det=0.2

ClientMultiSocket = socket.socket()
host = '192.168.0.108'
port = 2004
print('Waiting for connection response')
ClientMultiSocket.connect((host, port))
msg=('vision')
ClientMultiSocket.send(str.encode(msg))

#Calibration
sizes = ClientMultiSocket.recv(1024).decode('utf-8')
print("tamaños",sizes)
ClientMultiSocket.send(str.encode("Received sizes"))
holixxx= ClientMultiSocket.recv(1024).decode('utf-8')
# print("x", holixxx)
calibrationArea=int(holixxx)

ClientMultiSocket.send(str.encode("Received calibration area"))
holixxx= ClientMultiSocket.recv(1024).decode('utf-8')
# print("x", holixxx)
leeway=int(holixxx)
ClientMultiSocket.send(str.encode("Received leeway"))

sizeArray=sizes.split(";")
sizeList=[]
for s in sizeArray:
    sizeN=np.array([int(i) for i in s.split(',')])
    sizeN[0]-=1
    sizeN[1]-=1
    sizeList.append(sizeN)
boxSizes=np.array(sizeList)
areas=boxSizes[:,0]*boxSizes[:,1]*(60.6+boxSizes[:,2]-boxSizes[0,2])/60.6
pixelAreas=calibrationArea*areas/areas[0]
boxSizes[:,:2]=boxSizes[:,:2]+leeway



print(boxSizes)

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    controlQueue = device.getInputQueue('control')
    configQueue = device.getInputQueue('config')
    ispQueue = device.getOutputQueue('isp')
    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    expTime = 14000
    sensIso = 600
    print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTime, sensIso)
    controlQueue.send(ctrl)

    lensPos = 50
    print("Setting manual focus, lens position: ", lensPos)
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(lensPos)
    controlQueue.send(ctrl)

    ctrl = dai.CameraControl()
    ctrl.setManualWhiteBalance(4000)
    controlQueue.send(ctrl)
    na=10

    warmedUp=False
    warmup=0

    lastX=600
    lastY=-400
    lastAngle=0
    
    
    
    while True:
##        ctrl = dai.CameraControl()
##        ctrl.setManualExposure(expTime, sensIso)
##        controlQueue.send(ctrl)
##        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        t_h= 0.2
        t_b= 0.3
        t_d=0.15
        z=0.55

        ispFrames = ispQueue.tryGetAll()
        for ispFrame in ispFrames:
            image0=ispFrame.getCvFrame()
        ready=False

        msg='no box'
        if 'image0' in locals() and not warmedUp:
            warmup=warmup+1
        if warmup>=15 and not warmedUp:
            warmedUp=True

        if warmedUp:
            res = ClientMultiSocket.recv(1024)
    ##        print('Servidor dice: ', res.decode('utf-8'))
            if res.decode('utf-8') == "Ready":
                ready=True
            elif res.decode('utf-8') == "Busy":
                ready=False
            
        if 'image0' in locals() and ready and warmedUp:
            results = detector(image0.copy(), save=False, conf=0.7,verbose=False)
            bboxes=results[0].cpu().numpy().boxes.xyxy
            
            if len(bboxes)>0:
                # print('bboxes',bboxes,len(bboxes))
                bbox=[int(box) for box in bboxes[0]]
                # print('bbox',bbox)
                window_h=30
                window_w=40
                if bbox[0]>window_w and bbox[1]>window_h and bbox[2]<(526-window_w) and bbox[3]<(390-window_h):
                    pad=10
                    crop=image0[bbox[1]-pad:bbox[3]+pad,bbox[0]-pad:bbox[2]+pad]
                    corners=get_corners(model, crop,bbox[0]-pad,bbox[1]-pad)
                    im = cv2.polylines(image0, [corners], True, (255, 0, 0), 2)
                    im,pixelX,pixelY,coordX,coordY,angle,size=get_pixel_data(im,corners,pixelAreas)
                    # print(im.shape)
            
                    lastX=0.35*lastX+0.65*pixelX
                    lastY=0.35*lastY+0.65*pixelY
                    lastAngle=0.5*lastAngle+0.5*angle

                    det=0.001
                    lastX=0.35*lastX+0.65*pixelX
                    lastY=0.35*lastY+0.65*pixelY
                    lastAngle=0.8*lastAngle+0.2*angle

                    det=0.001

                    if (np.abs(pixelX-lastX)+ np.abs(pixelX-lastX))>0.05:
                        det=0.2
                    else:
                        det=0.001

                    boxSize=boxSizes[size].tolist()
                    angle_msg=lastAngle

                    if 120<pixelY<400:
                        msg=str((coordX-24)/1000)+','+str((coordY+11)/1000)+',0,'+str(boxSize[0])+','+str(boxSize[1])+','+str(boxSize[2])+','+str(angle_msg-0.9)+','+str(det)
                    else:
                        msg='no box'
                else:
                    im=image0
                    msg='no box'
            else:
                im=image0
                msg='no box'
               
            cv2.imshow('Detection', cv2.resize(im,(1052, 780)))

        # print(msg)
        if warmedUp:
            ClientMultiSocket.send(str.encode(msg))

        # Retrieve 'bgr' (opencv format) frame
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cv2.destroyAllWindows()
