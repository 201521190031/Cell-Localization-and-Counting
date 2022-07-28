import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch
import torchvision
import torch.nn.functional as F
def judge(img,target,density_value):
    # print(img.size())
    # print(img.size(),target.size())
    blob_img_ok=[]
    blob_den_ok=[]
    blob_img_nook=[]
    blob_den_nook=[]
    if torch.sum(target)/(300*400)<=density_value:
        blob_img_ok.append(img)
        blob_den_ok.append(target)
        return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
    else:
        if target.size()[1]>=target.size()[2]:
            A_img = img[:,:,0:int(0.5*target.size()[1]),:]
            A_den = target[:,0:int(0.5*target.size()[1]),:]
            B_img = img[:,:,int(0.5*target.size()[1]):,:]
            B_den = target[:,int(0.5*target.size()[1]):,:]
            if (torch.sum(A_den)/(300*400)<=density_value) & ((torch.sum(B_den)/(300*400)>density_value)):
                blob_img_ok.append(A_img)
                blob_den_ok.append(A_den)
                blob_img_nook.append(B_img)
                blob_den_nook.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)>density_value) & ((torch.sum(B_den)/(300*400)<=density_value)):
                blob_img_ok.append(B_img)
                blob_den_ok.append(B_den)
                blob_img_nook.append(A_img)
                blob_den_nook.append(A_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)<=density_value) & ((torch.sum(B_den)/(300*400)<=density_value)):
                blob_img_ok.append(A_img)
                blob_den_ok.append(A_den)
                blob_img_ok.append(B_img)
                blob_den_ok.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)>density_value) & ((torch.sum(B_den)/(300*400)>density_value)):
                blob_img_nook.append(A_img)
                blob_den_nook.append(A_den)
                blob_img_nook.append(B_img)
                blob_den_nook.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
        else:
            A_img = img[:,:,:,0:int(0.5*target.size()[2])]
            A_den = target[:,:,0:int(0.5*target.size()[2])]
            B_img = img[:,:,:,int(0.5*target.size()[2]):]
            B_den = target[:,:,int(0.5*target.size()[2]):]
            if (torch.sum(A_den)/(300*400)<=density_value) & ((torch.sum(B_den)/(300*400)>density_value)):
                blob_img_ok.append(A_img)
                blob_den_ok.append(A_den)
                blob_img_nook.append(B_img)
                blob_den_nook.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)>density_value) & ((torch.sum(B_den)/(300*400)<=density_value)):
                blob_img_ok.append(B_img)
                blob_den_ok.append(B_den)
                blob_img_nook.append(A_img)
                blob_den_nook.append(A_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)<=density_value) & ((torch.sum(B_den)/(300*400)<=density_value)):
                blob_img_ok.append(A_img)
                blob_den_ok.append(A_den)
                blob_img_ok.append(B_img)
                blob_den_ok.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook
            if (torch.sum(A_den)/(300*400)>density_value) & ((torch.sum(B_den)/(300*400)>density_value)):
                blob_img_nook.append(A_img)
                blob_den_nook.append(A_den)
                blob_img_nook.append(B_img)
                blob_den_nook.append(B_den)
                return blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook


def node_save(img,target,density_value,train=True):
    node_img=[]
    node_den=[]
    # print(len(node_den))
    node_img.append(img)
    node_den.append(target)
    solution_img=[]
    solution_den=[]
    while len(node_den)!=0:
        image=node_img[0]
        node_img.pop(0)
        density = node_den[0]
        node_den.pop(0)
        blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook = judge(image,density,density_value)
        # print(blob_img_ok,blob_den_ok,blob_img_nook,blob_den_nook)
        # print(len(blob_den_ok))
        if len(blob_den_ok)!=0:
            solution_den+=blob_den_ok
            solution_img+=blob_img_ok
            node_den+=blob_den_nook
            # print(node_den)
            node_img+=blob_img_nook
        else:
            node_den+=blob_den_nook
            node_img+=blob_img_nook
        
        if train==True & len(solution_den)==1:
            break
        # print(solution_den)
    return solution_img,solution_den
        

def crop_half(img,target,density_value,train=True,batch_size=8):
    image_blob=[]
    density_blob=[]
    if train==True:
        for i in range(8):
            crop_size = ((0.4+0.6*random.random())*img.size()[2]/2,(0.4+0.6*random.random())*img.size()[3]/2)

            if random.randint(0,9)<= -1:
                
                
                dx = int(random.randint(0,1)*img.size()[2]*1./2)
                dy = int(random.randint(0,1)*img.size()[3]*1./2)
            else:
                dx = int(random.random()*img.size()[2]*1./2)
                dy = int(random.random()*img.size()[3]*1./2)
            
            
            # print(img,target)

            tem_img = img[:,:,dx:dx+int(crop_size[0]),dy:dy+int(crop_size[1])]
            tem_den = target[:,dx:dx+int(crop_size[0]),dy:dy+int(crop_size[1])]
            
            
            
            
            # if random.random()>0.8:
            #     target = target[:,:,:-1:1]
            #     img = img[:,:,:,:-1:1]
            # img=img.copy()
            # target=target.copy()
            solution_img,solution_den=node_save(tem_img,tem_den,density_value)
            solution_den = solution_den[0]
            solution_img = solution_img[0]
            solution_den=torch.unsqueeze(solution_den,0)
            solution_den=F.upsample_bilinear(solution_den,(300,400))*(solution_den.size()[2]*solution_den.size()[3])/(300*400)
            solution_den=torch.squeeze(solution_den,0)
            solution_img=F.upsample_bilinear(solution_img,(300,400))
            # print(solution_img,solution_den)
            image_blob.append(solution_img)
            density_blob.append(solution_den)
        image = torch.squeeze(torch.stack(image_blob,1),0)
        density = torch.squeeze(torch.stack(density_blob,1),0)
        # for j in range(len(image_blob)):
        #     # h1=F.upsample_bilinear(pool1,(conv5.size()[2],conv5.size()[3]))
        #     image=F.upsample_bilinear(image_blob[j],(300,400))
        #     image_blob[j]=image
        #     density_gt = torch.unsqueeze(density_blob[j],1)
        #     density =F.upsample_bilinear(density_gt,(300,400))*density_blob[j].size()[1]*density_blob[j].size()[2]/(300*400)
        #     density_blob[j]=torch.squeeze(density,1)
        # print(image_blob,density_blob)
        
        return image,density
    else:
        solution_img,solution_den=node_save(img,target,density_value,train=False)
        image_blob+=solution_img
        density_blob+=solution_den
        for j in range(len(image_blob)):
            # h1=F.upsample_bilinear(pool1,(conv5.size()[2],conv5.size()[3]))
            image=F.upsample_bilinear(image_blob[j],(300,400))
            image_blob[j]=image
            density_gt = torch.unsqueeze(density_blob[j],0)
            density =F.upsample_bilinear(density_gt,(300,400))*(density_blob[j].size()[1]*density_blob[j].size()[2])/(300*400)
            density = torch.squeeze(density,0)
            density_blob[j]=density
        image = torch.squeeze(torch.stack(image_blob,1),0)
        density = torch.squeeze(torch.stack(density_blob,1),0)
        print(torch.sum(density),density.size())
        return image,density

    


