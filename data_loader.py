#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:27:54 2019
@author: customer
"""
import numpy as np
import random
import SimpleITK as sitk
from skimage.measure import label,regionprops
from skimage import filters
import torch
#Maximum Bbox Cropping to Reduce Image Dimension
def MaxBodyBox(input):
    Otsu=filters.threshold_otsu(input[input.shape[0]//2])
    Seg=np.zeros(input.shape)
    Seg[input>=Otsu]=255
    Seg=Seg.astype(np.int)
    ConnectMap=label(Seg, connectivity= 2)
    Props = regionprops(ConnectMap)
    Area=np.zeros([len(Props)])
    Area=[]
    Bbox=[]
    for j in range(len(Props)):
        Area.append(Props[j]['area'])
        Bbox.append(Props[j]['bbox'])
    Area=np.array(Area)
    Bbox=np.array(Bbox)
    argsort=np.argsort(Area)
    Area=Area[argsort]
    Bbox=Bbox[argsort]
    Area=Area[::-1]
    Bbox=Bbox[::-1,:]
    MaximumBbox=Bbox[0]
    return Otsu,MaximumBbox


    
def DataLoader(Patient,opt,Subset='Train'):
    assert Subset in ['Train','Valid','Test'] 
    SPACING_STR=str(opt.TO_SPACING[0])+str(opt.TO_SPACING[1])+str(opt.TO_SPACING[2])
    #Image Loading
    #ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+'/'+Patient+'/ResampledImage'+SPACING_STR+'.mhd')
    
    
    
    #ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'ImgMasWin.mhd')
    ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'hei.mhd')
    #ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'hei.mha')
    #ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+'img.nrrd')
    
    
    ImageInput_pre=sitk.GetArrayFromImage(ImageInput_pre)
    Shape=ImageInput_pre.shape
    #print (ImageInput_pre.shape[0],ImageInput_pre.shape[1],ImageInput_pre.shape[2])
    ImageInput=np.zeros((np.ceil(ImageInput_pre.shape[0]/opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*opt.DOWN_SAMPLE_RATE[0]+1,\
                         np.ceil(ImageInput_pre.shape[1]/opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*opt.DOWN_SAMPLE_RATE[1]+1,\
                         np.ceil(ImageInput_pre.shape[2]/opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*opt.DOWN_SAMPLE_RATE[2]+1))
    #print (opt.DOWN_SAMPLE_RATE[0],opt.DOWN_SAMPLE_RATE[1],opt.DOWN_SAMPLE_RATE[2])
    #print (1)
    ImageInput[:ImageInput_pre.shape[0],:ImageInput_pre.shape[1],:ImageInput_pre.shape[2]]=ImageInput_pre
    
    #RegionLabel_pre=sitk.ReadImage(opt.DATA_ROOT+'/'+Patient+'/ResampledLabel'+SPACING_STR+'.mhd')
    
    
    
    #RegionLabel_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'Mandible.mhd')
    
    #RegionLabel_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'Nerve.mhd')
    RegionLabel_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'label.mhd')
    #RegionLabel_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/structures/'+'Mandible.nrrd')
    
    
    RegionLabel_pre=sitk.GetArrayFromImage(RegionLabel_pre)
    RegionLabel=np.zeros((np.ceil(ImageInput_pre.shape[0]/opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*opt.DOWN_SAMPLE_RATE[0]+1,\
                         np.ceil(ImageInput_pre.shape[1]/opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*opt.DOWN_SAMPLE_RATE[1]+1,\
                         np.ceil(ImageInput_pre.shape[2]/opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*opt.DOWN_SAMPLE_RATE[2]+1))
    #RegionLabel[:ImageInput_pre.shape[0],:ImageInput_pre.shape[1],:ImageInput_pre.shape[2]]=RegionLabel_pre
    
    #print (RegionLabel.shape)
    #print (1)
    print (Patient)
    RegionLabel[:ImageInput_pre.shape[0],:ImageInput_pre.shape[1],:ImageInput_pre.shape[2]]=RegionLabel_pre
    #print (RegionLabel_pre.shape)
    #print (ImageInput_pre.shape[0],ImageInput_pre.shape[1],ImageInput_pre.shape[2])

    #print (1)

#    ContourLabel_pre=sitk.ReadImage(opt.DATA_ROOT+'/'+Patient+'/ResampledContour'+SPACING_STR+'.mhd')
#    ContourLabel_pre=sitk.GetArrayFromImage(ContourLabel_pre)
#    ContourLabel=np.zeros((np.ceil(ImageInput_pre.shape[0]/opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*opt.DOWN_SAMPLE_RATE[0]+1,\
#                         np.ceil(ImageInput_pre.shape[1]/opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*opt.DOWN_SAMPLE_RATE[1]+1,\
#                         np.ceil(ImageInput_pre.shape[2]/opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*opt.DOWN_SAMPLE_RATE[2]+1))
#    ContourLabel[:ImageInput_pre.shape[0],:ImageInput_pre.shape[1],:ImageInput_pre.shape[2]]=ContourLabel_pre    
    #Orig Shape Backup
    #Shape=ImageInput.shape
    #Body Bbox Compute
    Otsu,MaximumBbox=MaxBodyBox(ImageInput)
    MaximumBbox[0]=MaximumBbox[0]//opt.DOWN_SAMPLE_RATE[0]*opt.DOWN_SAMPLE_RATE[0]
    MaximumBbox[3]=MaximumBbox[3]//opt.DOWN_SAMPLE_RATE[0]*opt.DOWN_SAMPLE_RATE[0]+1
    MaximumBbox[1]=MaximumBbox[1]//opt.DOWN_SAMPLE_RATE[1]*opt.DOWN_SAMPLE_RATE[1]
    MaximumBbox[4]=MaximumBbox[4]//opt.DOWN_SAMPLE_RATE[1]*opt.DOWN_SAMPLE_RATE[1]+1
    MaximumBbox[2]=MaximumBbox[2]//opt.DOWN_SAMPLE_RATE[2]*opt.DOWN_SAMPLE_RATE[2]
    MaximumBbox[5]=MaximumBbox[5]//opt.DOWN_SAMPLE_RATE[2]*opt.DOWN_SAMPLE_RATE[2]+1
    Max=385
    if Subset=='Train':
        if MaximumBbox[5]-MaximumBbox[2]>Max:
            StartX=random.randint(0,MaximumBbox[5]-MaximumBbox[2]-Max)
            MaximumBbox[2]+=StartX
            MaximumBbox[5]=MaximumBbox[2]+Max

        if MaximumBbox[4]-MaximumBbox[1]>Max:
            StartX=random.randint(0,MaximumBbox[4]-MaximumBbox[1]-Max)
            MaximumBbox[1]+=StartX
            MaximumBbox[4]=MaximumBbox[1]+Max
    #Apply BodyBbox Cropping
    ImageInput=ImageInput[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
    RegionLabel=RegionLabel[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    #ContourLabel=ContourLabel[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
    
    if Subset=='Train':
        Xinvert=random.randint(0,1)
        IntensityScale=random.uniform(0.9,1.1)
    else:
        Xinvert=False
        IntensityScale=1
    #Apply Intensity Jitterring 
    ImageInput=((ImageInput-128.0)*IntensityScale+128.0)/255
    ImageInput[ImageInput>1]=1
    ImageInput[ImageInput<0]=0
    #Apply Random Flipping
    if Xinvert:
        ImageInput=ImageInput[:,:,::-1].copy()
        RegionLabel=RegionLabel[:,:,::-1].copy()
        #ContourLabel=ContourLabel[:,:,::-1].copy()
        
    #To Tensor
    ImageTensor=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ImageTensor[0,0]=ImageInput
    ImageTensor=ImageTensor.astype(np.float)
    ImageTensor=torch.from_numpy(ImageTensor)
    ImageTensor=ImageTensor.float()
    ImageTensor = ImageTensor.to(device=opt.GPU[0])

    RegionLabelTensor=np.zeros([1,len(opt.DICT_CLASS.keys()),ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    for ClassID in range(len(opt.DICT_CLASS.keys())):
        RegionLabelTensor[0,ClassID][RegionLabel==ClassID]=1
    RegionLabelTensor=torch.from_numpy(RegionLabelTensor)
    RegionLabelTensor=RegionLabelTensor.float()
    RegionLabelTensor=RegionLabelTensor.to(device=opt.GPU[0])
    
#    ContourLabelTensor=np.zeros([1,len(opt.DICT_CLASS.keys()),ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
#    for ClassID in range(len(opt.DICT_CLASS.keys())):
#        ContourLabelTensor[0,ClassID][ContourLabel==ClassID]=1
#    ContourLabelTensor=torch.from_numpy(ContourLabelTensor)
#    ContourLabelTensor=ContourLabelTensor.float()
#    ContourLabelTensor=ContourLabelTensor.to(device=opt.GPU[0])
    
    
    #return ImageTensor,RegionLabelTensor,ContourLabelTensor,Shape,MaximumBbox
    return ImageTensor,RegionLabelTensor,Shape,MaximumBbox

def Resampling(Image,opt):
    Size=Image.GetSize()
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    NewSpacing = (opt.TO_SPACING[0],opt.TO_SPACING[1],opt.TO_SPACING[2])
    NewSize=[int(Size[0]*Spacing[0]/NewSpacing[0]),int(Size[1]*Spacing[1]/NewSpacing[1]),int(Size[2]*Spacing[2]/NewSpacing[2])]       
    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    Resample.SetInterpolator(sitk.sitkLinear)
    Resample.SetOutputSpacing(NewSpacing)
    
    NewImage = Resample.Execute(Image)
    NewImage = sitk.GetArrayFromImage(NewImage)
    NewImage[NewImage>400]=400
    NewImage[NewImage<-400]=-400
    NewImage=((NewImage+400)/800*255).astype(np.uint8)
    #NewImage=NewImage[:,::-1]
    
    NewImageCropped=np.zeros((np.ceil(NewImage.shape[0]/opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*opt.DOWN_SAMPLE_RATE[0]+1,\
                np.ceil(NewImage.shape[1]/opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*opt.DOWN_SAMPLE_RATE[1]+1,\
                np.ceil(NewImage.shape[2]/opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*opt.DOWN_SAMPLE_RATE[2]+1))
    NewImageCropped[:NewImage.shape[0],:NewImage.shape[1],:NewImage.shape[2]]=NewImage
    
    NewImage=sitk.GetImageFromArray(NewImageCropped)
    NewImage.SetOrigin(Origin)
    NewImage.SetDirection(Direction)
    NewImage.SetSpacing(NewSpacing)
    
    return NewImage

def ArbitraryDataLoader(Image,opt): 
    #Image Loading
    Image=Resampling(Image,opt)
    ImageInput=sitk.GetArrayFromImage(Image)
    ImageInput=ImageInput[:,::-1]
    #Orig Shape Backup
    Shape=ImageInput.shape
    #Body Bbox Compute
    Otsu,MaximumBbox=MaxBodyBox(ImageInput)
           

    #Apply BodyBbox Cropping
    ImageInput=ImageInput[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
    ImageInput=ImageInput.astype(np.float16)/255.0
    #To Tensor
    ImageTensor=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ImageTensor[0,0]=ImageInput
    ImageTensor=ImageTensor.astype(np.float)
    ImageTensor=torch.from_numpy(ImageTensor)
    ImageTensor=ImageTensor.float()
    ImageTensor = ImageTensor.to(device=opt.GPU[0])
    
    return ImageTensor,Shape,MaximumBbox,Image
