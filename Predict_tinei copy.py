import os
import SimpleITK as sitk
from skimage.measure import label,regionprops
import numpy as np
import random
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from data_loader import ArbitraryDataLoader
from model import RU_Net
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import time
from skimage import filters
from matplotlib import pyplot as pl
TAG='ModerateRF'# or 'RF64' or 'RF88'
inplace=True


def MaxBodyBox1(input):
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


class Config():
    def __init__(self,TAG):
        self.DICT_CLASS={0:'Background',1:'root_canal'}
        self.CLASS_NAMES=['Background','root_canal']
        self.CLASS_NUM=len(self.CLASS_NAMES)
        self.MAX_ROIS_TEST={'Background':0,'root_canal':1}
        self.MAX_ROIS_TRAIN={'Background':0,'root_canal':1}
        self.CLASS_WEIGHTS={'Background':0,'root_canal':1}        
        self.ACTIVE_CLASS=['Background','root_canal']
        self.MAX_ROI_SIZE=[-1,-1,-1]
        self.TO_SPACING=[0.1,0.1,0.1]
        self.TAG=TAG+str(self.TO_SPACING[0])+str(self.TO_SPACING[1])+str(self.TO_SPACING[2])
        self.DOWN_SAMPLE_FACTORS=[np.array((1,1,1)).astype(np.int),np.array((1,2,2)).astype(np.int), np.array((1,2,2)).astype(np.int),np.array((2,2,2)).astype(np.int),np.array((2,2,2)).astype(np.int)]
        self.NUM_CONVS=[1,2,3,3,3]
        self.CONV_KERNELS=[(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
        self.DOWN_SAMPLE_RATE=np.array((1,1,1)).astype(np.int)
        for FACTOR in self.DOWN_SAMPLE_FACTORS:
            self.DOWN_SAMPLE_RATE *= FACTOR

        #self.DATA_ROOT='./debug/tineitest/'
        self.DATA_ROOT='/Users/jaymichael/Downloads/py_project/debug/tineitest/'
        self.INPLACE=True
        self.GPU=['cpu:0','cpu:0']
        self.MAX_EPOCHS=100
        self.WEIGHT_PATH='/Users/jaymichael/Downloads/py_project/debug/weighttinei/'+self.TAG+'.pkl'
        self.TEST_ONLY=False
        self.BASE_CHANNELS=16
        self.DATA_FORMAT='fp16'
        
opt=Config(TAG)

def DataLoader1(Patient,opt,Subset='Test'):
    assert Subset in ['Train','Valid','Test'] 
    SPACING_STR=str(opt.TO_SPACING[0])+str(opt.TO_SPACING[1])+str(opt.TO_SPACING[2])
    #Image Loading
    ImageInput_pre=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'hei.mhd')
    
    
    ImageInput_pre=sitk.GetArrayFromImage(ImageInput_pre)
    Shape=ImageInput_pre.shape
    #print (ImageInput_pre.shape[0],ImageInput_pre.shape[1],ImageInput_pre.shape[2])
    ImageInput=np.zeros((np.ceil(ImageInput_pre.shape[0]/opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*opt.DOWN_SAMPLE_RATE[0]+1,\
                         np.ceil(ImageInput_pre.shape[1]/opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*opt.DOWN_SAMPLE_RATE[1]+1,\
                         np.ceil(ImageInput_pre.shape[2]/opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*opt.DOWN_SAMPLE_RATE[2]+1))
    #print (opt.DOWN_SAMPLE_RATE[0],opt.DOWN_SAMPLE_RATE[1],opt.DOWN_SAMPLE_RATE[2])
    #print (1)
    ImageInput[:ImageInput_pre.shape[0],:ImageInput_pre.shape[1],:ImageInput_pre.shape[2]]=ImageInput_pre
    
    #print (RegionLabel.shape)
    #print (1)
    print (Patient)
    #Body Bbox Compute
    Otsu,MaximumBbox=MaxBodyBox1(ImageInput)
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
        #RegionLabel=RegionLabel[:,:,::-1].copy()
        #ContourLabel=ContourLabel[:,:,::-1].copy()
        
    #To Tensor
    ImageTensor=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ImageTensor[0,0]=ImageInput
    ImageTensor=ImageTensor.astype(np.float)
    ImageTensor=torch.from_numpy(ImageTensor)
    ImageTensor=ImageTensor.float()
    ImageTensor = ImageTensor.to(device=opt.GPU[0])

    #RegionLabelTensor=np.zeros([1,len(opt.DICT_CLASS.keys()),ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    for ClassID in range(len(opt.DICT_CLASS.keys())):
  
    #return ImageTensor,RegionLabelTensor,ContourLabelTensor,Shape,MaximumBbox
     return ImageTensor,Shape,MaximumBbox


def mhdPredict(Patient,Subset):
    MoveImage=sitk.ReadImage(opt.DATA_ROOT+Patient+'/'+Patient+'hei.mhd')
    Image,Shape,MaximumBbox=DataLoader1(Patient,opt,Subset)
    #Image,LabelRegion,LabelContour,Shape,MaximumBbox=DataLoader(Patient,opt,Subset)
    if MaximumBbox[3]>Shape[0]-1:
        MaximumBbox[3]=Shape[0]-1
    if MaximumBbox[4]>Shape[1]-1:
        MaximumBbox[4]=Shape[1]-1
    if MaximumBbox[5]>Shape[2]-1:
        MaximumBbox[5]=Shape[2]-1
    #Label=LabelRegion.to('cpu').detach().numpy()
    time1=time.time()
    with torch.no_grad():
        PredSeg=Model_pre.forward(Image)
    time2=time.time()
    print('time used:',time2-time1)
    #Global Localizer Output
    LocOut=PredSeg[2]#.to('cpu').numpy()
    OutputWhole_Loc=np.zeros([1,opt.CLASS_NUM,Shape[0],Shape[1],Shape[2]])
    OutputWhole_Loc[:,0]=1
    OutputWhole_Loc[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=LocOut[:,:,:MaximumBbox[3]-MaximumBbox[0],:MaximumBbox[4]-MaximumBbox[1],:MaximumBbox[5]-MaximumBbox[2]]
    OutputWhole_Loc[0,0]=1-np.max(OutputWhole_Loc[0,:],axis=0)
    LocSort=OutputWhole_Loc.argsort(axis=1)
    OutputWholeLoc=np.zeros(Shape,dtype=np.uint8)
    OutputWholeLoc=LocSort[0,-1]#i
    OutputWholeLoc=sitk.GetImageFromArray(OutputWholeLoc)
    #Full RU-Net Output
    RegionOutput=np.zeros([Image.shape[0],len(opt.CLASS_NAMES),Image.shape[2],Image.shape[3],Image.shape[4]])
    RegionWeight=np.zeros([Image.shape[0],len(opt.CLASS_NAMES),Image.shape[2],Image.shape[3],Image.shape[4]])+0.001
#    ContourOutput=np.zeros(Label.shape)
#    ContourWeight=np.zeros(Label.shape)+0.001
    RoIs=PredSeg[1]
    #Apply RoI region/contour predictions to in-body volume container
    #If overlapped, average
    for ClassID in range(opt.CLASS_NUM):
        Class=opt.CLASS_NAMES[ClassID]
        if Class=='Background':
            continue
        for i in range(len(PredSeg[0][Class])):
            PredSeg[0][Class][i]=PredSeg[0][Class][i].to('cpu').detach().numpy()
            #PredSeg[1][Class][i]=PredSeg[1][Class][i].to('cpu').detach().numpy()
            Coord=RoIs[Class][i]
            Weight=np.ones(np.asarray(PredSeg[0][Class][i].shape))
            RegionOutput[:,ClassID:ClassID+1,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[0][Class][i]
#             pl.imshow(PredSeg[0][Class][i][0,0,PredSeg[1][Class][i].shape[2]//2],cmap='gray')
#             pl.show()
            RegionWeight[:,ClassID:ClassID+1,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
            #ContourOutput[:,ClassID:ClassID+1,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[1][Class][i]
            #ContourWeight[:,ClassID:ClassID+1,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
    RegionOutput/=RegionWeight
    RegionOutput[0,0]=1-np.max(RegionOutput[0,:],axis=0)
    RegionSort=RegionOutput.argsort(axis=1)
    RegionClass=RegionSort[0,-1]

    #Apply in-body volume container to original volume size
    OutputWhole1=np.zeros(Shape,dtype=np.uint8)
    OutputWhole1[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=RegionClass.astype(np.uint8)[:MaximumBbox[3]-MaximumBbox[0],                                                                                                                     :MaximumBbox[4]-MaximumBbox[1],                                                                                                                     :MaximumBbox[5]-MaximumBbox[2]]
    
    OutputWhole=np.zeros(Shape,dtype=np.uint8)
    OutputWhole[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=RegionClass.astype(np.uint8)[:MaximumBbox[3]-MaximumBbox[0],                                                                                                                     :MaximumBbox[4]-MaximumBbox[1],                                                                                                                     :MaximumBbox[5]-MaximumBbox[2]]

    OutputWhole1=sitk.GetImageFromArray(OutputWhole1)
    OutputWhole1.SetSpacing(opt.TO_SPACING)
    
    #Draw bounding-boxes
    for ClassID in range(opt.CLASS_NUM):
        Class=opt.DICT_CLASS[ClassID]
        if Class=='Background':
            continue
        for Rid in range(len(RoIs[Class])):
            color=(ClassID,ClassID,ClassID)
            Coord=RoIs[Class][Rid]+np.array([MaximumBbox[0],MaximumBbox[1],MaximumBbox[2],MaximumBbox[0],MaximumBbox[1],MaximumBbox[2]])
            #Out-of-volume protection
            for protect in range(3):
                if Coord[protect+3]>=OutputWhole.shape[protect+0]:
                    Coord[protect+3]=OutputWhole.shape[protect+0]-1
            #Draw rectangles
            Rgb=np.zeros([OutputWhole.shape[1],OutputWhole.shape[2],3],dtype=np.uint8)
            Rgb[:,:,0]=OutputWhole[Coord[0]]
            OutputWhole[Coord[0]]=cv2.rectangle(Rgb,(Coord[2],Coord[1]),(Coord[5],Coord[4]),color=color,thickness=2)[:,:,0]
            Rgb[:,:,0]=OutputWhole[Coord[3]]
            OutputWhole[Coord[3]]=cv2.rectangle(Rgb,(Coord[2],Coord[1]),(Coord[5],Coord[4]),color=color,thickness=2)[:,:,0]

            Rgb=np.zeros([OutputWhole.shape[0],OutputWhole.shape[1],3],dtype=np.uint8)
            Rgb[:,:,0]=OutputWhole[:,:,Coord[2]]
            OutputWhole[:,:,Coord[2]]=cv2.rectangle(Rgb,(Coord[1],Coord[0]),(Coord[4],Coord[3]),color=color,thickness=2)[:,:,0]
            Rgb[:,:,0]=OutputWhole[:,:,Coord[5]]
            OutputWhole[:,:,Coord[5]]=cv2.rectangle(Rgb,(Coord[1],Coord[0]),(Coord[4],Coord[3]),color=color,thickness=2)[:,:,0]

            Rgb=np.zeros([OutputWhole.shape[0],OutputWhole.shape[2],3],dtype=np.uint8)
            Rgb[:,:,0]=OutputWhole[:,Coord[1],:]
            OutputWhole[:,Coord[1],:]=cv2.rectangle(Rgb,(Coord[2],Coord[0]),(Coord[5],Coord[3]),color=color,thickness=2)[:,:,0]
            Rgb[:,:,0]=OutputWhole[:,Coord[4],:]
            OutputWhole[:,Coord[4],:]=cv2.rectangle(Rgb,(Coord[2],Coord[0]),(Coord[5],Coord[3]),color=color,thickness=2)[:,:,0]
    #Save mhds        
    
    OutputWhole=sitk.GetImageFromArray(OutputWhole)
    OutputWhole.SetSpacing(opt.TO_SPACING)
    if os.path.exists('/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient)==False:
        os.makedirs('/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient)
    #sitk.WriteImage(MoveImage,'./debug/Outputtesttinei/'+Patient+'/'+Patient+'hei.mhd')
    sitk.WriteImage(MoveImage,'/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient+'/'+Patient+'hei.mhd')
    #sitk.WriteImage(OutputWhole1,'./debug/Outputtesttinei/'+Patient+'/'+Patient+'label.mhd')
    sitk.WriteImage(OutputWhole1,'/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient+'/'+Patient+'label.mhd')
    #sitk.WriteImage(OutputWholeLoc,'./debug/Outputtesttinei/'+Patient+'/'+Patient+'glob.mhd')
    sitk.WriteImage(OutputWholeLoc,'/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient+'/'+Patient+'globpre.mhd')
    #sitk.WriteImage(OutputWhole,'./debug/Outputtesttinei/'+Patient+'/'+Patient+'labelbox.mhd')
    sitk.WriteImage(OutputWhole,'/Users/jaymichael/Downloads/py_project/debug/Outputtesttinei/'+Patient+'/'+Patient+'labelbox.mhd')
    return Loss,len(RoIs)
    
    
    


if __name__=='__main__':
    lr=0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Model_pre=RU_Net(opt)
    Model_pre.GlobalImageEncoder=Model_pre.GlobalImageEncoder.to(device)
    Model_pre.LocalRegionDecoders=Model_pre.LocalRegionDecoders.to(device)
    #print(Model_pre)
    optimizer1 = optim.Adam(list(Model_pre.GlobalImageEncoder.parameters()),lr=lr,amsgrad=True)
    #Model, optimizer1 = amp.initialize(Model_pre, optimizer1,opt_level='O2',loss_scale='dynamic')
    #Model_pre.load_state_dict(torch.load(opt.WEIGHT_PATH,map_location='cpu'))
    AllPatient=os.listdir(opt.DATA_ROOT)
    Model_pre.load_state_dict(torch.load(opt.WEIGHT_PATH,map_location='cpu'),False)

    Model_pre.eval()
        #Lowest=1
    Loss=0
    NumRoIs=0
    for iteration in range(len(AllPatient)):
            Patient=AllPatient[iteration]
            print(Patient)
            Loss_temp,NumRoI=mhdPredict(Patient,'Test')
            NumRoIs+=NumRoI
            Loss+=Loss_temp
            print(Patient,' Loss=',Loss_temp)  
            #mhdPredict(Patient,'test')

    print('FINISHED！')


