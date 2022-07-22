import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from skimage.measure import label,regionprops

class ResBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, Inplace=True,Dilation=1,NumConv=3):
        super(ResBlock, self).__init__()
        if kernel[0]==1:
            padding=(0,1,1)
            dilation=(1,1,1)
        else:
            #padding=(1,1,1)
            #dilation=(1,1,1)
            padding=(Dilation,Dilation,Dilation)
            dilation=(Dilation,Dilation,Dilation)
        self.NumConv=NumConv
        self.Relu=nn.ReLU(inplace=Inplace)
        self.ConvLayers=nn.ModuleList()
        self.ConvLayers.append(nn.Conv3d(in_ch, out_ch, kernel, padding=padding,dilation=dilation))
        for ConvID in range(1,NumConv):
            self.ConvLayers.append(nn.Conv3d(out_ch, out_ch, kernel, padding=padding,dilation=dilation))
        if NumConv>1:
            self.NormLayers=nn.ModuleList()
            for ConvID in range(NumConv):
                self.NormLayers.append(torch.nn.InstanceNorm3d(out_ch))#, affine=True, track_running_stats=True))
                #self.NormLayers.append(torch.nn.BatchNorm3d(out_ch))
        
    def forward(self, x):
        x1=self.ConvLayers[0](x)
        Next=x1
        for ConvID in range(1,self.NumConv):
            Next = self.ConvLayers[ConvID](Next)
            Next = self.NormLayers[ConvID](Next)
            Next = self.Relu(Next)
        if self.NumConv>1:
            Next = torch.add(Next,x1)
            Next = self.NormLayers[-1](Next)
        Out = self.Relu(Next)
        return Out

class down(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, Inplace,Dilation=1,NumConv=3):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(p_kernel),
            ResBlock(in_ch, out_ch, (3,3,3), Inplace,Dilation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, c_kernel, Inplace=True,Dilation=1,NumConv=3):
        super(up, self).__init__()
        self.p_kernel=p_kernel
        self.fuse = ResBlock(in_ch, out_ch, c_kernel, Inplace,Dilation)
        self.conv = nn.Conv3d(in_ch, out_ch, (1,1,1))
        self.Relu=nn.ReLU(inplace=Inplace)
    def forward(self, x1, x2):
        x1 = F.upsample(x1, size=(x1.size()[2]*self.p_kernel[0],x1.size()[3]*self.p_kernel[1],x1.size()[4]*self.p_kernel[2]),mode='trilinear')
        x1 = self.conv(x1)
        x1 = self.Relu(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x)
        return x

class OutconvG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvG, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
class OutconvR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvR, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class OutconvC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvC, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class GlobalImageEncoder(nn.Module):
    def __init__(self, opt):
        super(GlobalImageEncoder, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=opt.BASE_CHANNELS
        self.inc = ResBlock(1, self.Base, (1,3,3), self.Inplace, Dilation=opt.STAGE_DILATION[0],NumConv=opt.NUM_CONVS[0])
        self.DownModules = nn.ModuleList()
        for i in range(1,len(opt.NUM_CONVS)):
            self.DownModules.append(down(self.Base*(2**(i-1)), self.Base*(2**i),(self.opt.DOWN_SAMPLE_FACTORS[i][0],self.opt.DOWN_SAMPLE_FACTORS[i][1],self.opt.DOWN_SAMPLE_FACTORS[i][2]),self.Inplace,Dilation=opt.STAGE_DILATION[i],NumConv=opt.NUM_CONVS[i]))
        self.DimRestoreModules = nn.ModuleList()
        for i in range(len(opt.NUM_CONVS)):
            self.DimRestoreModules.append(OutconvG(self.Base*(2**i),self.n_classes))
    def forward(self,x):
        x1=self.inc(x)
        Features=[x1]
        LocOut=self.DimRestoreModules[0](x1)
        Next=x1
        for i in range(len(self.DownModules)):
            Next=self.DownModules[i](Next)
            LocOut+=F.upsample(self.DimRestoreModules[1+i](Next), size=(x1.shape[2],x1.shape[3],x1.shape[4]), mode='trilinear')
            Features.append(Next)
        LocOut=F.softmax(LocOut)
        return LocOut,Features   
    def TrainForward(self,x,y,GetGlobalFeat=False):
        #y= F.max_pool3d(y,kernel_size=(2,4,4),stride=(2,4,4))
        LocOut,GlobalFeatPyramid=self.forward(x)
        if GetGlobalFeat:
            return LocOut,y,GlobalFeatPyramid
        else:
            return LocOut,y
        
class GlobalImageEncoder_bak(nn.Module):
    def __init__(self, opt):
        super(GlobalImageEncoder_bak, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=opt.BASE_CHANNELS
        self.inc = ResBlock(1, self.Base, (1,3,3), self.Inplace, Dilation=opt.STAGE_DILATION[0],NumConv=opt.NUM_CONVS[0])
        self.DownModules = nn.ModuleList()
        for i in range(1,len(opt.NUM_CONVS)):
            self.DownModules.append(down(self.Base*(2**(i-1)), self.Base*(2**i),(self.opt.DOWN_SAMPLE_FACTORS[i][0],self.opt.DOWN_SAMPLE_FACTORS[i][1],self.opt.DOWN_SAMPLE_FACTORS[i][2]),self.Inplace,Dilation=opt.STAGE_DILATION[i],NumConv=opt.NUM_CONVS[i]))
        self.DimRestoreModules = nn.ModuleList()
        for i in range(len(opt.NUM_CONVS)):
            self.DimRestoreModules.append(nn.Conv3d(self.Base*(2**i),self.Base, 1))#OutconvG(self.Base*(2**i),self.Base))
        self.OutConv=OutconvG(self.Base,self.n_classes)#*(len(opt.NUM_CONVS)-1),self.n_classes)
    def forward(self,x):
        x1=self.inc(x)
        Features=[x1]
        #LocOut=self.DimRestoreModules[0](x1)
        LocOut=[]                 
        Next=x1
        for i in range(len(self.DownModules)):
            Next=self.DownModules[i](Next)
            LocOut.append(F.upsample(self.DimRestoreModules[1+i](Next), size=(x1.shape[2],x1.shape[3]//2,x1.shape[4]//2), mode='trilinear'))
            Features.append(Next)
        LocOut=F.upsample(F.softmax(self.OutConv(torch.cat(LocOut,dim=1))), size=(x1.shape[2],x1.shape[3],x1.shape[4]), mode='trilinear')
        return LocOut,Features   
    def TrainForward(self,x,y,GetGlobalFeat=False):
        #y= F.max_pool3d(y,kernel_size=(2,4,4),stride=(2,4,4))
        LocOut,GlobalFeatPyramid=self.forward(x)
        if GetGlobalFeat:
            return LocOut,y,GlobalFeatPyramid
        else:
            return LocOut,y
class LocalRegionDecoder(nn.Module):
    def __init__(self, opt):
        super(LocalRegionDecoder, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=opt.BASE_CHANNELS
        self.UpModules=nn.ModuleList()
        for i in range(len(opt.NUM_CONVS)-1):
            self.UpModules.append(up(self.Base*(2**(i+1)), self.Base*(2**i),(self.opt.DOWN_SAMPLE_FACTORS[i+1][0],self.opt.DOWN_SAMPLE_FACTORS[i+1][1],self.opt.DOWN_SAMPLE_FACTORS[i+1][2]),self.opt.CONV_KERNELS[i],self.Inplace,Dilation=self.opt.STAGE_DILATION[i],NumConv=self.opt.NUM_CONVS[i]))
        self.UpModules=self.UpModules[::-1]
        self.SegTop1 = OutconvR(self.Base, 1)
        self.SegTop2 = OutconvC(self.Base, 1)
    def forward(self,GlobalFeatPyramid,RoIs):
        P_Region=[]
        #P_Contour=[]
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            
            #RoI TensorPyramid
            RoiTensorPyramid=[]
            CurrScale=self.opt.DOWN_SAMPLE_FACTORS[0].copy()
            
            for Level in range(len(GlobalFeatPyramid)):
                RoiTensorPyramid.append(GlobalFeatPyramid[Level][:,:,Zstart//CurrScale[0]:Zend//CurrScale[0],Ystart//CurrScale[1]:Yend//CurrScale[1],Xstart//CurrScale[2]:Xend//CurrScale[2]].to(self.opt.GPU[1]))
                
                if Level!=len(GlobalFeatPyramid)-1:
                    CurrScale*=self.opt.DOWN_SAMPLE_FACTORS[Level+1]
                 
            RoiTensorPyramid = RoiTensorPyramid[::-1]
            
            Next=RoiTensorPyramid[0]
            for i in range(4):
                Next=self.UpModules[i](Next,RoiTensorPyramid[i+1])
            p_r = self.SegTop1(Next)
            p_r = F.sigmoid(p_r)
            p_c = self.SegTop2(Next) 
            p_c = F.sigmoid(p_c)
                                 
            P_Region.append(p_r)
            #P_Contour.append(p_c) 
        return P_Region
    def TrainForward(self,GlobalFeatPyramid,RoIs,y_region):
        Y_Region=[]
        #Y_Contour=[]
        #Extract in-region labels
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            y_region_RoI=y_region[:,:,Zstart:Zend,\
                                  Ystart:Yend,\
                                  Xstart:Xend]
#            y_contour_RoI=y_contour[:,:,Zstart:Zend,\
#                                  Ystart:Yend,\
#                                  Xstart:Xend]
            Y_Region.append(y_region_RoI)
            #Y_Contour.append(y_contour_RoI)
        P_Region=self.forward(GlobalFeatPyramid,RoIs)
        return P_Region,Y_Region
    
class RU_Net(nn.Module):
    def __init__(self, opt):
        super(RU_Net, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=48
        self.GlobalImageEncoder=GlobalImageEncoder(opt)
        self.DownSampleRate=(self.opt.DOWN_SAMPLE_RATE[0],self.opt.DOWN_SAMPLE_RATE[1],self.opt.DOWN_SAMPLE_RATE[2])
        self.LabelDownSize=nn.MaxPool3d(self.DownSampleRate)
        self.LocalRegionDecoders=nn.ModuleDict()
        for ClassName in opt.CLASS_NAMES:
            if ClassName!='BackGround':
                self.LocalRegionDecoders[ClassName]=LocalRegionDecoder(opt)     
    def forward_RoI_Loc(self, x,y):
        LocOut,Y=self.GlobalImageEncoder.TrainForward(x,y,False)
        return [LocOut,Y]
    def Localization(self,LocOut,Train=True):
        if Train:
            MAX_ROIS=self.opt.MAX_ROIS_TRAIN
        else:
            MAX_ROIS=self.opt.MAX_ROIS_TEST
        HeatMapShape=LocOut[0,0].shape
        LocOut = self.LabelDownSize(LocOut)
        LocOut = LocOut.to(device='cpu').detach().numpy()
        RoIs={}
        for i in range(1,self.n_classes):
            RoIs[self.opt.DICT_CLASS[i]]=[]
        #num=0
        for i in range(1,self.n_classes):
            
            Heatmap = LocOut[0,i]
            
            if np.max(Heatmap)==0:
                continue
            Heatmap = (Heatmap-np.min(Heatmap))/(np.max(Heatmap)-np.min(Heatmap))
            Heatmap[Heatmap<0.5]=0
            Heatmap[Heatmap>=0.5]=1
            Heatmap*=255
            ConnectMap=label(Heatmap, connectivity= 2)
            Props = regionprops(ConnectMap)
            Area=np.zeros([len(Props)])
            Area=[]
            Bbox=[]
            for j in range(len(Props)):
                bbox=list(Props[j]['bbox'])
                bbox[0]*=self.DownSampleRate[0]
                bbox[1]*=self.DownSampleRate[1]
                bbox[2]*=self.DownSampleRate[2]
                bbox[3]*=self.DownSampleRate[0]
                bbox[4]*=self.DownSampleRate[1]
                bbox[5]*=self.DownSampleRate[2]
                bbox[0]=bbox[0]//self.opt.DOWN_SAMPLE_RATE[0]*self.opt.DOWN_SAMPLE_RATE[0]
                bbox[1]=bbox[1]//self.opt.DOWN_SAMPLE_RATE[1]*self.opt.DOWN_SAMPLE_RATE[1]
                bbox[2]=bbox[2]//self.opt.DOWN_SAMPLE_RATE[2]*self.opt.DOWN_SAMPLE_RATE[2]
                bbox[3]=np.ceil(bbox[3]/self.opt.DOWN_SAMPLE_RATE[0]).astype(np.int)*self.opt.DOWN_SAMPLE_RATE[0]
                bbox[4]=np.ceil(bbox[4]/self.opt.DOWN_SAMPLE_RATE[1]).astype(np.int)*self.opt.DOWN_SAMPLE_RATE[1]
                bbox[5]=np.ceil(bbox[5]/self.opt.DOWN_SAMPLE_RATE[2]).astype(np.int)*self.opt.DOWN_SAMPLE_RATE[2]
                
                for k in range(3):
                    if bbox[k]<0:
                        bbox[k]=0
                for k in range(3,6):
                    if bbox[k]>=HeatMapShape[k-3]-1:
                        bbox[k]=HeatMapShape[k-3]-1
                area=(bbox[3]-bbox[0])*(bbox[4]-bbox[1])*(bbox[5]-bbox[2])
                if area>0:
                    Area.append(area)
                    Bbox.append(bbox)
            #print(Bbox)
            Area=np.array(Area)
            Bbox=np.array(Bbox)
            argsort=np.argsort(Area)
            Area=Area[argsort]
            Bbox=Bbox[argsort]

            Area=Area[::-1]
            Bbox=Bbox[::-1,:]
            
            max_boxes=MAX_ROIS[self.opt.DICT_CLASS[i]]
            if Area.shape[0]>=max_boxes:
                OutBbox=Bbox[:max_boxes,:]
            elif Area.shape[0]==0:
                OutBbox=np.zeros([1,6],dtype=np.int)
                OutBbox[0]=[0,0,0,4,16,16]
            else:
                OutBbox=Bbox
            for j in range(OutBbox.shape[0]):
                RoIs[self.opt.DICT_CLASS[i]].append(OutBbox[j,:])
        return RoIs
            
        
    def TrainForward(self, x, y_region):
    #def TrainForward(self, x, y_region, y_contour):
        LocOut,y_region,GlobalFeatPyramid=self.GlobalImageEncoder.TrainForward(x,y_region,True)
        #RoIs=self.Localization(LocOut,Train=True)
        RoIs=self.Localization(y_region,Train=True)
        P_Regions={}
        #P_Contours={}
        Y_Regions={}
        #Y_Contours={}
        for ClassID in range(1,self.n_classes):
            Class=self.opt.DICT_CLASS[ClassID]
            P_Regions[Class],Y_Regions[Class]=\
            self.LocalRegionDecoders[Class].TrainForward(GlobalFeatPyramid,RoIs[Class],y_region[:,ClassID:ClassID+1])

        
        #return P_Regions,P_Contours,Y_Regions,Y_Contours,RoIs,[LocOut,y_region]
        return P_Regions,Y_Regions,RoIs,[LocOut,y_region]
    def forward(self, x):
        LocOut,GlobalFeatPyramid=self.GlobalImageEncoder.forward(x)
        LocOut_cpu=LocOut.to('cpu').detach().numpy()
        RoIs=self.Localization(LocOut,Train=False) 
        P_Regions={}
        #P_Contours={}
        
        for ClassID in range(self.n_classes):
            Class=self.opt.DICT_CLASS[ClassID]
            if Class=='Background':
                continue
            #print(Class,RoIs[Class])
            P_Regions[Class]=self.LocalRegionDecoders[Class](GlobalFeatPyramid,RoIs[Class])
        #return P_Regions,P_Contours,RoIs,LocOut_cpu
        return P_Regions,RoIs,LocOut_cpu