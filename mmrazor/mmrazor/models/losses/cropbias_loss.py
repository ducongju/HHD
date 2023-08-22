# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional

import numpy as np
import cv2
from mmrazor.registry import MODELS
import torch.nn.functional as F
from torch import Tensor, nn

stu_path='/data-8T/lzy/mmrazor/data/lite-txt/'
stu_filename='s_crop32_b10_nomin-3.txt'
tea_path='/data-8T/lzy/mmrazor/data/lite-txt/'
tea_filename='t_crop32_b10_nomin-3.txt'
@MODELS.register_module()
class CropbiasLoss(nn.Module):
    """Calculate the two-norm loss between the two features.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        normalize (bool): Whether to normalize the feature. Defaults to True.
        mult (float): Multiplier for feature normalization. Defaults to 1.0.
        div_element (bool): Whether to divide the loss by element-wise.
            Defaults to False.
        dist (bool): Whether to conduct two-norm dist as torch.dist(p=2).
            Defaults to False.
        mode:  1 is L2loss,0 is L1loss
        ifsoftmax: 0:both no ;1:both yes;2:teacher no,student yes
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        normalize: int = 0,
        mult: float = 1.0,
        div_element: bool = False,
        dist: bool = False,
        mode:int = 1,
        ifsoftmax:int = 1,
        t:int = 1,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'none',
        norm_mode_s: int = 2,
        norm_mode_t:int = 1,
        crop_size: list=[],
        ifskip:bool=False
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.mult = mult
        self.div_element = div_element
        self.dist = dist
        self.flag=0
        self.mode=mode
        self.ifsoftmax=ifsoftmax
        self.t=t
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.norm_mode_s=norm_mode_s
        self.norm_mode_t=norm_mode_t
        self.crop_size=crop_size
        self.ifskip=ifskip
    def flat_softmax(self, feature: Tensor) -> Tensor:
        """Use Softmax to normalize the feature in depthwise."""

        H, W = feature.shape

        feature = feature.reshape(-1, H * W)
        heatmaps = F.softmax(feature, dim=1)

        return heatmaps.reshape( H, W)
    
    def normalize_feature(self, feature: torch.Tensor,mode) -> torch.Tensor:
        """Normalize the input feature.

        Args:
            feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        H, W = feature.shape

        feature = feature.reshape(-1, H * W)

        if mode==2:
            min_,_=feature.min(dim=-1,keepdim=True)
            # feature=feature-min_
            feature=feature/ feature.norm(2, dim=1, keepdim=True)
            # feature = feature.view(feature.size(0),-1)
            return feature.reshape( H , W)
        elif mode==1:
            min_,_=feature.min(dim=-1,keepdim=True)
            # feature=feature-min_
            # if self.flag%100==0:
            # # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
            # #     fp.write(str(self.flag)+'-feature-aftersoftmax')
            # #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
            #     with open(tea_path+tea_filename, 'a') as fp:
            #         fp.write(str(self.flag)+'-T-feature_min\n')
            #         fp.write(str(min_.cpu().detach().numpy()))
            #     for i in range(feature.shape[1]):
            #         feature_ = feature[:,i,:] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
            #         feature_ = feature_.view(H, W) # batch为1，所以可以直接view成二维张量
            #         feature_ = feature_.cpu().detach().numpy() # 转为numpy
            #         with open(tea_path+tea_filename, 'a') as fp:
                        
            #             fp.write(str(self.flag)+'-'+str(i)+'-T-feature_aftermin_sum\n')
            #             fp.write(str(np.sum(feature_))+'\n')
            #             fp.write(str(self.flag)+'-'+str(i)+'-feature_aftermin\n')
            #             np.savetxt(fp, feature_,fmt='%.6f',  footer='----------')
            #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
            feature=feature/ feature.norm(1, dim=1, keepdim=True)
            # feature = feature.view(feature.size(0),-1)

            return feature.reshape( H , W)
        else:
            return feature.reshape( H , W)
        
    def crop_single_map(self,feature_map,crop_size,coord):

        h,w=feature_map.size()
        # print(h,w)
        # max_index = int(torch.argmax(feature_map))
        peak_row=coord[1]
        peak_col=coord[0]
        crop_h=crop_size[0]
        crop_w=crop_size[1]
        half_h=crop_h//2
        half_w=crop_w//2

        crop_row_start = peak_row
        crop_row_end = peak_row + crop_h
        crop_col_start = peak_col
        crop_col_end = peak_col + crop_w
        pad_top=max(0,half_h-peak_row)
        pad_bottom=max(0,peak_row+half_h-h)
        pad_left=max(0,half_w-peak_col)
        pad_right=max(0,peak_col+half_w-w)
        # print(crop_row_start,crop_row_end, crop_col_start,crop_col_end)

        crop_feature_map = torch.nn.functional.pad(feature_map, (half_w,half_w,half_h,half_h),value=0)
        padded_feature_map = crop_feature_map[crop_row_start:crop_row_end, crop_col_start:crop_col_end].clone()
        # crop_feature_map[crop_row_start:crop_row_end, crop_col_start:crop_col_end]=0
        # loss=torch.sum(crop_feature_map)

        # if self.ifskip and pad_bottom+pad_left+pad_right+pad_top != 0:
        #     padded_feature_map.zero_()
        #     return padded_feature_map,loss

        if pad_left>0:
            padded_feature_map[ :, :pad_left] = torch.flip(padded_feature_map[:, -pad_left:], dims=(1,))
        if pad_right>0:
            padded_feature_map[ :, -pad_right:] = torch.flip(padded_feature_map[:,:pad_right], dims=(1,))
        if pad_top>0:
            padded_feature_map[ :pad_top,:] = torch.flip(padded_feature_map[-pad_top:, :], dims=(0,))
        if pad_bottom>0:
            padded_feature_map[ -pad_bottom:,:] = torch.flip(padded_feature_map[ :pad_bottom, :], dims=(0,))
    
        # 进行裁剪
        return padded_feature_map
    def crop_map(self,s_single_map,t_single_map,crop_size):
        H,W=s_single_map.size()
        # linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1,  W) / W
        # linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1,H) / H
        max_index = int(torch.argmax(t_single_map))
        t_y=max_index//W
        t_x=max_index%W
        # print(t_x,t_y)
        # vecx=t_single_map.sum(dim=0)
        # # print(linspace_x,vecx)
        # vecy=t_single_map.sum(dim=1)
        # # print(linspace_y,vecy)
        # s_x=(linspace_x*vecx.cpu().detach()).sum()
        # s_y=(vecy.cpu().detach()*linspace_y).sum()
        # print(s_x,s_y)
        c=float(s_single_map.exp().sum())
        s_x=int(c/(c-1.)*(t_x-1./(2.*c)))
        s_y=int(c/(c-1.)*(t_y-1./(2.*c)))
        # print(s_x,s_y)
        # s_aftersoftmax=self.flat_softmax(s_single_map)
        if self.ifsoftmax==1:
            s_afternorm=self.flat_softmax(s_single_map)
            t_afternorm=self.flat_softmax(t_single_map)
        elif self.ifsoftmax==0:
            s_afternorm=self.normalize_feature(s_single_map,self.norm_mode_s)
            t_afternorm=self.normalize_feature(t_single_map,self.norm_mode_t)
        elif self.ifsoftmax==2:
            s_afternorm=s_single_map
            t_afternorm=t_single_map
        else:
            s_afternorm=self.flat_softmax(s_single_map)
            t_afternorm=t_single_map

        s_crop_map=self.crop_single_map(s_afternorm,crop_size,[s_x,s_y])
        t_crop_map=self.crop_single_map(t_afternorm,crop_size,[t_x,t_y])

        return s_crop_map,t_crop_map

        

    
    def crop_and_concat_feature_maps(self,s_feature:torch.Tensor,t_feature:torch.Tensor, crop_size):

        batch_size, num_channels, height, width = s_feature.size()
        s_cropped_feature_maps=[]
        t_cropped_feature_maps=[]
        # loss=0
        for i in range(batch_size):
            s_all_channel=[]
            t_all_channel=[]
            for j in range(num_channels):
                s_single_map=s_feature[i,j]
                t_single_map=t_feature[i,j]
                s_cropped_feature,t_cropped_feature = self.crop_map(s_single_map,t_single_map,crop_size)
                # print("loss_permap:{}",loss_permap)
                s_all_channel.append(s_cropped_feature)
                t_all_channel.append(t_cropped_feature)
            s_all_channel=torch.stack(s_all_channel)
            t_all_channel=torch.stack(t_all_channel)
            s_cropped_feature_maps.append(s_all_channel)
            t_cropped_feature_maps.append(t_all_channel)
        s_cropped_feature_maps_tensor = torch.stack(s_cropped_feature_maps)
        t_cropped_feature_maps_tensor = torch.stack(t_cropped_feature_maps)  
        # print(cropped_feature_maps_tensor.type())
        return s_cropped_feature_maps_tensor,t_cropped_feature_maps_tensor
    
    def stu_crop(self, feature: torch.Tensor) -> torch.Tensor:

        feature=self.flat_softmax(feature*self.t)
        # print(feature.size())
        # if self.flag%100==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature-aftersoftmax')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(feature.shape[1]):
        #         feature_ = feature[:,i,:, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature_ = feature_.view(feature_.shape[1], feature_.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature_ = feature_.cpu().detach().numpy() # 转为numpy
        #         with open(stu_path+stu_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-S-feature_aftersoftmax_sum\n')
        #             fp.write(str(np.sum(feature_))+'\n')
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature_aftersoftmax\n')
        #             np.savetxt(fp, feature_,fmt='%.6f',  footer='----------')
                # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        crop_feature,loss=self.crop_and_concat_feature_maps(feature,self.crop_size,self.flag)
        # print("crop_feature.type():{}",crop_feature.type())
        return crop_feature,loss

    def tea_crop(self, feature: torch.Tensor) -> torch.Tensor:

        feature=torch.clamp(feature,min=0.0)

        feature=self.normalize_feature(feature)
        # print(f"t_feature_afternorm{torch.sum(feature)}")
        # if self.flag%100==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature-aftersoftmax')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(feature.shape[1]):
        #         feature_ = feature[:,i,:, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature_ = feature_.view(feature_.shape[1], feature_.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature_ = feature_.cpu().detach().numpy() # 转为numpy
        #         with open(tea_path+tea_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-T-feature_afternorm_sum\n')
        #             fp.write(str(np.sum(feature_))+'\n')
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature_afternorm\n')
        #             np.savetxt(fp, feature_,fmt='%.6f',  footer='----------')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        crop_feature,loss=self.crop_and_concat_feature_maps(feature,self.crop_size,self.flag)

        return crop_feature,loss

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        # print('s_feature_size')
        # print(s_feature.size())
        # print('t_feature_size')
        # print(t_feature.size())
        # if self.flag%100==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature-aftersoftmax')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #             # print('s_feature_size')
        #     # print('s_feature_size')
        #     # print(s_feature.size())
        #     # print('t_feature_size')
        #     # print(t_feature.size())
        #     for i in range(s_feature.shape[1]):
        #         feature_s = s_feature[:, i, :, :] 
        #         feature_s = feature_s.view(feature_s.shape[1], feature_s.shape[2])
        #         feature_t=t_feature[:, i, :, :]
        #         feature_t = feature_t.view(feature_t.shape[1], feature_t.shape[2])
        #         feature_s = feature_s.cpu().detach().numpy() # 转为numpy
        #         feature_t = feature_t.cpu().detach().numpy()
        #         with open(stu_path+stu_filename, 'a') as sfp:
        #             sfp.write(str(self.flag)+'-'+str(i)+'-S-feature_sum\n')
        #             sfp.write(str(np.sum(feature_s))+'\n')
        #             sfp.write(str(self.flag)+'-'+str(i)+'-S-feature\n')
        #             np.savetxt(sfp, feature_s,fmt='%.6f',  footer='----------')
        #         with open(tea_path+tea_filename, 'a') as tfp:
        #             tfp.write(str(self.flag)+'-'+str(i)+'T-feature_sum\n')
        #             tfp.write(str(np.sum(feature_t))+'\n')
        #             tfp.write(str(self.flag)+'-'+str(i)+'T-feature\n')
        #             np.savetxt(tfp, feature_t, footer='------------',fmt='%.6f')
                # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
            
        s_feature,t_feature=self.crop_and_concat_feature_maps(s_feature=s_feature,t_feature=t_feature,crop_size=self.crop_size)
        # print(loss2)

        # if self.flag%100==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature-aftersoftmax')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #             # print('s_feature_size')
        #     # print('s_feature_size')
        #     # print(s_feature.size())
        #     # print('t_feature_size')
        #     # print(t_feature.size())
        #     for i in range(s_feature.shape[1]):
        #         feature_s = s_feature[:, i, :, :] 
        #         feature_s = feature_s.view(feature_s.shape[1], feature_s.shape[2])
        #         feature_t=t_feature[:, i, :, :]
        #         feature_t = feature_t.view(feature_t.shape[1], feature_t.shape[2])
        #         feature_s = feature_s.cpu().detach().numpy() # 转为numpy
        #         feature_t = feature_t.cpu().detach().numpy()
        #         with open(stu_path+stu_filename, 'a') as sfp:
        #             sfp.write(str(self.flag)+'-'+str(i)+'-S-feature_aftercrop_sum\n')
        #             sfp.write(str(np.sum(feature_s))+'\n')
        #             sfp.write(str(self.flag)+'-'+str(i)+'-S-feature_aftercrop\n')
        #             np.savetxt(sfp, feature_s,fmt='%.6f',  footer='----------')
        #         with open(tea_path+tea_filename, 'a') as tfp:
        #             tfp.write(str(self.flag)+'-'+str(i)+'T-feature_aftercrop_sum\n')
        #             tfp.write(str(np.sum(feature_t))+'\n')
        #             tfp.write(str(self.flag)+'-'+str(i)+'T-feature_aftercrop\n')
        #             np.savetxt(tfp, feature_t, footer='------------',fmt='%.6f')
                # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];

        if self.ifskip:
            batch_size, num_channels, _, _ = s_feature.size()
            for i in range(batch_size):
                for j in range(num_channels):
                    if torch.count_nonzero(s_feature[i,j])==0 or torch.count_nonzero(t_feature[i,j])==0:
                        s_feature[i,j].zero_()
                        t_feature[i,j].zero_()
        # print("s_feature.type():{}",s_feature.type())
        # print('s_feature_size')
        # print(s_feature.size())
        # print('t_feature_size')
        # print(t_feature.size())
        # if self.flag==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(s_feature.shape[1]):
        #         feature = s_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open(stu_path+stu_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature\n')
        #             np.savetxt(fp, feature, footer='------------',fmt='%.6f')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite(stu_path +str(self.flag)+'-'+ str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        
        # if self.flag==1:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, t_feature.cpu().detach().numpy(), footer='----')
            
        #     for i in range(t_feature.shape[1]):
        #         feature = t_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open(tea_path+tea_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature\n')
        #             np.savetxt(fp, feature, footer='------------',fmt='%.6f')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite(tea_path + str(self.flag)+'-'+str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        # print('s_feature(before normalize)')
        # print(s_feature)
        # print('t_feature(before normalize)')
        # print(t_feature)

        # if self.ifsoftmax==1:
        #     s_feature=self.flat_softmax(s_feature*self.t)
        #     t_feature=self.flat_softmax(t_feature*self.t)
        # elif self.ifsoftmax==2:
        #     s_feature=self.flat_softmax(s_feature*self.t)
        # _, N, H, W = s_feature.shape
        # test=s_feature.reshape(-1, N, H * W)
        # print('t_feature L1')
        # print(test.norm(1, dim=2, keepdim=True))
        # if self.flag==0:
            # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
            #     fp.write(str(self.flag)+'-feature')
            #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(s_feature.shape[1]):
        #         feature = s_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open(stu_path+stu_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature-aftersoftmax\n')
        #             np.savetxt(fp, feature, footer='------------',fmt='%.6f')
        # #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        # #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        # #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        # #         cv2.imwrite('/data-8T/lzy/mmrazor/data/stu-lite-16/' +str(self.flag)+'-'+ str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        
        # if self.flag==1:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, t_feature.cpu().detach().numpy(), footer='----')
            
        #     for i in range(t_feature.shape[1]):
        #         feature = t_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open(tea_path+tea_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature-aftersoftmax\n')
        #             np.savetxt(fp, feature, footer='------------',fmt='%.6f')

        # if self.flag==0:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature-aftersoftmax')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(s_feature.shape[1]):
        #         feature = s_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open('/data-8T/lzy/mmrazor/data/test/s_feature.txt', 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature_distill')
        #             np.savetxt(fp, feature,fmt='%.6f',  footer='----')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/test/' +str(self.flag)+'-'+ str(i) + '_distill.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        
        if self.normalize==1:
            # s_feature = self.normalize_feature(s_feature)
            s_feature = s_feature.view(s_feature.size(0), -1)
            t_feature = self.normalize_feature(t_feature)
        elif self.normalize==0:
            s_feature = s_feature.view(s_feature.size(0), -1)
            t_feature = t_feature.view(s_feature.size(0), -1)
        else:
            t_feature = self.normalize_feature(t_feature)
            s_feature = self.normalize_feature(s_feature)
        # if self.flag==1:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, t_feature.cpu().detach().numpy(), footer='----')

            
        #     for i in range(t_feature.shape[0]):
        #         feature = t_feature[ i, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         # feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open(tea_path+tea_filename, 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature\n')
        #             np.savetxt(fp, feature, footer='------------',fmt='%.6f')
        # if self.flag%100==0:
            
            # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
            #     fp.write(str(self.flag)+'-feature-aftersoftmax')
            #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
                    # print('s_feature_size')
            # print('s_feature_size')
            # print(s_feature.size())
            # print('t_feature_size')
            # print(t_feature.size())
            # for i in range(s_feature.shape[0]):
            #     feature_s = s_feature[i, :] 
            #     # print(feature_s)
            #     feature_t=t_feature[i,:]
            #     feature_s = feature_s.cpu().detach().numpy() # 转为numpy
            #     feature_t = feature_t.cpu().detach().numpy()
            #     with open(stu_path+stu_filename, 'a') as sfp:
            #         sfp.write(str(self.flag)+'-'+str(i)+'-S-feature_sum\n')
            #         sfp.write(str(np.sum(feature_s))+'\n')
            #         sfp.write(str(self.flag)+'-'+str(i)+'-S-feature\n')
            #         np.savetxt(sfp, feature_s,fmt='%.6f',  footer='----------')
            #     with open(tea_path+tea_filename, 'a') as tfp:
            #         tfp.write(str(self.flag)+'-'+str(i)+'T-feature_sum\n')
            #         tfp.write(str(np.sum(feature_t))+'\n')
            #         tfp.write(str(self.flag)+'-'+str(i)+'T-feature\n')
            #         np.savetxt(tfp, feature_t, footer='------------',fmt='%.6f')
            #     # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];

        if self.mode==1:
            # loss1=torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))
            # loss2=torch.pow(loss2_t-loss2_s/100,2)
            # print(f'loss_s:{loss2_s}')
            # print(f'loss_t"{loss2_t}')
            # print(f'loss1:{loss1}')
            # print(f'loss2"{loss2}')
            loss = torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))
        elif self.mode==2:
            loss=torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))
        else:
            loss=F.l1_loss(s_feature, t_feature, self.size_average, self.reduce,
                         self.reduction)
        

        

        # Calculate l2_loss as dist.
        if self.dist:
            loss = torch.sqrt(loss)
        else:
            if self.div_element:
                loss = loss / s_feature.numel()
            else:
                loss = loss / s_feature.size(0)
        self.flag +=1

        return self.loss_weight * loss
        

