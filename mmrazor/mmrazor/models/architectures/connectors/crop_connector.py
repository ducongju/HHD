import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector
import torch
import torch.nn as nn
from typing import Dict, Optional

import torch.nn.functional as F
from torch import Tensor, nn

import cv2
import numpy as np

@MODELS.register_module()
class CropConnector(BaseConnector):
    """Connector with crop.

    Args:
        dim_in (int, optional): input channels. Defaults to 1024.
        dim_out (int, optional): output channels. Defaults to 128.
    """

    def __init__(
        self,
        crop_size: list,
        # teacher_channels: int,
        # lambda_mgd: float = 0.65,
        # mask_on_channel: bool = False,
        init_cfg: Optional[Dict] = None,
        ifzero:bool=True

    ) -> None:
        super().__init__(init_cfg)
        self.crop_size = crop_size
        self.flag=0
        self.ifzero=ifzero
        self.norm_mode=1
        self.mult=1

    def crop_single_map(self,feature_map,crop_size):

        h,w=feature_map.size()
        # print(feature_map)
        # print(torch.max(feature_map))
        max_index = int(torch.argmax(feature_map))
        # print(max_index.type)
        peak_row=max_index // w
        peak_col=max_index % w
        crop_h=crop_size[0]
        crop_w=crop_size[1]
        half_h=crop_h//2
        half_w=crop_w//2

        # pad_top = crop_h // 2 
        # pad_bottom = crop_h // 2
        # pad_left = crop_w // 2 
        # pad_right = crop_w // 2 
        # # padded_feature_maps = torch.nn.functional.pad(feature_maps, (pad_left, pad_right, pad_top, pad_bottom))
        
        # 计算裁剪的起始和结束位置
        # crop_row_start = max(0,peak_row-half_h)
        # crop_row_end = min(peak_row + half_h,h)
        # crop_col_start = max(0,peak_col-half_w)
        # crop_col_end = min(peak_col + half_w,w)
        crop_row_start = peak_row
        crop_row_end = peak_row + crop_h
        crop_col_start = peak_col
        crop_col_end = peak_col + crop_w
        pad_top=max(0,half_h-peak_row)
        pad_bottom=max(0,peak_row+half_h-h)
        pad_left=max(0,half_w-peak_col)
        pad_right=max(0,peak_col+half_w-w)
        # if peak_row-half_h<0:
        #     pad_top=half_h-peak_row
        # if peak_row+half_h>h:
        #     pad_bottom=peak_row+half_h-h
        # if peak_col-half_w<0:
        #     pad_left=half_w-peak_col
        # if peak_col+half_w>w:
        #     pad_right=peak_col+half_w-w

        crop_feature_map = torch.nn.functional.pad(feature_map, (half_w,half_w,half_h,half_h),value=0)
        padded_feature_map = crop_feature_map[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
        # padded_feature_map = torch.nn.functional.pad(feature_map, (pad_left, pad_right, pad_top, pad_bottom),value=0)
        # cropped_feature_map = feature_map[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
        # padded_feature_map = torch.nn.functional.pad(cropped_feature_map, (pad_left, pad_right, pad_top, pad_bottom))

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

    def crop_single_map_0pad(self,feature_map,crop_size):

        h,w=feature_map.size()
        # print(feature_map)
        # print(torch.max(feature_map))
        max_index = torch.argmax(feature_map)
        peak_row=max_index // w
        peak_col=max_index % w
        crop_h=crop_size[0]
        crop_w=crop_size[1]

        pad_top = crop_h // 2 
        pad_bottom = crop_h // 2
        pad_left = crop_w // 2 
        pad_right = crop_w // 2 
        # padded_feature_maps = torch.nn.functional.pad(feature_maps, (pad_left, pad_right, pad_top, pad_bottom))
        
        # 计算裁剪的起始和结束位置
        crop_row_start = peak_row
        crop_row_end = peak_row + crop_h
        crop_col_start = peak_col
        crop_col_end = peak_col + crop_w
        # pad_left=0
        # pad_right=0
        # pad_top=0
        # pad_bottom=0
        # if peak_row-half_h<0:
        #     pad_top=half_h-peak_row
        # if peak_row+half_h>h:
        #     pad_bottom=peak_row+half_h-h
        # if peak_col-half_w<0:
        #     pad_left=half_w-peak_col
        # if peak_col+half_w>w:
        #     pad_right=peak_col+half_w-w


        padded_feature_map = torch.nn.functional.pad(feature_map, (pad_left, pad_right, pad_top, pad_bottom),value=0)
        cropped_feature_map = padded_feature_map[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
        # padded_feature_map = torch.nn.functional.pad(cropped_feature_map, (pad_left, pad_right, pad_top, pad_bottom))

        # if pad_left>0:
        #     padded_feature_map[ :, :pad_left] = torch.flip(padded_feature_map[:, -pad_left:], dims=(1,))
        # if pad_right>0:
        #     padded_feature_map[ :, -pad_right:] = torch.flip(padded_feature_map[:,:pad_right], dims=(1,))
        # if pad_top>0:
        #     padded_feature_map[ :pad_top,:] = torch.flip(padded_feature_map[-pad_top:, :], dims=(0,))
        # if pad_bottom>0:
        #     padded_feature_map[ -pad_bottom:,:] = torch.flip(padded_feature_map[ :pad_bottom, :], dims=(0,))
    
        # 进行裁剪
        return cropped_feature_map

    def crop_and_concat_feature_maps(self,feature_maps, crop_size,flag):
        batch_size, num_channels, height, width = feature_maps.size()
        # num_feature_maps = feature_maps.size(0)
        # print(feature_maps.size())
        # if flag%2==0 and flag<20:
        #     for i in range(feature_maps.shape[1]):
        #         feature = feature_maps[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy

        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/stu_HM_9_2/'+ str(flag)+'-' + str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        # if flag%2==1 and flag<20:
        #     for i in range(feature_maps.shape[1]):
        #         feature = feature_maps[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy

        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/tea_HM_9_2/'+ str(flag)+'-' + str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        cropped_feature_maps=[]
        for i in range(batch_size):
            all_channel=[]
            for j in range(num_channels):

                # 进行裁剪
                # print('crop')
                single_map=feature_maps[i,j]
                if self.ifzero:
                    cropped_feature_map = self.crop_single_map_0pad(single_map,crop_size)
                else:
                    # print('crop')
                    cropped_feature_map = self.crop_single_map(single_map,crop_size)
                all_channel.append(cropped_feature_map)
            # print(all_channel)
            all_channel=torch.stack(all_channel)
            cropped_feature_maps.append(all_channel)
        # 将裁剪后的特征图合并为一个tensor
        cropped_feature_maps_tensor = torch.stack(cropped_feature_maps)
        # if flag%2==0 and flag<20:
        #     for i in range(cropped_feature_maps_tensor.shape[1]):
        #         feature = cropped_feature_maps_tensor[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy

        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/stu_HM_9_2/'+ str(flag)+'-' + str(i) + '_croped.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图
        # if flag%2==1 and flag<20:
        #     for i in range(cropped_feature_maps_tensor.shape[1]):
        #         feature = cropped_feature_maps_tensor[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy

        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/tea_HM_9_2/' + str(flag)+'-'+ str(i) + '_croped.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        
        
        return cropped_feature_maps_tensor


    def flat_softmax(self, feature: Tensor) -> Tensor:
        """Use Softmax to normalize the feature in depthwise."""

        _, N, H, W = feature.shape

        feature = feature.reshape(-1, N, H * W)
        heatmaps = F.softmax(feature, dim=2)

        return heatmaps.reshape(-1, N, H, W)
    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Normalize the input feature.

        Args:
            feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        _, N, H, W = feature.shape

        feature = feature.reshape(-1, N, H * W)

        if self.norm_mode==2:
            # print('t_feature L2')
            # print(feature.norm(2, dim=2, keepdim=True))
            feature=feature / feature.norm(2, dim=2, keepdim=True)
            # print('t_feature L2')
            # print(feature.norm(2, dim=2, keepdim=True))
            feature = feature.view(feature.size(0),-1)
            return feature.reshape(-1, N, H , W)
        else:
            # print('t_feature L1')
            # print(feature.norm(1, dim=2, keepdim=True))
            feature=feature / feature.norm(1, dim=2, keepdim=True)

            feature = feature.view(feature.size(0),-1)
            return feature.reshape(-1, N, H , W)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:

        # print('_size')
        # print(feature.size())
        # feature=self.flat_softmax(feature)
        feature=self.normalize_feature(feature)
        crop_feature=self.crop_and_concat_feature_maps(feature,self.crop_size,self.flag)
        # print('_size')
        # print(feature.size())
        # self.flag =self.flag+1

        return crop_feature
