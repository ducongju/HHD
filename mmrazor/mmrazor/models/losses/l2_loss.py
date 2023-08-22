# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional

import numpy as np
import cv2
from mmrazor.registry import MODELS
import torch.nn.functional as F


@MODELS.register_module()
class L2Loss(nn.Module):
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
        mode:bool = True,
        ifsoftmax:int = 1,
        t:int = 1,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'none',
        norm_mode: int = 2
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
        self.norm_mode=norm_mode

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
        # if self.flag%2==0 and self.flag<100:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
        #     for i in range(s_feature.shape[1]):
        #         feature = s_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature')
        #             np.savetxt(fp, feature, footer='----')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/stu_HM_15_1_inloss/' +str(self.flag)+'-'+ str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        # if self.flag%2==1 and self.flag<100:
        #     # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #     #     fp.write(str(self.flag)+'-feature')
        #     #     np.savetxt(fp, t_feature.cpu().detach().numpy(), footer='----')
            
        #     for i in range(t_feature.shape[1]):
        #         feature = t_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        #         feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
        #         feature = feature.cpu().detach().numpy() # 转为numpy
        #         with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #             fp.write(str(self.flag)+'-'+str(i)+'-feature')
        #             np.savetxt(fp, feature, footer='----')
        #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
        #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

        #         cv2.imwrite('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/' + str(self.flag)+'-'+str(i) + '.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像
        # print('s_feature(before normalize)')
        # print(s_feature)
        # print('t_feature(before normalize)')
        # print(t_feature)
        if self.ifsoftmax==1:
            s_feature=(s_feature*self.t).softmax(dim=-1)
            t_feature=(t_feature*self.t).softmax(dim=-1)
        elif self.ifsoftmax==2:
            s_feature=(s_feature*self.t).softmax(dim=-1)

        # if self.flag%2==0 and self.flag<100:
            # with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
            #     fp.write(str(self.flag)+'-feature-aftersoftmax')
            #     np.savetxt(fp, s_feature.cpu().detach().numpy(), footer='----')
            # for i in range(s_feature.shape[1]):
            #     feature = s_feature[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
            #     feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
            #     feature = feature.cpu().detach().numpy() # 转为numpy
            #     with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
            #         fp.write(str(self.flag)+'-'+str(i)+'-feature-aftersoftmax')
            #         np.savetxt(fp, feature, footer='----')
            #     # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
            #     feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
            #     feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

            #     cv2.imwrite('/data-8T/lzy/mmrazor/data/stu_HM_15_1_inloss/' +str(self.flag)+'-'+ str(i) + '_aftersoftmax.jpg',feature)  # 保存当前层输出的每个channel上的特征图为一张图像

        

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
        # if self.flag%2==0 and self.flag<100:
        #     with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/s_feature.txt', 'a') as fp:
        #         fp.write(str(self.flag)+'-feature-afternorm')
        #         np.savetxt(fp, s_feature.cpu().detach().numpy().reshape(1,-1), footer='----')
        # if self.flag%2==1 and self.flag<100:
        #     with open('/data-8T/lzy/mmrazor/data/tea_HM_15_1_inloss/t_feature.txt', 'a') as fp:
        #         fp.write(str(self.flag)+'-feature_afternorm')
        #         np.savetxt(fp, t_feature.cpu().detach().numpy().reshape(1,-1), footer='----')
            
        if self.mode:
            loss = torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))
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

    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Normalize the input feature.

        Args:
            feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        feature = feature.view(feature.size(0), -1)
        if self.norm_mode==2:
            return feature / feature.norm(2, dim=1, keepdim=True) * self.mult
        else:
            return feature / feature.norm(1, dim=1, keepdim=True) * self.mult

