import torch
from torchsummary import summary
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.utils import register_all_modules
register_all_modules()
if __name__ == '__main__':
    # model_pth1 = r'/data-8T/lzy/mmpose/work_dirs/td-hm_hrnet_small-w32_8xb64-210e_crowdpose-256x192/best_crowdpose_AP_epoch_270.pth'
    # student = torch.load(model_pth1, map_location=torch.device('cpu'))
    # student=init_pose_estimator(
    #     '/data-8T/lzy/mmpose/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_hrnet_small-w32_8xb64-210e_crowdpose-256x192.py', '/data-8T/lzy/mmpose/work_dirs/td-hm_hrnet_small-w32_8xb64-210e_crowdpose-256x192/best_crowdpose_AP_epoch_270.pth', device='cuda:0')
    # print('student:\n')
    # print(student)
    teather=init_pose_estimator(
         '/data-8T/lzy/mmpose/configs/lzy_configs/dsnt_lite18.py',
        '/data-8T/lzy/mmrazor/work_dirs/15-0/epoch_50.pth', device='cuda:2')
    print('teather:\n')
    # print(teather)
    with open('/data-8T/lzy/DATA/student_lite18.txt','a') as f:
        print(teather, file = f)
        for params in teather.state_dict():   
            f.write("{}\t{}\n".format(params, teather.state_dict()[params]))

    # summary(student,(3,128,96))
    # for key, value in student["state_dict"].items():
    #     # print('student:\n')
    #     print(key,value.size(),sep="  ")
    # with open('/data-8T/lzy/DATA/student.txt', 'w') as f:
    #     for key, value in student["state_dict"].items():
    #         # print(key,value.size(),sep="  ")
    #         f.write('\n'.join((str(key),str(value.size()))))
    # model_pth2 = r'/data-8T/lzy/mmpose/work_dirs/td-hm_hrnet_teather-w32_8xb64-210e_crowdpose-256x192/best_crowdpose_AP_epoch_200.pth'
    # teather = torch.load(model_pth2, map_location=torch.device('cpu'))
    # print('teather:\n')
    # print('teather')
    # with open('/data-8T/lzy/DATA/teather.txt', 'w') as f:
    #     for key, value in teather["state_dict"].items():
    #         # print(key,value.size(),sep="  ")
    #         f.write('\n'.join((str(key),str(value.size()))))

    