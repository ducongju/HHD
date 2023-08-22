# CUDA_VISIBLE_DEVICES=2 python /data-8T/lzy/mmpose/demo/image_demo.py \
#     /data-8T/lzy/mmpose/tests/data/coco/000000000785.jpg \
#     /data-8T/lzy/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
#     /data-8T/lzy/mmpose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
#     --out-file /data-8T/lzy/mmpose/test_dirs/showHM/hrnet48.jpg \
#     --draw-heatmap \

# CUDA_VISIBLE_DEVICES=2 python /data-8T/lzy/mmpose/demo/image_demo.py \
#     /data-8T/lzy/mmpose/tests/data/coco/000000000785.jpg \
#     /data-8T/lzy/mmpose/configs/lzy_configs/dsnt_lite18.py \
#     /data-8T/lzy/mmpose/work_dirs/dsnt_lite18/epoch_210.pth\
#     --out-file /data-8T/lzy/mmpose/test_dirs/showHM/lite_nodistill_2.jpg \
#     --draw-heatmap \
# CUDA_VISIBLE_DEVICES=2 python /data-8T/lzy/mmpose/demo/image_demo.py \
#     /data-8T/lzy/mmpose/tests/data/coco/000000000785.jpg \
#     /data-8T/lzy/mmpose/configs/lzy_configs/dsnt_lite18_t1.py \
#     /data-8T/lzy/mmrazor/stu_model/17-1-1-150.pth\
#     --out-file /data-8T/lzy/mmpose/test_dirs/showHM/17-1-1-150-t1.jpg \
#     --draw-heatmap \

# CUDA_VISIBLE_DEVICES=2 python /data-8T/lzy/mmpose/demo/image_demo.py \
#     /data-8T/lzy/mmpose/tests/data/coco/000000000785.jpg \
#     /data-8T/lzy/mmpose/configs/body_2d_keypoint/topdown_regression/mpii/td-dsnt_mobilenetv2_8xb64-210e_mpii-256x192.py \
#     /data-8T/lzy/mmrazor/work_dirs/001/epoch_210.pth\
#     --out-file /data-8T/lzy/mmpose/test_dirs/showHM/lite_distill_nocrop_t1.jpg \
#     --draw-heatmap \

# CUDA_VISIBLE_DEVICES=2 python ./tools/test.py \
#   /data-8T/lzy/mmrazor/work_dirs/15-2-3/15-2-3.py \
#   /data-8T/lzy/mmrazor/work_dirs/15-2-3/epoch_210.pth --show

CUDA_VISIBLE_DEVICES=2 python /data-8T/lzy/mmpose/demo/topdown_demo_with_mmdet.py \
    /data-8T/lzy/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    /data-8T/lzy/mmpose/configs/lzy_configs/dsnt_lite18_t1.py \
    /data-8T/lzy/mmrazor/stu_model/15-1-100.pth \
    --input tests/data/coco/000000000785.jpg --show --draw-heatmap \
    --output-root  /data-8T/lzy/mmpose/test_dirs/showHM/