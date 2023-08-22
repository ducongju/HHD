# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules

from robot_control_code.UBTech import *
from robot_control_code import point2angle

# vid = 0x0525
# pid = 0xA4AC
# robot = UBTech(vid, pid)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmpose into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # inference a single image
    results = inference_topdown(model, args.img)
    results = merge_data_samples(results)
    y=[0,1]
    keypoints=results.pred_instances.keypoints
    print(keypoints)
    angle12y=point2angle.vector2angle(keypoints[0,8]-keypoints[0,6],keypoints[0,12]-keypoints[0,6])
    angle123=point2angle.vector2angle(keypoints[0,8]-keypoints[0,6],keypoints[0,10]-keypoints[0,8])
    angle45y=point2angle.vector2angle(keypoints[0,7]-keypoints[0,5],keypoints[0,11]-keypoints[0,5])
    angle456=point2angle.vector2angle(keypoints[0,7]-keypoints[0,5],keypoints[0,9]-keypoints[0,7])
    print(angle12y)
    print(angle123)
    print(angle45y)
    print(angle456)
    # robot.controlSigleServo(6,angle12y,0.1,0.1)
    # robot.controlSigleServo(8,angle123,0.1,0.1)
    # robot.controlSigleServo(5,angle45y,0.1,0.1)
    # robot.controlSigleServo(7,angle456,0.1,0.1)

    #print('results:{}',results)
    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        draw_heatmap=args.draw_heatmap,
        show=True,
        out_file=args.out_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    