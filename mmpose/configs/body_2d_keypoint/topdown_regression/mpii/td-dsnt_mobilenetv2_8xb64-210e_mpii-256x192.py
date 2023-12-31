_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(interval=50,save_best='pck/PCKh', rule='greater'))

# codec settings
codec = dict(
    type='IntegralRegressionLabel',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2.0,
    normalize=True)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7, ),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='mmcls://mobilenet_v2',
        # )
        ),
    head=dict(
        type='DSNTHead',
        in_channels=1280,
        in_featuremap_size=(8, 8),
        num_joints=16,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='SmoothL1Loss', use_target_weight=True),
                dict(type='KeypointMSELoss', use_target_weight=True)
            ]),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
        shift_heatmap=True,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='/data-8T/lzy/mmpose/litehrnet18_mpii_256x256-cabd7984_20210623.pth')
    )

# base dataset settings
dataset_type = 'MpiiDataset'
data_mode = 'topdown'
data_root = '/data-4T/DATASET/mpii/'

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args={{_base_.file_client_args}}),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform',shift_prob=0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateTarget',
        target_type='heatmap+keypoint_label',
        encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', file_client_args={{_base_.file_client_args}}),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]


# data loaders
train_dataloader = dict(
   batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mpii_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mpii_val.json',
        headbox_file=f'{data_root}/annotations/mpii_gt_val.mat',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='MpiiPCKAccuracy', norm_item='head')
test_evaluator = val_evaluator