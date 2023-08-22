_base_ = [
    'mmpose::_base_/default_runtime.py',
    # 'mmpose::_base_/datasets/crowdpose.py'
]
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
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(interval=50))

# codec settings
codec = dict(
    type='IntegralRegressionLabel',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2.0,
    normalize=True)

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
   batch_size=64,
    num_workers=8,
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
    batch_size=32,
    num_workers=4,
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

teacher_ckpt = '/data-8T/lzy/mmpose/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmpose::lzy_configs/dsnt_lite18_nopre.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmpose::body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w48_8xb64-210e_mpii-256x256.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    teacher_trainable=False,
    teacher_norm_eval=True,
    student_trainable=True,
    calculate_student_loss=True,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            head=dict(type='ModuleOutputs', source='head.final_layer')),

        teacher_recorders=dict(
            head=dict(type='ModuleOutputs', source='head.final_layer')),

        distill_losses=dict(
            loss_0=dict(type='CropbiasLoss',loss_weight=0.5,mode=2,ifsoftmax=0,normalize=0,norm_mode_s=2,norm_mode_t=2,crop_size=[8,8],t=10)),

        loss_forward_mappings=dict(
            loss_0=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='head',),
                t_feature=dict(
                    from_student=False,
                    recorder='head',)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SelfDistillValLoop')