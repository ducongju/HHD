_base_ = [
    'mmpose::_base_/default_runtime.py',
    'mmpose::_base_/datasets/crowdpose.py'
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
auto_scale_lr = dict(base_batch_size=32)

# hooks
default_hooks = dict(checkpoint=dict(interval=50,save_best='crowdpose/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(96, 128), heatmap_size=(24, 32), sigma=2)

# base dataset settings
dataset_type = 'CrowdPoseDataset'
data_mode = 'topdown'
data_root = '/data-4T/DATASET/crowdpose/'

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args={{_base_.file_client_args}}),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
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
        ann_file='annotations/mmpose_crowdpose_trainval.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mmpose_crowdpose_test.json',
        bbox_file='/data-4T/DATASET/crowdpose/annotations/det_for_crowd_test_0.1_0.5.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/mmpose_crowdpose_test.json',
    use_area=False,
    iou_type='keypoints_crowd',
    prefix='crowdpose')
test_evaluator = val_evaluator

teacher_ckpt = '/data-8T/lzy/mmpose/work_dirs/td-hm_hrnet_teather-w32_8xb64-210e_crowdpose-256x192/best_crowdpose_AP_epoch_200.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmpose::body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_hrnet_small-w32_8xb64-210e_crowdpose-256x192.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmpose::body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_hrnet_teather-w32_8xb64-210e_crowdpose-256x192.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    teacher_trainable=False,
    teacher_norm_eval=True,
    student_trainable=True,
    calculate_student_loss=True,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='backbone.layer1.3.conv2'),
            neck_s1=dict(type='ModuleOutputs', source='backbone.stage2.1.branches.0.3.conv2'),
            neck_s2=dict(type='ModuleOutputs', source='backbone.stage2.3.branches.0.3.conv2'),
            neck_s3=dict(type='ModuleOutputs', source='backbone.stage3.0.branches.2.3.conv2'),
            neck_s4=dict(type='ModuleOutputs', source='backbone.stage3.2.branches.2.3.conv2')),

        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='backbone.stage2.0.branches.1.3.conv2'),
            neck_s1=dict(type='ModuleOutputs', source='backbone.stage3.1.branches.1.3.conv2'),
            neck_s2=dict(type='ModuleOutputs', source='backbone.stage3.3.branches.1.3.conv2'),
            neck_s3=dict(type='ModuleOutputs', source='backbone.stage4.0.branches.3.3.conv2'),
            neck_s4=dict(type='ModuleOutputs', source='backbone.stage4.2.branches.3.3.conv2')),

        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss'),
            loss_s4=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s0_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=64,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s0_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=64,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=128,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=128,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=128,
                mode='dot_product',
                sub_sample=True),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=128,
                mode='dot_product',
                sub_sample=True),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s4_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s4_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_s0=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s0',
                    connector='loss_s0_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s0',
                    connector='loss_s0_tfeat')),
            loss_s1=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s1',
                    connector='loss_s1_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s1',
                    connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s2',
                    connector='loss_s2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s3',
                    connector='loss_s3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s3',
                    connector='loss_s3_tfeat')),
            loss_s4=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s4',
                    connector='loss_s4_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s4',
                    connector='loss_s4_tfeat')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SelfDistillValLoop')
