# dataset settings
dataset_type = 'CIFAR10'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

neg_num = 16384
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        test_mode=False,
        pipeline=train_pipeline,
        neg_num=neg_num),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=test_pipeline,
        neg_num=neg_num),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
