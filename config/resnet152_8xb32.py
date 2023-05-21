crop_size = 224
max_iters = 20000
batch_size = 32
num_classes = 4
data_root = "/your/path/to/directory"
dataset_type = "YourDataset"

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="ResNet", depth=152, num_stages=4, out_indices=(3,), style="pytorch"
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1),
    ),
)
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", scale=224),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=int(crop_size * 1.143), edge="short"),
    dict(type="CenterCrop", crop_size=crop_size),
    dict(type="PackInputs"),
]

bgr_mean = data_preprocessor["mean"][::-1]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='train.txt',
        data_prefix="train",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="RandomResizedCrop", scale=crop_size),
            dict(
                type="AutoAugment",
                policies="imagenet",
                hparams=dict(
                    pad_val=[round(x) for x in bgr_mean], interpolation="bicubic"
                ),
            ),
            dict(type="RandomFlip", prob=0.5, direction="horizontal"),
            dict(type="PackInputs"),
        ],
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='val.txt',
        data_prefix="val",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="ResizeEdge", scale=int(crop_size * 1.143), edge="short"),
            dict(type="CenterCrop", crop_size=crop_size),
            dict(type="PackInputs"),
        ],
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='test.txt',
        data_prefix="test",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="ResizeEdge", scale=int(crop_size * 1.143), edge="short"),
            dict(type="CenterCrop", crop_size=crop_size),
            dict(type="PackInputs"),
        ],
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
test_evaluator = dict(type="Accuracy", topk=(1))
optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
)
param_scheduler = dict(
    type="MultiStepLR",
    by_epoch=False,
    milestones=[int(max_iters * 0.3), int(max_iters * 0.6), int(max_iters * 0.9)],
    gamma=0.1,
)
train_cfg = dict(
    by_epoch=False, max_iters=max_iters, val_interval=int(max_iters * 0.05)
)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256)
default_scope = "mmpretrain"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(
        type="MlflowLoggerHook",
        interval=int(max_iters * 0.005),
        exp_name="plants-condition",
        tags={"file": "resnet152_8xb32_hexa6_autoaug"},
        params={
            "crop_size": crop_size,
            "optimizer.type": optim_wrapper.get("optimizer").get("type"),
            "optimizer.lr": optim_wrapper.get("optimizer").get("lr"),
            "optimizer.momentum": optim_wrapper.get("optimizer").get("momentum"),
            "optimizer.weight_decay": optim_wrapper.get("optimizer").get(
                "weight_decay"
            ),
            "param_scheduler.type": param_scheduler.get("type"),
            "param_scheduler.gamma": param_scheduler.get("gamma"),
            "param_scheduler.milestones": param_scheduler.get("milestones"),
            "max_iters": max_iters,
            "model.type": model.get("type"),
            "backbone.type": model.get("backbone").get("type"),
            "img_norm_cfg.mean": data_preprocessor.get("mean"),
            "img_norm_cfg.std": data_preprocessor.get("std"),
            "aug_seq_train": [i.get("type") for i in train_pipeline],
            "aug_seq_val": [i.get("type") for i in test_pipeline],
        },
        log_model=True,
        by_epoch=False,
    ),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=int(max_iters * 0.1),
        max_keep_ckpts=2,
        save_best="accuracy/top1",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="VisualizationHook", enable=True),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="UniversalVisualizer", vis_backends=[dict(type="LocalVisBackend")]
)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
launcher = "none"
