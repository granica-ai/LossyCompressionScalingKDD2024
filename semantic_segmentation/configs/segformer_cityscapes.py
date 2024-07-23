_base_ = [
    "/home/km/mmsegmentation/configs/_base_/models/segformer_mit-b0.py",
    "/home/km/mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py",
    "/home/km/mmsegmentation/configs/_base_/default_runtime.py",
    "/home/km/mmsegmentation/configs/_base_/schedules/schedule_80k.py",
]

custom_imports = dict(
    imports=["utils.mmdet_jxl_transform"],
    allow_failed_imports=False,
)

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

data_root = "/mnt/aurora/km/datasets/cityscapes/"
train_dataloader = dict(dataset=dict(data_root=data_root), batch_size=3, num_workers=4)
val_dataloader = dict(dataset=dict(data_root=data_root), batch_size=3, num_workers=4)
test_dataloader = val_dataloader

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth"  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    test_cfg=dict(mode="slide", crop_size=(1024, 1024), stride=(768, 768)),
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

work_dir = "/mnt/aurora/km/models/cityscapes/scaling/"
