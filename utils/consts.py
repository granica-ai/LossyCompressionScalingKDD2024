import socket
import os

MAX_GPUS = 2

DEFAULT_FOOD101_PARAMS = {
    "num_epochs": 10,
    "batch_size": 64,
    "lr": 1e-4,
}


COMPRESSION_LEVELS = [1, 2, 4, 7, 10, 15]


location = "enigma0" if socket.gethostname() == "aura7" else "aurora"
FOOD101_PATH = f"/mnt/{location}/km/models/food101/scaling"
FOOD101_SIZES = (
    f"/mnt/{location}/km/results/scaling/food101/train_jxl_sizes_from_file.json"
)
FOOD101_SS = [48480000, 96960000, 290880000, 484800000, 727200000]

if location == "aurora":
    MMDET_PATH = "/home/km/mmdetection"
    data_folder = "datasets"
else:
    MMDET_PATH = "/mnt/enigma0/km/mmdetection"
    data_folder = "data"


CITYSCAPES_PATH = f"/mnt/{location}/km/models/cityscapes/scaling"
CITYSCAPES_SIZES = f"/mnt/{location}/km/results/scaling/cityscapes/train0.2_jxl_sizes_from_file_larger.json"
CITYSCAPES_DATA_ROOT = f"/mnt/{location}/km/{data_folder}/cityscapes/leftImg8bit"
CITYSCAPES_SUBSET_ROOT = os.path.join(CITYSCAPES_PATH, "subsets")
MMSEG_PATH = "/home/km/mmsegmentation"
CITYSCAPES_BASE_CONFIG = "./semantic_segmentation/configs/segformer_cityscapes.py"
CITYSCAPES_SS = [36360000, 57267000, 78174000, 99080999, 119988000]


ISAID_DATA_ROOT = (
    f"/mnt/{location}/km/{data_folder}/iSAID_Devkit/preprocess/dataset/iSAID_patches"
)
ISAID_ANN_PATH = os.path.join(
    ISAID_DATA_ROOT, "train/instancesonly_filtered_train.json"
)
ISAID_PATH = f"/mnt/{location}/km/models/isaid/scaling"
ISAID_SUBSET_ROOT = os.path.join(ISAID_PATH, "subsets")
ISAID_ORIGINAL_CONFIG = os.path.join(
    MMDET_PATH, "projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py"
)
ISAID_TMP_CONFIG_PATH = "./object_detection/configs/isaid-maskrcnn-tmp.py"
ISAID_SIZES = (
    f"/mnt/{location}/km/results/scaling/isaid/train1000_jxl_sizes_from_file.json"
)
ISAID_SS = [84087000, 168174000, 252261000, 336348000, 420435000]


THEORY_PATH = f"/mnt/{location}/km/results/scaling/theory"
