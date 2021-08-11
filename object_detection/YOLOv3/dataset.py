#import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    iou_width_height as iou,
    non_max_suppression_as_nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset) :
    def __init__(
        self, 
        csv_filem,
        img_dir, label_dir,
        anchors,
        image_size = 416,
        S = [13, 26, 52],
        C = 20, 
        transform = None,
    ):

        self.annotations = pd.read_csv(csv_file) 
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self) :
        print(123)
