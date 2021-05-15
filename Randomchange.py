from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch


class RandomChanging(object):
    '''
    I apply this Transform by getting inspiration from Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                change_x1 = random.randint(0, img.size()[1] - h)  # 새로운 위치 정의
                change_y1 = random.randint(0, img.size()[2] - w)  # 새로운 위치 정의
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, change_x1:change_x1 + h,
                                                   change_y1:change_y1 + w]  # 새로운 위치의 값으로 변경
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, change_x1:change_x1 + h,
                                                   change_y1:change_y1 + w]  # 새로운 위치의 값으로 변경
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, change_x1:change_x1 + h,
                                                   change_y1:change_y1 + w]  # 새로운 위치의 값으로 변경
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, change_x1:change_x1 + h,
                                                   change_y1:change_y1 + w]  # 새로운 위치의 값으로 변경
                return img

        return img