from enum import Enum


class Augmentation(Enum):
    NONE= "no_augment"
    NORMAL = "normal_augment"
    STRONG = "strong_augment"
