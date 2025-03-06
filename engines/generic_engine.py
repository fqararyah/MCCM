import __init__
from enum import Enum

class ParallelizationStrategies(Enum):
    OFMS_IFM = 0
    OFMS_W = 1
    OFMS_H = 2
    OFMS_H_W = 3
    IN_FILTER_H_W = 4
    CUSTOM = 11