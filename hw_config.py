import constants
import json
import constants
from enum import Enum
import os
import numpy as np
from sklearn.linear_model import LinearRegression

MEMORY_EFFICIENCY = 0.98
FREQUENCY = 200 * (10 ** 6)
VALIDATION_FREQUENCY = 150 * (10 ** 6)

class Resource(Enum):
    DSP = 0
    BRAM_R = 1
    BRAM_RW = 2

class HWConfig:
    def __init__(self, board_name = '', num_pes = -1, on_chip_memory = -1, bw = -1, frequency = FREQUENCY,
                 hw_config_file = None):
        self.board_name = board_name
        if board_name != '':
            self.hw_config_file = hw_config_file
            board_specs = self.get_board_specs(board_name)
            self.num_pes = board_specs['num_pes']
            self.on_chip_memory = board_specs['on_chip_memory'] * constants.MiB * MEMORY_EFFICIENCY
            self.bw = board_specs['bw'] * constants.GB
            self.frequency = frequency
            if 'off_chip_access_energy' in board_specs:
                self.off_chip_access_energy = board_specs['off_chip_access_energy'] 
        else:
            self.num_pes = num_pes
            self.on_chip_memory = on_chip_memory * constants.MiB * MEMORY_EFFICIENCY
            self.bw = bw * constants.GB
            self.frequency = frequency

    def copy_hw_config(self):
        return HWConfig(self.board_name, self.num_pes, self.on_chip_memory, self.bw, self.frequency)

    def get_board_specs(self, board_name):

        if self.hw_config_file is not None:
            with open(self.hw_config_file, 'r') as f:
                content = json.load(f)
                for entry in content:
                    if entry['board_name'] == board_name:
                        if entry is not None:
                            return entry

        for hw_config_file in constants.HW_CONFIGS_FILES_v2:
            if not os.path.exists(hw_config_file):
                continue
            assert self.hw_config_file is None
            with open(hw_config_file, 'r') as f:
                content = json.load(f)
                for entry in content:
                    if entry['board_name'] == board_name:
                        if entry is not None:
                            return entry