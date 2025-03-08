
from abc import ABC, abstractmethod
import __init__
import utils
import constants
import math
import mapping_utils
import mapping_utils.mapping_general_utils as mapping_utils

class GenericMapping(ABC):
    '''This class defines the basic interface of a mapping (i.e. mutlipe-CE accelerator)
    A customized multiple-CE accelerator needs mus inherit this class'''

    #main design overheads plus internal temporary buffers
    EXTRA_MEMORY_OVERHEADS_W = 0#0.05
    EXTRA_MEMORY_OVERHEADS_FM = 0#0.05
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB
    def __init__(self, hw_config, model_dag, layers=None, engines=None, first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False) -> None:
        super().__init__()
        self.hw_config = hw_config
        self.model_dag = model_dag
        self.layers = layers
        self.engines = engines
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.first_layer_on_chip_buffer = 0
        self.off_chip_fms_access_of_first_and_last_layers = -1

    @abstractmethod
    def calc_exec_time(self, print_desc = False):
        pass

    @abstractmethod
    def calc_throughput(self):
        pass

    @abstractmethod
    def calc_weights_buffer_sz(self):
        pass

    @abstractmethod
    def calc_off_chip_weights_access(self):
        pass
    
    @abstractmethod
    def calc_fms_buffer_sz(self, print_desc = False):
        pass

    @abstractmethod
    def calc_off_chip_fms_access(self, print_desc = False):
        pass
    
    @abstractmethod
    def get_segment_exec_times(self):
        pass