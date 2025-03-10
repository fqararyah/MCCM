
from abc import ABC, abstractmethod
import __init__
import utils
import constants
import math
import mapping_utils.mapping_general_utils as mapping_utils


class GenericMapping(ABC):
    '''
    This class defines the basic interface of a mapping (i.e. mutlipe-CE accelerator).
    A customized multiple-CE accelerator needs mus inherit this class
    '''

    # main design overheads plus internal temporary buffers
    EXTRA_MEMORY_OVERHEADS_W = 0  # 0.05
    EXTRA_MEMORY_OVERHEADS_FM = 0  # 0.05
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB

    def __init__(self, hw_config, model_dag, layers=None, engines=None, first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False) -> None:
        '''
        Constructor.

        Parameters:
        hw_config (HWConfig): basic HW configurations, see HWConfig Class.
        model_dag (dict): modeld description.
        layers (list): in case the mapping targets subset of the layers rather than the whole model.
        engines (list): in case the engines configurations are already specified.
        first_layer_ifms_are_on_chip (boolean): in case the mapping targets subet of the model and the input
        of the first targeted layer is already stored on-chip.
        last_layer_ofms_are_on_chip (boolean): in case the mapping targets subet of the model and the output
        of the last targeted layer is to be stored on-chip.
        '''
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
    def calc_exec_time(self, print_desc=False):
        '''
        This function calculates the end-to-end inference time.

        Returns:
        float: execution time in seconds.
        '''
        pass

    @abstractmethod
    def calc_throughput(self):
        '''
        This function calculates the throughput as frames per second (FPS).

        Returns:
        float: FPS.
        '''

        pass

    @abstractmethod
    def calc_weights_buffer_sz(self):
        '''
        This function calculates the size of the on-chip weights buffer.

        Returns:
        int: weights buffer size.
        '''
        pass

    @abstractmethod
    def calc_off_chip_weights_access(self):
        '''
        This function calculates the number of the off-chip weights accesses.

        Returns:
        int: number of weight accesses.
        '''

        pass
    
    @abstractmethod
    def calc_fms_buffer_sz(self, print_desc=False):
        '''
        This function calculates the size of the on-chip FMs buffer.

        Returns:
        int: FMs buffer size.
        '''

        pass

    @abstractmethod
    def calc_off_chip_fms_access(self, print_desc=False):
        '''
        This function calculates the number of the off-chip FMs accesses.

        Returns:
        int: number of FMs accesses.
        '''

        pass

    @abstractmethod
    def get_segment_exec_times(self):
        '''
        This function returns the breakdown of per-segment execution time.

        Returns:
        list: per-segment execution time.
        '''

        pass

    @abstractmethod
    def calc_energy(self):
        pass
