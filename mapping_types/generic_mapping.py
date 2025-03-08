
from abc import ABC, abstractmethod
import __init__
import utils
import constants
import math
import mapping_utils
import mapping_utils.mapping_general_utils as mapping_utils

class GenericMapping(ABC):
    #main design overheads plus internal temporary buffers
    EXTRA_MEMORY_OVERHEADS_W = 0#0.05
    EXTRA_MEMORY_OVERHEADS_FM = 0#0.05
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB
    def __init__(self, hw_config, layers=None, engines=None, first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False) -> None:
        super().__init__()
        self.hw_config = hw_config
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
    def get_label(self):
        pass

    @abstractmethod
    def calc_weights_buffer_sz(self):
        pass

    @abstractmethod
    def calc_off_chip_weights_access(self):
        pass

    @abstractmethod
    def calc_off_chip_fms_access(self, print_desc = False):
        pass

    @abstractmethod
    def calc_fms_buffer_sz(self, print_desc = False):
        pass
    
    @abstractmethod
    def get_segment_exec_times(self):
        pass
    
    @abstractmethod
    def get_num_engines(self):
        pass

    def get_engines(self):
        return self.engines
    
    def get_off_chip_tmp_channels_layers(self):
        return []
    
    def get_pipe_num_passes(self):
        return 1

    def get_off_total_chip_access(self):
        return self.calc_off_chip_weights_access() + self.calc_off_chip_fms_access()

    def get_off_chip_bw(self):
        return self.calc_off_total_chip_access() / self.calc_latency()

    def calc_off_chip_access_time(self):
        off_chip_fms_access = self.calc_off_chip_fms_access()
        off_chip_weight_access = self.calc_off_chip_weights_access()

        return (off_chip_fms_access + off_chip_weight_access) / self.hw_config.bw

    def calc_fms_buffer_sz_intra(self):
        return 0

    def calc_total_weights(self):
        weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            weights_buffer_sz += utils.get_layer_weights_size(layer_specs)

        return weights_buffer_sz

    def calc_weights_buffer_sz_full_on_chip(self):
        return self.calc_total_weights()

    def calc_total_weights_in_layers(self, layers):
        weights_buffer_sz = 0
        for layer_index in layers:
            layer_specs = self.model_dag[layer_index]
            weights_buffer_sz += utils.get_layer_weights_size(layer_specs)

        return weights_buffer_sz

    def calc_total_fms_in_layers(self, layers):
        weights_buffer_sz = 0
        for layer_index in layers:
            layer_specs = self.model_dag[layer_index]
            weights_buffer_sz += utils.get_layer_ifms_size(layer_specs) + \
                utils.get_layer_ofms_size(layer_specs)

        return weights_buffer_sz

    def on_chip_buffer_sz_for_first_last_iofms(self, available_on_chip_memory,
                                               first_layer_ifms_sz, last_layer_ofms_sz):
        current_segment_on_chip_buffer_sz = 0

        if not self.first_layer_ifms_are_on_chip and not self.last_layer_ofms_are_on_chip:
            max_ifms_ofms = 0
            min_ifms_ofms = 0
        elif not self.first_layer_ifms_are_on_chip:
            max_ifms_ofms = last_layer_ofms_sz
            min_ifms_ofms = 0
        elif not self.last_layer_ofms_are_on_chip:
            max_ifms_ofms = first_layer_ifms_sz
            min_ifms_ofms = 0
        else:
            min_ifms_ofms = min(first_layer_ifms_sz, last_layer_ofms_sz)
            max_ifms_ofms = max(first_layer_ifms_sz, last_layer_ofms_sz)

        if available_on_chip_memory > max_ifms_ofms:
            current_segment_on_chip_buffer_sz = max_ifms_ofms
        if available_on_chip_memory - current_segment_on_chip_buffer_sz > min_ifms_ofms:
            current_segment_on_chip_buffer_sz += min_ifms_ofms
            
        if current_segment_on_chip_buffer_sz > max_ifms_ofms or current_segment_on_chip_buffer_sz == first_layer_ifms_sz:
            self.first_layer_on_chip_buffer = first_layer_ifms_sz

        self.off_chip_fms_access_of_first_and_last_layers = first_layer_ifms_sz + last_layer_ofms_sz - \
            current_segment_on_chip_buffer_sz

        return current_segment_on_chip_buffer_sz

    def calc_actual_bram_cons(self, buffer_size, parallelsim, data_bit_width = constants.BIT_WIDTH):
        if buffer_size < 0.5 * constants.KiB:
            return buffer_size
        
        split_size = int(math.ceil(buffer_size / parallelsim))
        blocks_per_bank = int(
            math.ceil(split_size / mapping_utils.get_bram_block_used_byes(data_bit_width)))
        
        return blocks_per_bank * parallelsim * constants.BRAM_BLOCK_BYTES
    
    def calc_on_chip_buffer_sz_pure(self):
        return self.calc_fms_buffer_sz() + self.calc_weights_buffer_sz()
    
    def calc_on_chip_buffer_sz(self):
        buffer_sz = self.calc_fms_buffer_sz() * (1 + self.EXTRA_MEMORY_OVERHEADS_FM) + \
            self.calc_weights_buffer_sz() * (1 + self.EXTRA_MEMORY_OVERHEADS_W) + self.EXTRA_MEMORY_OVERHEADS_CONST
        return  buffer_sz
