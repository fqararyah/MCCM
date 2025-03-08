
from .generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from .seml_mapping_lbl import *
from .seml_mapping_fused import *
from .sesl_mapping import *
from .segment_grained_mapping_rr import *
from hw_config import *
import copy
from basic_mapping import BasicMapping

class HybridRRMapping(BasicMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    FIRST_PART_DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1

    def __init__(self, hw_config, model_dag, layers, segment_rr_mapping_num_engines,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_part_rows_to_produce_in_pipe_pass=FIRST_PART_DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 fuse_in_the_second_part=False):
        super().__init__(hw_config, model_dag, layers, [])
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.first_part_rows_to_produce_in_pipe_pass = first_part_rows_to_produce_in_pipe_pass
        self.all_pes = hw_config.num_pes
        self.num_layers = len(layers)
        self.segment_rr_mapping_num_engines = segment_rr_mapping_num_engines
        self.on_chip_memory = hw_config.on_chip_memory
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.fuse_in_the_second_part = fuse_in_the_second_part
        self.initialize_engines_and_mappings()

    def get_label(self):
        return 'Hetro'

    def initialize_engines_and_mappings(self):
        self.base_rr_mapping = (SegmentMappingRR(
                    self.hw_config, self.model_dag, self.layers, self.segment_rr_mapping_num_engines))
        base_latency = self.base_rr_mapping.calc_exec_time()
        base_access = self.base_rr_mapping.calc_off_chip_fms_access() + \
            self.base_rr_mapping.calc_off_chip_weights_access()
        segmented_rr_layers = self.num_layers
        second_last_segment_end_layer = segmented_rr_layers - (segmented_rr_layers % self.segment_rr_mapping_num_engines)
        best_latency = base_latency
        tolerable_latency_ratio = 2
        best_access = base_access
        sweet_spot = -1
        while second_last_segment_end_layer >= self.segment_rr_mapping_num_engines:
            self.spliting_point = second_last_segment_end_layer + 1
            if self.spliting_point >= self.num_layers:
                self.spliting_point = self.num_layers - self.segment_rr_mapping_num_engines
            self.initialize_segment_layers()
            self.initialize_balanced_segments_pes_and_buffers()
            self.initialize_mappings()
            latency = self.calc_exec_time()
            access = self.calc_off_chip_fms_access() + \
                self.calc_off_chip_weights_access()
            if latency <= best_latency and access <= best_access:#TODO add othe cases (trade-off)
                best_latency = latency
                best_access = access
                sweet_spot = self.spliting_point
            
            second_last_segment_end_layer -= self.segment_rr_mapping_num_engines
        
        if sweet_spot == -1:
            while second_last_segment_end_layer >= self.segment_rr_mapping_num_engines:
                self.spliting_point = second_last_segment_end_layer + 1
                if self.spliting_point >= self.num_layers:
                    self.spliting_point = self.num_layers - self.segment_rr_mapping_num_engines
                self.initialize_segment_layers()
                self.initialize_balanced_segments_pes_and_buffers()
                self.initialize_mappings()
                latency = self.calc_exec_time()
                access = self.calc_off_chip_fms_access() + \
                    self.calc_off_chip_weights_access()
                if latency <= min(best_latency, base_latency) * tolerable_latency_ratio and access < best_access:#TODO add othe cases (trade-off)
                    best_latency = latency
                    best_access = access
                    sweet_spot = self.spliting_point
        
        print(sweet_spot)
        self.spliting_point = sweet_spot
        self.initialize_segment_layers()
        self.initialize_balanced_segments_pes_and_buffers()
        self.initialize_mappings()
                    

    def initialize_segment_layers(self):
        self.first_part_layers = self.layers[0: self.spliting_point]
        self.second_part_layers = self.layers[self.spliting_point: self.num_layers]

    def initialize_balanced_segments_pes_and_buffers(self):
        op_counts = utils.get_layers_op_counts_by_indices(self.model_dag, self.layers)
        overall_op_count = sum(op_counts)
        first_part_op_count = sum(utils.get_layers_op_counts_by_indices(
            self.model_dag, self.first_part_layers))
        first_part_pes = int(
            first_part_op_count * self.all_pes / overall_op_count)
        if first_part_pes > self.base_rr_mapping.hw_config.num_pes:
            first_part_pes = self.base_rr_mapping.hw_config.num_pes
        self.first_part_hw_config = copy.deepcopy(self.hw_config)
        self.first_part_hw_config.num_pes = first_part_pes
        second_part_pes = self.all_pes - first_part_pes
        self.second_part_hw_config = copy.deepcopy(self.hw_config)
        self.second_part_hw_config.num_pes=second_part_pes

        # heuristic
        weights_szs = utils.get_weights_sizes(self.model_dag)
        ifms_szs = utils.get_ifms_sizes(self.model_dag)
        ofms_szs = utils.get_ofms_sizes(self.model_dag)
        # first paty contribution
        first_part_prefered_weights_buffer_sz = self.get_max_segment_weights_buffer_sz(weights_szs)
        first_part_prefered_fms_buffer_sz = self.get_max_segment_fms_buffer_sz(ifms_szs, ofms_szs)
        first_part_szs = first_part_prefered_fms_buffer_sz + first_part_prefered_weights_buffer_sz
        # second part contribution, in the secod part the FMs size is the dominant and weights size is negligable
        second_part_sz = max(max(ifms_szs[self.spliting_point: self.num_layers]), max(ofms_szs[self.spliting_point: self.num_layers])) * 2
        comibed_sz = first_part_szs + second_part_sz
        first_part_on_chip_mem = int(
            self.on_chip_memory * first_part_szs / comibed_sz)
        self.first_part_hw_config.on_chip_memory = first_part_on_chip_mem
        second_part_on_chip_mem = self.on_chip_memory - first_part_on_chip_mem
        self.second_part_hw_config.on_chip_memory = second_part_on_chip_mem

    def get_max_segment_weights_buffer_sz(self, weights_szs):
        max_weight_buffer_sz = 0
        for i in range(0, self.spliting_point, self.segment_rr_mapping_num_engines):
            max_weight_buffer_sz = max(max_weight_buffer_sz, sum(weights_szs[i: i + self.segment_rr_mapping_num_engines]))
            
        return max_weight_buffer_sz
    
    def get_max_segment_fms_buffer_sz(self, ifms_szs, ofms_szs):
        max_fms_buffer_sz = 0
        for i in range(0, self.spliting_point, self.segment_rr_mapping_num_engines):
            last_layer = min(self.num_layers - 1, i + self.segment_rr_mapping_num_engines - 1)
            max_fms_buffer_sz = max(max_fms_buffer_sz, max(ifms_szs[i], ofms_szs[last_layer]))
            
        return max_fms_buffer_sz
            
    def initialize_mappings(self):
        self.first_part_mapping = SegmentMappingRR(self.first_part_hw_config,
                                              self.model_dag, self.first_part_layers,
                                              self.segment_rr_mapping_num_engines,
                                              last_layer_ofms_are_on_chip=True)
        if self.fuse_in_the_second_part:
            self.second_part_mapping = SEMLMapping_FUSED(self.second_part_hw_config,
                                                         self.model_dag, self.second_part_layers,
                                                         self.rows_to_produce_in_pipe_pass, first_layer_ifms_are_on_chip=True)
        else:
            self.second_part_mapping = SEMLMapping_LBL(self.second_part_hw_config,
                                                       self.model_dag, self.second_part_layers, first_layer_ifms_are_on_chip=True)

    def calc_exec_time_semll_lbl(self):
        #print(self.first_part_mapping.calc_exec_time() / self.second_part_mapping.calc_exec_time())
        return self.first_part_mapping.calc_exec_time() + self.second_part_mapping.calc_exec_time()

    def calc_exec_time(self, print_desc = False):
        return self.calc_exec_time_semll_lbl()
    
    def calc_throughput(self):
        return 1 / max(self.first_part_mapping.calc_exec_time(), self.second_part_mapping.calc_exec_time())

    def get_first_part_on_chip_fms_buffer_sz(self):
        return self.first_part_mapping.calc_fms_buffer_sz_intra()

    def get_second_part_on_chip_fms_buffer_sz(self):
        return self.second_part_mapping.calc_fms_buffer_sz() + self.second_part_mapping.calc_tmp_fms_buffer_sz()

    def calc_fms_buffer_sz(self, print_desc = False):
        first_part_fms_buffer_sz = self.get_first_part_on_chip_fms_buffer_sz()
        second_part_fms_buffer_sz = self.get_second_part_on_chip_fms_buffer_sz()
        
        return first_part_fms_buffer_sz + second_part_fms_buffer_sz

    def calc_weights_buffer_sz(self):
        first_part_weights_buffer_sz = self.first_part_mapping.calc_weights_buffer_sz()
        second_part_weights_buffer_sz = self.second_part_mapping.calc_weights_buffer_sz()

        return first_part_weights_buffer_sz + second_part_weights_buffer_sz

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        first_part_weights_off_chip_access = self.first_part_mapping.calc_off_chip_weights_access()
        second_part_weights_off_chip_access = self.second_part_mapping.calc_off_chip_weights_access()

        return first_part_weights_off_chip_access + second_part_weights_off_chip_access

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc = False):
        first_part_off_chip_fms_access = self.first_part_mapping.calc_off_chip_fms_access()
        second_part_off_chip_fms_access = self.second_part_mapping.calc_off_chip_fms_access()
        return first_part_off_chip_fms_access + second_part_off_chip_fms_access

    # def get_pipe_num_passes(self):
    #     last_conv_layer_index = self.layers[-1]
    #     last_conv_layer_specs = self.model_dag[last_conv_layer_index]
    #     last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]

    #     return last_conv_layer_ofms_height / self.rows_to_produce_in_pipe_pass

    # def first_part_to_produce_row_count(self):
    #     if self.fuse_in_the_second_part:
    #         return utils.get_layer_ofms_shape(self.first_part_layers[-1])[1] / utils.get_layer_ofms_shape(self.layers[-1])[1]
    #     else:
    #         return self.first_part_rows_to_produce_in_pipe_pass

    # def calc_compute_time_seml_fused(self):
    #     pipe_num_passes = self.get_pipe_num_passes()
    #     first_part_pipe_filling_time = self.first_part_mapping.calc_pipe_filling_time()
    #     first_part_bottleneck = max(
    #         self.first_part_mapping.layers_exec_times) * self.first_part_to_produce_row_count()
    #     second_part_bottleneck = self.second_part_mapping.calc_compute_time() / \
    #         pipe_num_passes
    #     # one pass of the first part is part of the fillinf time for the second
    #     filling_time = first_part_pipe_filling_time + first_part_bottleneck

    #     return filling_time + pipe_num_passes * max(first_part_bottleneck, second_part_bottleneck)
