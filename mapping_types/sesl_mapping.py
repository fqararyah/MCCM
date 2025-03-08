
import __init__
from generic_mapping import GenericMapping
import utils
from engines.engine import *
import constants
from basic_mapping import BasicMapping


class SESLMapping(BasicMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'SESL'

    def __init__(self, hw_config, model_dag, layers, engines=None,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 pre_balanced_engines=False,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 engine_parallelization_strategy=ParallelizationStrategies.OFMS_H_W):

        super().__init__(hw_config, model_dag, layers, engines,
                         first_layer_ifms_are_on_chip, last_layer_ofms_are_on_chip)
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.exec_time = -1
        self.num_engines = len(self.layers)
        self.engine_parallelization_strategy = engine_parallelization_strategy
        self.layers_to_produce_row_counts = [
            self.DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS] * self.num_engines
        self.layers_exec_times = [-1] * len(layers)
        self.calc_to_produce_row_counts()
        self.pre_balanced_engines = pre_balanced_engines
        if not self.pre_balanced_engines:
            self.initialize_engines()
            self.allocate_and_balance_pes_to_engines()

    def get_label(self):
        return self.MAPPING_LABEL

    def calc_to_produce_row_counts(self):
        last_conv_layer_index = self.layers[-1]
        last_conv_layer_specs = self.model_dag[last_conv_layer_index]
        last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ofms_height = layer_specs['ofms_shape'][1]

            self.layers_to_produce_row_counts[i] = self.rows_to_produce_in_pipe_pass * \
                int(round(ofms_height / last_conv_layer_ofms_height))

    def initialize_engines(self):
        if self.engines == None:
            self.engines = []
        for i in range(self.num_engines):
            if utils.is_dw_layer(self.model_dag[self.layers[i]]):
                self.engines.append(Engine(1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W))
            else:
                self.engines.append(Engine(1, parallelization_strategy = self.engine_parallelization_strategy))

    def get_pipe_num_passes(self):
        last_conv_layer_index = self.layers[-1]
        last_conv_layer_specs = self.model_dag[last_conv_layer_index]
        last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]

        return last_conv_layer_ofms_height / self.rows_to_produce_in_pipe_pass

    def allocate_and_balance_pes_to_engines(self):
        used_pes = self.num_engines
        all_pes = self.hw_config.num_pes

        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            self.layers_exec_times[i] = \
                self.engines[i].calc_layer_exec_time(
                    layer_specs, self.layers_to_produce_row_counts[i])

        while used_pes < all_pes:
            max_latency = max(self.layers_exec_times)
            max_latency_index = self.layers_exec_times.index(max_latency)
            layer_index = self.layers[max_latency_index]
            layer_specs = self.model_dag[layer_index]
            to_add_pes = self.engines[max_latency_index].num_pes
            if used_pes + to_add_pes < all_pes:
                self.engines[max_latency_index].num_pes += to_add_pes
                self.engines[max_latency_index].distribute_PEs_on_dims(self.model_dag, [self.layers[max_latency_index]],
                                                                       [self.layers_to_produce_row_counts[max_latency_index]])
                used_pes += to_add_pes
                self.layers_exec_times[max_latency_index] = \
                    self.engines[max_latency_index].calc_layer_exec_time(
                        layer_specs, self.layers_to_produce_row_counts[max_latency_index])
            else:
                break

    def calc_pipe_filling_time(self):
        pipe_filling_time = 0
        for i in range(1, self.num_engines):
            pipe_filling_time += max(self.layers_exec_times[0: i])

        return pipe_filling_time

    def calc_compute_time(self):
        pipe_filing_time = self.calc_pipe_filling_time()
        pipe_bottleneck = max(self.layers_exec_times)
        return (pipe_filing_time + pipe_bottleneck * self.get_pipe_num_passes()) / self.hw_config.frequency

    def calc_exec_time(self, print_desc = False):
        #print(self.calc_compute_time(), self.calc_off_chip_access_time())
        self.exec_time = max(self.calc_compute_time(),
                             self.calc_off_chip_access_time())
        return self.exec_time
    
    def get_num_engines(self):
        return self.num_engines
    
    def get_segment_exec_times():
        pass

    def calc_throughput(self):
        pipe_bottleneck = max(self.layers_exec_times)
        throughput = 1 / (pipe_bottleneck *
                          self.get_pipe_num_passes() / self.hw_config.frequency)
        return throughput

    def calc_fms_buffer_sz_intra(self):
        fms_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ifms_depth = utils.get_layer_ifms_shape(layer_specs)[0]
            [ofms_depth, _, ofms_width] = utils.get_layer_ofms_shape(
                layer_specs)
            filter_dim = utils.get_filter_dim(layer_specs)
            strides = layer_specs['strides']
            ofms_buffer_sz = ofms_depth * ofms_width * \
                self.layers_to_produce_row_counts[i]
            ofms_buffer_sz = self.calc_actual_bram_cons(ofms_buffer_sz, self.engines[i].get_parallelism_fms())
            if utils.get_layer_num_children(layer_specs) > 1:
                ofms_buffer_sz += 2 * ofms_buffer_sz
            ifms_buffer_sz = ifms_depth * (ofms_width * strides) * \
                (self.layers_to_produce_row_counts[i] - 1) * strides + filter_dim
            ifms_buffer_sz = self.calc_actual_bram_cons(ifms_buffer_sz, self.engines[i].get_parallelism_fms())
            fms_buffer_sz += ifms_buffer_sz + ofms_buffer_sz

        return fms_buffer_sz

    def calc_fms_buffer_sz(self, print_desc = False):
        on_chip_memory = self.hw_config.on_chip_memory
        intra_pipe_fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        weights_buffer_sz = self.calc_weights_buffer_sz()

        first_layer_ifms_size = utils.get_layer_ifms_size(
            self.model_dag[self.layers[0]])
        last_layer_ofms_size = utils.get_layer_ofms_size(
            self.model_dag[self.layers[-1]])
        # The priority is for weights and intermediate results, as they are more on the critical path
        # if there is a space left, then ...
        avialable_on_chip_memoy = on_chip_memory - \
            (intra_pipe_fms_buffer_sz + weights_buffer_sz)
        iofms_on_chip_sz = \
            self.on_chip_buffer_sz_for_first_last_iofms(avialable_on_chip_memoy,
                                                        first_layer_ifms_size, last_layer_ofms_size)
        iofms_on_chip_sz = self.calc_actual_bram_cons(iofms_on_chip_sz, self.engines[0].get_parallelism_fms())

        if print_desc:
            print(self.MAPPING_LABEL , ' calc_fms_buffer_sz ', intra_pipe_fms_buffer_sz, iofms_on_chip_sz)

        return intra_pipe_fms_buffer_sz + iofms_on_chip_sz

    # rm stands for requiren minimum
    def calc_weights_buffer_sz_rm(self):
        weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            rm_weights_sz = utils.get_layer_weights_size(layer_specs) * \
                self.engines[i].par_ofms / \
                utils.get_layer_weights_shape(layer_specs)[0]
            weights_buffer_sz += rm_weights_sz

        return weights_buffer_sz

    def calc_weights_buffer_sz_full_on_chip(self):
        weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            weights_buffer_sz += self.calc_actual_bram_cons(utils.get_layer_weights_size(layer_specs)
                                                         , self.engines[i].get_parallelism_weights())
        return weights_buffer_sz
    
    def calc_weights_buffer_sz(self):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        full_weights_sz = self.calc_weights_buffer_sz_full_on_chip()
        if full_weights_sz + fms_buffer_sz < on_chip_memory:
            return full_weights_sz

        return self.calc_weights_buffer_sz_rm()
    
    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        weights_buffer_sz = self.calc_total_weights()
        layer_Weights = self.calc_total_weights()
        if fms_buffer_sz + weights_buffer_sz > on_chip_memory:
            return layer_Weights * self.get_pipe_num_passes()

        return layer_Weights

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc = False):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        if self.off_chip_fms_access_of_first_and_last_layers < 0:
            # needed to upodate off_chip_fms_access_of_first_and_last_layers
            self.calc_fms_buffer_sz()
        weights_buffer_sz = self.calc_weights_buffer_sz()

        on_chip_memory -= fms_buffer_sz + weights_buffer_sz
        off_chip_access = 0

        if on_chip_memory <= 0:
            off_chip_access = self.calc_total_fms_in_layers(self.layers)

        off_chip_access += self.off_chip_fms_access_of_first_and_last_layers

        return off_chip_access
