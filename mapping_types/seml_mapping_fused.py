
from .generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *


class SEMLMapping_FUSED(GenericMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1

    def __init__(self, hw_config, model_dag, layers,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False):
        super().__init__(hw_config, layers, [],
                         first_layer_ifms_are_on_chip, last_layer_ofms_are_on_chip)
        has_dw_layers = utils.has_dw_layers(model_dag, layers[0], len(layers))
        engines = []
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.model_dag = model_dag
        if has_dw_layers:
            engines = [Engine(1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W)]
        else:
            engines = [Engine(hw_config.mapping_pes)]
        self.engines = engines
        self.layers_to_produce_row_counts = [
            self.DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS] * len(layers)
        self.calc_to_produce_row_counts()
        self.mapping_pes = hw_config.mapping_pes
        self.num_engines = len(engines)
        self.layers_exec_times = [-1] * len(layers)
        self.allocate_and_balance_pes_to_engines(hw_config.mapping_pes)

    def get_label(self):
        return 'SEML_FUSED'

    def calc_to_produce_row_counts(self):
        last_conv_layer_index = self.layers[-1]
        last_conv_layer_specs = self.model_dag[last_conv_layer_index]
        last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ofms_height = layer_specs['ofms_shape'][1]

            self.layers_to_produce_row_counts[i] = self.rows_to_produce_in_pipe_pass * ofms_height \
                / last_conv_layer_ofms_height

    def get_pipe_num_passes(self):
        last_conv_layer_index = self.layers[-1]
        last_conv_layer_specs = self.model_dag[last_conv_layer_index]
        last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]

        return last_conv_layer_ofms_height / self.rows_to_produce_in_pipe_pass

    def allocate_and_balance_pes_to_engines_unpipelined(self, num_pes):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        dw_ops = 0
        conv_ops = 0

        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                dw_ops += op_counts[i]
            else:
                conv_ops += op_counts[i]

        self.engines[0].num_pes = math.ceil(
            num_pes * conv_ops / (conv_ops + dw_ops))
        self.engines[0].distribute_PEs_on_dims(
            self.model_dag, self.layers, self.layers_to_produce_row_counts, targeted_layer_type=['pw', 's'])

        if len(self.engines) > 1:
            self.engines[1].num_pes = num_pes - self.engines[0].num_pes
            self.engines[1].distribute_PEs_on_dims(
                self.model_dag, self.layers, self.layers_to_produce_row_counts, targeted_layer_type=['dw'])

    def allocate_and_balance_pes_to_engines(self, num_pes):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        max_dw_ops = 0
        max_conv_ops = 0

        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            current_op_count = op_counts[i] * self.layers_to_produce_row_counts[i] / \
                utils.get_layer_ofms_shape(layer_specs)[1]
            if utils.is_dw_layer(layer_specs):
                max_dw_ops = max(max_dw_ops, current_op_count)
            else:
                max_conv_ops = max(max_conv_ops, current_op_count)
        self.engines[0].num_pes = math.ceil(
            num_pes * max_conv_ops / (max_conv_ops + max_dw_ops))
        self.engines[0].distribute_PEs_on_dims(
            self.model_dag, self.layers, self.layers_to_produce_row_counts, targeted_layer_type=['pw', 's'])
        if len(self.engines) > 1:
            self.engines[1].num_pes = num_pes - self.engines[0].num_pes
            self.engines[1].distribute_PEs_on_dims(
                self.model_dag, self.layers, self.layers_to_produce_row_counts, targeted_layer_type=['dw'])

    def calc_compute_time(self):
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                self.layers_exec_times[i] = \
                    self.engines[1].calc_layer_exec_time(layer_specs, self.layers_to_produce_row_counts[i]) * \
                    self.get_pipe_num_passes()
            else:
                self.layers_exec_times[i] = \
                    self.engines[0].calc_layer_exec_time(layer_specs, self.layers_to_produce_row_counts[i]) * \
                    self.get_pipe_num_passes()

        exec_time = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            # assuming engines are pipelined
            if utils.is_dw_layer(layer_specs):
                if i + 1 < len(self.layers):
                    exec_time += max(self.layers_exec_times[i],
                                     self.layers_exec_times[i + 1])
                else:
                    exec_time += self.layers_exec_times[i]
            elif i > 0 and not utils.is_dw_layer(self.model_dag[self.layers[i - 1]]):
                exec_time += self.layers_exec_times[i]

        return exec_time / self.hw_config.frequency

    def calc_throughput(self):
        if self.exec_time == -1:
            self.exec_time = self.calc_exec_time()
        
        return 1 / self.exec_time
    
    def calc_layer_ofm_buffer_size(self, layer_specs):
        return utils.get_layer_ofms_size(layer_specs) / self.get_pipe_num_passes()

    def calc_layer_ifm_buffer_size(self, layer_specs, layer_offset):
        strides = layer_specs['strides']
        filter_dim = utils.get_filter_dim(layer_specs)
        num_ifms_rows_in_pass = self.layers_to_produce_row_counts[layer_offset] * \
            strides + filter_dim - strides

        return utils.get_layer_ifms_size(layer_specs) * num_ifms_rows_in_pass \
            / utils.get_layer_ifms_shape(layer_specs)[2]

    # assuming that the first layer ifms and the last layer ofms are to be stored off-chip
    # if this is not the case, the buffers should be deducted by the owner(the caller script) of the mapping object
    def calc_fms_buffer_sz_intra(self):
        fms_buffer_sz = 0
        on_chip_memory = self.hw_config.on_chip_memory
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ifms_buffer_size = self.calc_layer_ifm_buffer_size(layer_specs, i)
            ofms_buffer_size = self.calc_layer_ofm_buffer_size(layer_specs)
            ifm_ofm_size = ifms_buffer_size + ofms_buffer_size
            if ifm_ofm_size < on_chip_memory:
                fms_buffer_sz = max(fms_buffer_sz, ifm_ofm_size)

        return fms_buffer_sz

    def calc_fms_buffer_sz(self, print_desc = False):
        fms_buffer_sz = 0
        on_chip_memory = self.hw_config.on_chip_memory
        first_layer_ifms_size = utils.get_layer_ifms_size(
            self.model_dag[self.layers[0]])
        last_layer_ofms_size = utils.get_layer_ofms_size(
            self.model_dag[self.layers[-1]])
        iofms_on_chip_sz = \
            self.on_chip_buffer_sz_for_first_last_iofms(
                on_chip_memory, first_layer_ifms_size, last_layer_ofms_size)
        fms_buffer_sz += iofms_on_chip_sz

        return self.calc_fms_buffer_sz_intra(on_chip_memory) + fms_buffer_sz

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = self.calc_fms_buffer_sz_intra(on_chip_memory)
        weights_buffer_sz = self.calc_total_weights()

        if fms_buffer_sz + weights_buffer_sz > on_chip_memory:
            return weights_buffer_sz * self.get_pipe_num_passes()

        return self.calc_total_weights()

    # rm stands for requiren minimum
    def calc_weights_buffer_sz_rm(self):
        weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            current_engine = self.engines[0]
            if utils.is_dw_layer(layer_specs):
                current_engine = self.engines[1]
            rm_weights_sz = utils.get_layer_weights_size(layer_specs) * \
                current_engine.par_ofms / \
                utils.get_layer_weights_shape(layer_specs)[0]
            weights_buffer_sz += rm_weights_sz

        return weights_buffer_sz

    def calc_weights_buffer_sz(self):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = self.calc_fms_buffer_sz_intra(on_chip_memory)
        full_weights_sz = self.calc_weights_buffer_sz_full_on_chip()
        if full_weights_sz + fms_buffer_sz < on_chip_memory:
            return full_weights_sz

        return self.calc_weights_buffer_sz_rm()

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc = False):
        on_chip_memory = self.hw_config.on_chip_memory
        weights_buffer_sz = self.calc_weights_buffer_sz(on_chip_memory)
        off_chip_accesses = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ifms_buffer_size = self.calc_layer_ifm_buffer_size(layer_specs, i)
            ofms_buffer_size = self.calc_layer_ofm_buffer_size(layer_specs)
            ifm_buffer_size = ifms_buffer_size + ofms_buffer_size
            if ifm_buffer_size >= on_chip_memory - weights_buffer_sz:
                off_chip_accesses += ifm_buffer_size

        off_chip_accesses += self.off_chip_fms_access_of_first_and_last_layers

        return off_chip_accesses
