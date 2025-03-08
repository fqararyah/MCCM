
import __init__
from generic_mapping import GenericMapping
import utils
from engines.engine import *


class SEMLMapping_LBL(GenericMapping):

    MAPPING_LABEL = 'SEML_LBL'
    def __init__(self, hw_config, model_dag, layers,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 exec_v2 = False):
        super().__init__(hw_config, layers, [],
                         first_layer_ifms_are_on_chip, last_layer_ofms_are_on_chip)
        has_dw_layers = utils.has_dw_layers(model_dag, layers[0], len(layers))
        engines = []
        if has_dw_layers:
            engines = [Engine(1), Engine(
                1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W)]
        else:
            engines = [Engine(hw_config.num_pes)]
        self.engines = engines
        self.model_dag = model_dag
        self.mapping_pes = hw_config.num_pes
        self.num_engines = len(engines)
        self.num_layers = len(layers)
        self.layers_exec_times = [-1] * self.num_layers
        self.layer_comp_times = [-1] * self.num_layers
        self.layers_off_chip_fms_access = [-1] * self.num_layers
        self.layers_off_chip_weight_access = [0] * self.num_layers
        self.exec_time = -1
        self.tmp_channels_buffer_sz = -1
        self.tmp_channels_data_sz = -1
        self.max_on_chip_fms_layers = -1
        self.exec_v2 = exec_v2
        self.allocate_and_balance_pes_to_engines(hw_config.num_pes)
        self.calc_layers_off_chip_weight_access()

    def get_label(self):
        return self.MAPPING_LABEL

    def get_pipe_num_passes(self):
        return 1

    def get_num_engines(self):
        if utils.has_dw_layers(self.model_dag, self.layers[0], len(self.layers)):
            return self.num_engines / 2

        return self.num_engines

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
            self.model_dag, self.layers, targeted_layer_type=['pw', 's'])

        if len(self.engines) > 1:
            self.engines[1].num_pes = num_pes - self.engines[0].num_pes
            self.engines[1].distribute_PEs_on_dims(
                self.model_dag, self.layers, targeted_layer_type=['dw'])

    def allocate_and_balance_pes_to_engines(self, num_pes):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        max_dw_ops = 0
        max_conv_ops = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                max_dw_ops = max(max_dw_ops, op_counts[i])
            else:
                max_conv_ops = max(max_conv_ops, op_counts[i])

        self.engines[0].num_pes = max(min(math.ceil(
            num_pes * max_conv_ops / (max_conv_ops + max_dw_ops)), num_pes - 1), 1)
        self.engines[0].distribute_PEs_on_dims(
            self.model_dag, self.layers, targeted_layer_type=['pw', 's'])

        if len(self.engines) > 1:
            self.engines[1].num_pes = num_pes - self.engines[0].num_pes
            self.engines[1].distribute_PEs_on_dims(
                self.model_dag, self.layers, targeted_layer_type=['dw'])

    def calc_layer_compute_times(self):
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                self.layer_comp_times[i] = \
                    self.engines[1].calc_layer_exec_time(
                        layer_specs) / self.hw_config.frequency
            else:
                self.layer_comp_times[i] = \
                    self.engines[0].calc_layer_exec_time(
                        layer_specs) / self.hw_config.frequency
        
        return self.layer_comp_times

    def calc_layer_off_chip_access_time(self, layer_index):
        return (self.layers_off_chip_fms_access[layer_index] + self.layers_off_chip_weight_access[layer_index]) / self.hw_config.bw

    def calc_off_chip_access_time_only(self):
        access_time = 0
        for i in range(len(self.layers)):
            access_time += self.calc_layer_off_chip_access_time(i)
        
        return access_time
        
    def calc_compute_time_only(self):
        return sum(self.calc_layer_compute_times())
        
    def calc_exec_time(self, print_desc = False):

        if self.exec_v2:
            return self.calc_exec_time_v2()
        
        self.calc_layer_compute_times()
        exec_time = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            self.layers_exec_times[i] = max(
                self.layer_comp_times[i], self.calc_layer_off_chip_access_time(i))
            # assuming engines are pipelined
            if utils.is_dw_layer(layer_specs):
                if i + 1 < len(self.layers):
                    exec_time += max(self.layers_exec_times[i],
                                     self.layers_exec_times[i + 1])
                else:
                    exec_time += self.layers_exec_times[i]
            elif i > 0 and not utils.is_dw_layer(self.model_dag[self.layers[i - 1]]):
                exec_time += self.layers_exec_times[i]

        self.exec_time = exec_time
        return exec_time
    
    def calc_exec_time_v2(self, print_desc = False):
        self.calc_layer_compute_times()
        exec_time = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            self.layers_exec_times[i] = max(
                self.layer_comp_times[i], self.calc_layer_off_chip_access_time(i))
        for i in range(len(self.layers)):
            # assuming engines are pipelined
            if utils.is_dw_layer(layer_specs):
                if i + 1 < len(self.layers):
                    exec_time += max(self.layers_exec_times[i],
                                     self.layers_exec_times[i + 1])
                else:
                    exec_time += self.layers_exec_times[i]
            elif (i > 0 and not utils.is_dw_layer(self.model_dag[self.layers[i - 1]]) or not utils.has_dw_layers(self.model_dag)):
                exec_time += self.layers_exec_times[i]

        self.exec_time = exec_time
        return exec_time
    
    def get_segment_exec_times():
        pass
    
    def calc_throughput(self):
        if self.exec_time == -1:
            self.exec_time = self.calc_exec_time()

        return 1 / self.exec_time

    def calc_layer_ifm_buffer_size(self, layer_specs):
        ifms_shape = utils.get_layer_ifms_shape(layer_specs)
        original_height = ifms_shape[1]
        original_width = ifms_shape[2]
        tiled_height = original_height
        tiled_width = original_width
        if original_height > self.engines[0].par_height and original_height % self.engines[0].par_height != 0:
            tiled_height = original_height + self.engines[0].par_height - (original_height % self.engines[0].par_height)
        if original_width > self.engines[0].par_width and original_width % self.engines[0].par_width != 0:
            tiled_width = original_width + self.engines[0].par_width - (original_width % self.engines[0].par_width)

        tiled_to_original_ratio = (tiled_height / original_height) * (tiled_width / original_width)

        return int(tiled_to_original_ratio * utils.get_layer_ifms_size(layer_specs))

    def calc_layer_ofm_buffer_size(self, layer_specs):
        ofms_shape = utils.get_layer_ofms_shape(layer_specs)
        original_height = ofms_shape[1]
        original_width = ofms_shape[2]
        tiled_height = original_height
        tiled_width = original_width
        if original_height > self.engines[0].par_height and original_height % self.engines[0].par_height != 0:
            tiled_height = original_height + self.engines[0].par_height - (original_height % self.engines[0].par_height)
        if original_width > self.engines[0].par_width and original_width % self.engines[0].par_width != 0:
            tiled_width = original_width + self.engines[0].par_width - (original_width % self.engines[0].par_width)

        tiled_to_original_ratio = (tiled_height / original_height) * (tiled_width / original_width)

        return int(tiled_to_original_ratio * utils.get_layer_ofms_size(layer_specs))
    
    def calc_fms_buffer_sz_intra(self):
        fms_buffer_sz = 0
        for i in range(len(self.layers) - 1):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ifm_ofm_size = max(utils.get_layer_ifms_size(
                layer_specs), utils.get_layer_ofms_size(layer_specs))
            ifm_ofm_size = self.calc_actual_bram_cons(
                ifm_ofm_size, self.engines[0].get_parallelism_fms())
            ifm_ofm_size *= 2
            if ifm_ofm_size < self.hw_config.on_chip_memory:
                if ifm_ofm_size > fms_buffer_sz:
                    self.max_on_chip_fms_layers = layer_index
                    
                fms_buffer_sz = max(fms_buffer_sz, ifm_ofm_size)

        return fms_buffer_sz

    def calc_tmp_fms_buffer_sz(self):
        fms_buffer_sz = self.calc_main_fms_buffer_sz()
        weights_buffer_sz = self.calc_weights_buffer_sz()
        tmp_channels_buffer_sz = 0
        for layer in self.layers:
            tmp_channels_sz = 0
            layer_specs = self.model_dag[layer]
            layer_chidren = utils.get_layer_children_with_fusion(
                self.model_dag, layer_specs)
            if len(layer_chidren) > 1:
                tmp_channels_sz = utils.get_layer_ofms_size(layer_specs)

            tmp_channels_buffer_sz = self.calc_actual_bram_cons(
                tmp_channels_sz, self.engines[0].get_parallelism_fms())
            if self.hw_config.on_chip_memory >= fms_buffer_sz + weights_buffer_sz + tmp_channels_buffer_sz:
                self.tmp_channels_buffer_sz = max(
                    self.tmp_channels_buffer_sz, tmp_channels_buffer_sz)
                
                self.tmp_channels_data_sz = max(
                    self.tmp_channels_data_sz, tmp_channels_sz)
            
        return self.tmp_channels_buffer_sz

    def calc_main_fms_buffer_sz(self):
        on_chip_memory = self.hw_config.on_chip_memory
        intra_fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        weights_buffer_sz = self.calc_weights_buffer_sz()

        first_layer_ifms_size = utils.get_layer_ifms_size(
            self.model_dag[self.layers[0]])
        last_layer_ofms_size = utils.get_layer_ofms_size(
            self.model_dag[self.layers[-1]])
        # The priority is for weights and intermediate results, as they are more on the critical path
        # if there is a space left, then ...
        avialable_on_chip_memoy = on_chip_memory - \
            (intra_fms_buffer_sz + weights_buffer_sz)
        iofms_on_chip_sz = \
            self.on_chip_buffer_sz_for_first_last_iofms(avialable_on_chip_memoy,
                                                        first_layer_ifms_size, last_layer_ofms_size)

        return max(intra_fms_buffer_sz, iofms_on_chip_sz)

    def calc_fms_buffer_sz(self, print_desc = False):
        if print_desc:
            print(self.MAPPING_LABEL, 'calc_fms_buffer_sz', self.calc_main_fms_buffer_sz(), self.calc_tmp_fms_buffer_sz())
        return self.calc_main_fms_buffer_sz() + self.calc_tmp_fms_buffer_sz()

    def calc_weights_buffer_sz(self):
        weights_buffer_sz = 0
        dw_weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            filter_dim = utils.get_filter_dim(layer_specs)
            if utils.is_dw_layer(layer_specs):
                weights_buffer_sz = max(weights_buffer_sz, utils.get_layer_ifms_shape(layer_specs)[0] * filter_dim * filter_dim *
                                        self.engines[1].par_ofms)
            else:
                dw_weights_buffer_sz = max(weights_buffer_sz, utils.get_layer_ifms_shape(layer_specs)[0] * filter_dim * filter_dim *
                                           self.engines[0].par_ofms)

        weights_buffer_sz = self.calc_actual_bram_cons(
            weights_buffer_sz, self.engines[0].get_parallelism_weights())

        if len(self.engines) > 1:
            dw_weights_buffer_sz = self.calc_actual_bram_cons(
                dw_weights_buffer_sz, self.engines[1].get_parallelism_weights())

        # double buffering
        return dw_weights_buffer_sz + 2 * weights_buffer_sz

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        return self.calc_total_weights()

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access_intra(self):
        weights_buffer_sz = self.calc_weights_buffer_sz()
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            layer_ifms_sz = utils.get_layer_ifms_size(
                layer_specs) if i > 0 else 0
            layer_ofms_sz = utils.get_layer_ofms_size(
                layer_specs) if i < len(self.layers) - 1 else 0
            ifm_ofm_size = max(layer_ifms_sz, layer_ofms_sz) * 2
            if ifm_ofm_size >= self.hw_config.on_chip_memory - weights_buffer_sz:
                self.layers_off_chip_fms_access[i] = ifm_ofm_size
            else:
                self.layers_off_chip_fms_access[i] = 0

    def calc_layers_off_chip_weight_access(self):
        for i in range(self.num_layers):
            self.layers_off_chip_weight_access[i] = utils.get_layer_weights_size(
                self.model_dag[self.layers[i]])

    def calc_off_chip_fms_access(self, print_desc = False):
        off_chip_accesses = sum(self.layers_off_chip_fms_access)
        fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        
        if off_chip_accesses < 0:
            self.calc_off_chip_fms_access_intra()
            off_chip_accesses = sum(self.layers_off_chip_fms_access)
        
        weights_buffer_sz = self.calc_weights_buffer_sz()
        available_on_chip = self.hw_config.on_chip_memory - \
            (weights_buffer_sz + fms_buffer_sz)

        layer_specs = self.model_dag[self.layers[0]]
        first_layer_ifms_sz = utils.get_layer_ifms_size(layer_specs)
        layer_specs = self.model_dag[self.layers[-1]]
        last_layer_ifms_sz = utils.get_layer_ofms_size(layer_specs)

        if self.off_chip_fms_access_of_first_and_last_layers < 0:
            self.on_chip_buffer_sz_for_first_last_iofms(available_on_chip,
                                                        first_layer_ifms_sz,
                                                        last_layer_ifms_sz)

        off_chip_accesses += self.off_chip_fms_access_of_first_and_last_layers

        return off_chip_accesses

    def get_off_chip_tmp_channels_layers(self):
        are_tmp_channels_on_chip = []
        if self.tmp_channels_data_sz < 0:
            self.calc_tmp_fms_buffer_sz()
        for layer in self.layers:
            layer_specs = self.model_dag[layer]
            layer_chidren = utils.get_layer_children_with_fusion(
                self.model_dag, layer_specs)
            if len(layer_chidren) > 1:
                if self.calc_actual_bram_cons(
                utils.get_layer_ofms_size(layer_specs), self.engines[0].get_parallelism_fms()) > self.tmp_channels_buffer_sz:
                    are_tmp_channels_on_chip.append(layer)

        return are_tmp_channels_on_chip