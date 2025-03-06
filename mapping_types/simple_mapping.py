import __init__
import utils
import constants
from generic_mapping import GenericMapping

class SimpleMapping(GenericMapping):

    def calc_energy(self):
        #num_ops = sum(utils.get_layers_op_counts_by_indices(self.model_dag, self.layers))
        #weights_sz = sum(utils.get_weights_sizes_by_indices(self.model_dag, self.layers))
        #fms_sz = sum(utils.get_fms_sizes_by_indices(self.model_dag, self.layers))
        weights_buffer_brams = self.calc_weights_buffer_sz() / constants.BRAM_BLOCK_BYTES
        fms_buffer_brams = self.calc_fms_buffer_sz() / constants.BRAM_BLOCK_BYTES

        exec_time = self.calc_exec_time()

        bram_energy_weights = (self.hw_config.bram_r_pow * weights_buffer_brams) * \
            exec_time
            #(1 / self.hw_config.frequency) * weights_sz
        bram_energy_fms = (self.hw_config.bram_rw_pow * fms_buffer_brams / 2) * \
            exec_time
            #(1 / self.hw_config.frequency) * fms_sz
        bram_energy = bram_energy_weights + bram_energy_fms

        dsps_energy =  self.hw_config.dsp_pow * self.hw_config.num_pes * \
            exec_time
            #(1 / self.hw_config.frequency) * num_ops

        off_chip_access = self.calc_off_chip_fms_access() + self.calc_off_chip_weights_access()

        off_chip_access_energy = self.hw_config.off_chip_access_energy * off_chip_access

        return (dsps_energy + bram_energy + off_chip_access_energy)#, \
            # {'dsps_energy': dsps_energy, 'bram_energy_weights': bram_energy_weights,
            # 'bram_energy_fms': bram_energy_fms,
            #     'off_chip_access_energy': off_chip_access_energy}