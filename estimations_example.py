import mapping_utils.mapping_exec_utils as mapping_exec_utils
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
import mapping_utils.custom_mapping_utils as custom_mapping_utils
from preformance_record import *

model_name = 'resnet50'
model_dag = utils.read_model_dag_v2(
    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

board_name = 'zcu102'

dict = {
    "0-5": "0-5", "5-40": "1", "40-last": "2"
}
mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
    board_name, model_dag, dict)

record = mapping_exec_utils.run_mapping(
    board_name, model_name, mapping)

print(mapping.calc_exec_time(), 's')
print(mapping.calc_throughput(), 'FPS')
print(mapping.calc_on_chip_buffer_sz() / constants.MiB, 'MiB')
print((mapping.calc_off_chip_fms_access() + mapping.calc_off_chip_weights_access()) / constants.MiB, 'MiB')