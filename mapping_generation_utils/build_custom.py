
from mapping_types.generic_mapping import *
from mapping_types.sesl_mapping import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from mapping_types.hybrid_mapping import *
from mapping_types.hybrid_rr_mapping import *
from mapping_utils.custom_mapping_utils import *
import __init__
import mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
import experiments as exps
from preformance_record import *
from mapping_types.custom_mapping import *


mapping_configs_list = mapping_general_utils.read_mappings_json(constants.CUSTOM_MAPPING_FILE)

for current_mapping_configs in mapping_configs_list:
    #current_mapping_configs = mapping_configs_list[-1]

    model_name = current_mapping_configs['model_name']
    board_name = current_mapping_configs['board_name']
    model_dag = utils.read_model_dag_v2(
        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

    layers = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))

    mapping_details = current_mapping_configs['layer_engine_mapping']

    mappings_segments_config_list = infer_mapping_types(mapping_details, model_dag)
    hw_config = HWConfig(board_name)

    mapping2 = CustomMapping(hw_config, model_dag, layers, mappings_segments_config_list,
                            first_layer_ifms_are_on_chip=False,
                            last_layer_ofms_are_on_chip=False,
                            apply_fusion=False)

    print(mapping2.calc_throughput())
    print(mapping2.calc_on_chip_buffer_sz() / constants.MiB)
    print(mapping2.get_segment_exec_times())
    print('*****************')