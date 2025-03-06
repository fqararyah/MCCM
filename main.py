from plot import plot_results
from mapping_types.generic_mapping import *
from mapping_types.sesl_mapping import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from mapping_types.hybrid_mapping import *
from mapping_types.hybrid_rr_mapping import *
import __init__
import mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
import experiments as exps
from preformance_record import *
import constants
import generate_custom_mappings_json as gen_mappings
import os
import time
import paper_specific_exps

# tmp = ['resnet50', 'resnet152', 'mob_v2', 'dense121', 'xce_r']#vgg16
# for model_name in tmp:
#     model_dag = utils.read_model_dag_v2(
#                 constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
#     print(utils.get_num_conv_layer_count_in_range(model_dag, 0, len(model_dag)))
# exit(0)

min_engines = 2
max_engines = 11
board_names = mapping_general_utils.read_board_names(file_name=constants.HW_CONFIGS_FILE_v1)
print(board_names)
model_names = ['resnet152', 'resnet50', 'xce_r', 'dense121', 'mob_v2']

plot_bests = False
plot_all_metric_pairs = False
custom_mappings = False
run_paper_specific_exps = True


all_metrics = [constants.metric_display_names[Metrics.LATENCY],
                                           constants.metric_display_names[Metrics.THROUGHPUT],constants.metric_display_names[Metrics.ACCESS],
                                           constants.metric_display_names[Metrics.BUFFER]]

paper_specific_exps.des_section_bottlenecks_v2()
if run_paper_specific_exps:
    paper_specific_exps.fine_grained()
    paper_specific_exps.des_section_bottlenecks_v2()
    paper_specific_exps.des_section_bottlenecks()
    paper_specific_exps.generate_best_mappings_table(board_names, model_names, constants.MIN_ENGINES, constants.MAX_ENGINES, all_metrics)


def all_models_boards_and_engines(plot):
    exps.engines_and_metrics(board_names, model_names,
                             min_engines, max_engines, plot)

def all_metric_pairs(plot):
    exps.metric_pairs(board_names, model_names, min_engines, max_engines, plot)


def metric_pairs(plot, board_names, model_names, custom_perf_data=None, plot_line = None):
    exps.metric_pairs(board_names, model_names, min_engines,
                      max_engines, plot, custom_perf_data, plot_line=plot_line)

#paper_specific_exps.bottlenecks()

def generate_and_evaluate_custom_mappings(custom_mapping_key, board_name, model_name, num_instances):

    custom_perf_data = {}
    gen_mappings.generate_hetero_sesl_seg(board_name, model_name,
                                          [constants.MIN_ENGINES,
                                              constants.MIN_ENGINES],
                                          [constants.MAX_ENGINES - constants.MIN_ENGINES,
                                           constants.MAX_ENGINES - constants.MIN_ENGINES], num_instances)
    perf_data_file = constants.FIGURES_DATA_DIR + '/custom_mappings/performance_{}_{}_{}.json'.format(
        board_name, model_name, num_instances)
    mapping_configs_file = constants.FIGURES_DATA_DIR + '/custom_mappings/mapping_configs_{}_{}_{}.json'.format(
        board_name, model_name, num_instances)
    
    print(perf_data_file)
    if not os.path.exists(perf_data_file):
        mapping_configs_list = mapping_general_utils.read_mappings_json(constants.CUSTOM_MAPPINGS_JSON_DIR + '{}_{}/{}_{}.json'.format(
            custom_mapping_key, num_instances, board_name, model_name))
        t0 = time.time()
        exps.eval_custom_mappings(board_name, model_name, mapping_configs_list)
        print('eval time is:', time.time() - t0)

    with open(perf_data_file, 'r') as f:
        print('perf_data_file ALREADY EXISTS!!!')
        custom_perf_data = json.load(f)

    return custom_perf_data
    
perf_data = None
if custom_mappings:
    board_name = 'vcu110'
    model_name = 'xce_r'
    custom_mapping_key = 'hetero_sesl_seg'
    num_instances = 100000
    perf_data = generate_and_evaluate_custom_mappings(custom_mapping_key, board_name, model_name, num_instances)
    metric_pairs(True, [board_name], [model_name], custom_perf_data= perf_data)

if plot_all_metric_pairs:
    all_metric_pairs(True)

if plot_bests:
    exps.best_perf_across_model_and_board(board_names, model_names, min_engines, max_engines,
                                          [constants.metric_display_names[Metrics.LATENCY],
                                           constants.metric_display_names[Metrics.THROUGHPUT]],
                                          from_file=False,
                                          plot_the_normalization_base_as_bars=False)

    model_names = ['resnet152', 'mob_v2']
    exps.best_perf_across_model_and_board(board_names, model_names, min_engines, max_engines,
                                          [constants.metric_display_names[Metrics.ACCESS]], from_file=False)

    board_names = ['zc706']
    exps.best_access_breakdown_across_model_and_board(board_names, model_names, min_engines, max_engines,
                                                      ['FMs access', 'Weights access'])

    board_names = ['vcu108']
    exps.best_access_breakdown_across_model_and_board(board_names, model_names, min_engines, max_engines,
                                                      ['FMs buffers', 'Weights buffers'])