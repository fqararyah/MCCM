
import __init__
import utils
from engines.engine import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.sesl_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.mapping_description import *
from hw_config import *
import random
from mapping_types.custom_mapping import *


def prepare_custom_mapping_desc(segment_layers_list, segment_blocks_list, block_engines_list):
    mapping_desc = {}
    for segment_index in range(len(segment_layers_list)):
        segment_layers = segment_layers_list[segment_index]
        segment_engines = block_engines_list[segment_blocks_list[segment_index]]

        layers_str = str(segment_layers[0])
        if len(segment_layers) == 2:
            layers_str += '-' + str(segment_layers[1])
        engines_str = str(segment_engines[0])
        if len(segment_engines) == 2:
            engines_str += '-' + str(segment_engines[1])

        mapping_desc[layers_str] = engines_str

    return mapping_desc


def custom_mapping_from_desc_dict(board_name, model_dag, mapping_desc_dict, timing_metric=Metrics.LATENCY):
    layers = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))

    mappings_segments_config_list = infer_mapping_types(
        mapping_desc_dict, model_dag)
    hw_config = HWConfig(board_name)

    return CustomMapping(hw_config, model_dag, layers, mappings_segments_config_list, timing_metric=timing_metric)


def custom_mapping_from_desc(mapping_desc):
    mapping_desc_dict = custom_mapping_utils.prepare_custom_mapping_desc(
        mapping_desc.segment_layers_list, mapping_desc.segment_block_list, mapping_desc.block_engines_list)
    mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
        mapping_desc.board_name, mapping_desc.model_dag, mapping_desc_dict, timing_metric=mapping_desc.timing_metric)

    return mapping


def infer_mapping_types(mapping_details, model_dag):
    mappings_segments_list = []
    for key, val in mapping_details.items():
        engine_list = mapping_general_utils.clean_engine_range(val)
        layer_lits = mapping_general_utils.clean_layer_range(key, model_dag)
        if len(engine_list) > 1:
            if len(layer_lits) == len(engine_list):
                mapping_type = SESLMapping.MAPPING_LABEL
            else:
                mapping_type = SegmentMappingRR.MAPPING_LABEL
        else:
            mapping_type = SEMLMapping_LBL.MAPPING_LABEL

        mappings_segments_list.append(
            {'mapping': mapping_type, 'engine_list': engine_list, 'layer_list': layer_lits})

    return mappings_segments_list


def copy_segment_layers_or_engines(segment_src_list, segment_dst_list,
                                   starting_segment_in_src, starting_segment_in_dst,
                                   end_segment, add_offset=0):

    index_in_dst = starting_segment_in_dst
    assert index_in_dst <= len(segment_dst_list)
    # print(starting_segment_in_src, end_segment, len(segment_src_list))
    for i in range(starting_segment_in_src, end_segment):
        if index_in_dst < len(segment_dst_list):
            segment_dst_list[index_in_dst] = []
        else:
            segment_dst_list.append([])
        for layer in segment_src_list[i]:
            segment_dst_list[index_in_dst].append(layer + add_offset)

        index_in_dst += 1


def generate_random_mapping(board_name, model_name, model_dag, num_segments=-1, timing_metric=Metrics.LATENCY):

    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    max_engines = min(constants.MAX_ENGINES_V2, num_conv_layers)
    max_engines_intra = constants.MAX_ENGINES
    if num_segments == -1:
        num_segments = random.randint(
            constants.MIN_SEGMENTS, max_engines)
    num_blocks = num_segments  # random.randint(2, num_segments)
    segment_layers_list = []
    block_engines_list = []
    segment_blocks_list = [0] * num_segments
    block_segments_list = []
    unassigned_layers = num_conv_layers
    unassigned_segments = num_segments

    # segment layers and segment blocks
    for i in range(num_segments):
        unassigned_segments -= 1
        if i == num_segments - 1:
            current_segment_num_layers = unassigned_layers
        else:
            current_segment_num_layers = random.randint(
                1, unassigned_layers - unassigned_segments)
        first_layer = num_conv_layers - unassigned_layers
        segment_layers_list.append([first_layer])
        if current_segment_num_layers > 1:
            if unassigned_segments == 0:
                segment_layers_list[i].append(num_conv_layers)
            else:
                segment_layers_list[i].append(
                    first_layer + current_segment_num_layers)
        unassigned_layers -= current_segment_num_layers
        if unassigned_segments > num_blocks:
            segment_blocks_list[i] = random.randint(0, num_blocks - 1)

    if num_blocks != num_segments:
        last_segemetns_blocks = random.sample(range(num_blocks), num_blocks)
        segment_blocks_list.extend(last_segemetns_blocks)
    else:
        segment_blocks_list = [*range(num_blocks)]

    # blocks segments and blocks engines
    for i in range(num_blocks):
        block_segments_list.append([])
        block_engines_list.append([])

    for i in range(num_segments):
        block_segments_list[segment_blocks_list[i]].append(i)

    engines_so_far = 0
    for i in range(num_segments):
        block_index = segment_blocks_list[i]
        if len(block_engines_list[block_index]) != 0:
            continue
        else:
            min_segment_engines = max_engines_intra
            for block_segment in block_segments_list[block_index]:
                if len(segment_layers_list[block_segment]) == 1:
                    current_segment_layers = 1
                else:
                    current_segment_layers = segment_layers_list[block_segment][1] - \
                        segment_layers_list[block_segment][0]

                min_segment_engines = min(
                    min_segment_engines, current_segment_layers)

        block_type = random.random()
        if num_segments > 1 and block_type < 0.25:
            current_num_engines = 1
        else:
            min_engines_per_block = 2 if num_segments == 1 else 1
            current_num_engines = random.randint(
                min_engines_per_block, min_segment_engines)

        block_engines_list[block_index].append(engines_so_far)
        if current_num_engines > 1:
            block_engines_list[block_index].append(
                engines_so_far + current_num_engines)

        engines_so_far += current_num_engines

    # mapping
    mapping_desc = prepare_custom_mapping_desc(
        segment_layers_list, segment_blocks_list, block_engines_list)
    mapping = custom_mapping_from_desc_dict(
        board_name, model_dag, mapping_desc, timing_metric=timing_metric)
    mapping_desc = MappingDescription(board_name, model_name, model_dag, num_segments, num_blocks, segment_layers_list, block_engines_list,
                                      segment_blocks_list, timing_metric)

    return mapping, mapping_desc


def generate_random_mapping_for_sub_model(in_mapping_desc, layers, num_segments,
                                          prefered_block_types=None):

    num_conv_layers = len(layers)

    num_blocks = num_segments  # random.randint(2, num_segments)
    segment_layers_list = []
    block_engines_list = []
    segment_block_list = [0] * num_segments
    block_segments_list = []
    unassigned_layers = num_conv_layers
    unassigned_segments = num_segments
    max_engines = min(constants.MAX_ENGINES, num_conv_layers)

    # segment layers and segment blocks
    for i in range(num_segments):
        unassigned_segments -= 1
        if i == num_segments - 1:
            current_segment_num_layers = unassigned_layers
        else:
            current_segment_num_layers = random.randint(
                1, unassigned_layers - unassigned_segments)
        first_layer = num_conv_layers - unassigned_layers
        segment_layers_list.append([first_layer])
        if current_segment_num_layers > 1:
            if unassigned_segments == 0:
                segment_layers_list[i].append(num_conv_layers)
            else:
                segment_layers_list[i].append(
                    first_layer + current_segment_num_layers)
        unassigned_layers -= current_segment_num_layers

        if unassigned_segments > num_blocks:
            segment_block_list[i] = random.randint(0, num_blocks - 1)

    if num_blocks != num_segments:
        last_segemetns_blocks = random.sample(range(num_blocks), num_blocks)
        segment_block_list.extend(last_segemetns_blocks)
    else:
        segment_block_list = [*range(num_blocks)]

    # blocks segments and blocks engines
    for i in range(num_blocks):
        block_segments_list.append([])
        block_engines_list.append([])

    for i in range(num_segments):
        block_segments_list[segment_block_list[i]].append(i)

    engines_so_far = 0
    for i in range(num_segments):
        block_index = segment_block_list[i]
        if len(block_engines_list[block_index]) != 0:
            continue

        if prefered_block_types is not None and i < len(prefered_block_types):
            block_type = prefered_block_types[i]
        else:
            # 0 for single and 1 for pipelined
            block_type = random.random()

        if block_type < 0.25:
            current_num_engines = 1
        else:
            min_segment_engines = max_engines
            for block_segment in block_segments_list[block_index]:
                if len(segment_layers_list[block_segment]) == 1:
                    current_segment_layers = 1
                else:
                    current_segment_layers = segment_layers_list[block_segment][1] - \
                        segment_layers_list[block_segment][0]

                min_segment_engines = min(
                    min_segment_engines, current_segment_layers)
            current_num_engines = random.randint(1, min_segment_engines)

        block_engines_list[block_index].append(engines_so_far)
        if current_num_engines > 1:
            block_engines_list[block_index].append(
                engines_so_far + current_num_engines)

        engines_so_far += current_num_engines

    # mapping
    mapping_desc = prepare_custom_mapping_desc(
        segment_layers_list, segment_block_list, block_engines_list)

    mapping_desc = MappingDescription(board_name=in_mapping_desc.board_name, model_name=in_mapping_desc.model_name,
                                      num_segments=num_segments, num_blocks=num_blocks,
                                      segment_layers_list=segment_layers_list, block_engines_list=block_engines_list,
                                      segment_block_list=segment_block_list)

    return mapping_desc


def mapping_from_crossover_of_two_mappings(mapping_desc1, mapping_desc2, crossover_point):

    board_name = mapping_desc1.board_name
    model_name = mapping_desc1.model_name
    model_dag = mapping_desc1.model_dag
    num_segments = mapping_desc1.num_segments
    num_blocks = mapping_desc1.num_blocks
    segment_layers_list1 = mapping_desc1.segment_layers_list
    block_engines_list1 = mapping_desc1.block_engines_list
    segment_blocks_list1 = mapping_desc1.segment_block_list
    segment_layers_list2 = mapping_desc2.segment_layers_list
    block_engines_list2 = mapping_desc2.block_engines_list
    segment_blocks_list2 = mapping_desc2.segment_block_list

    # print(segment_layers_list1, segment_layers_list2, crossover_point)

    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))

    last_segment_of_first_part_has_one_layer = len(
        segment_layers_list1[crossover_point - 1]) == 1
    last_block_of_first_part_has_one_engine = len(
        block_engines_list1[crossover_point - 1]) == 1
    last_layer_in_first_part = segment_layers_list1[crossover_point - 1][-1] - (
        1 if not last_segment_of_first_part_has_one_layer else 0)
    first_layer_in_second_parent_second_part = segment_layers_list2[crossover_point][0]
    first_engine_in_first_parent_second_part = block_engines_list1[crossover_point][0]
    first_engine_in_second_parent_second_part = block_engines_list2[crossover_point][0]

    offspring_segment_layers_list = []
    offspring_block_engines_list = []

    copy_segment_layers_or_engines(segment_layers_list1,
                                   offspring_segment_layers_list, 0,
                                   len(offspring_segment_layers_list), crossover_point)
    copy_segment_layers_or_engines(
        block_engines_list1, offspring_block_engines_list, 0,
        len(offspring_block_engines_list), crossover_point)

    if first_layer_in_second_parent_second_part == last_layer_in_first_part + 1:
        copy_segment_layers_or_engines(
            segment_layers_list2, offspring_segment_layers_list, crossover_point,
            len(offspring_segment_layers_list), num_segments)
        engine_offset = first_engine_in_first_parent_second_part - \
            first_engine_in_second_parent_second_part
        copy_segment_layers_or_engines(
            block_engines_list2, offspring_block_engines_list, crossover_point,
            len(offspring_block_engines_list), num_segments, add_offset=engine_offset)
    elif first_layer_in_second_parent_second_part > last_layer_in_first_part + 1:
        engine_offset = first_engine_in_first_parent_second_part - \
            first_engine_in_second_parent_second_part
        new_first_layer = last_layer_in_first_part + 1
        second_parent_crossover_last_layer = segment_layers_list2[crossover_point][-1]
        if len(segment_layers_list2[crossover_point]) == 1 and new_first_layer != second_parent_crossover_last_layer:
            second_parent_crossover_last_layer += 1
        offspring_segment_layers_list.append([
            new_first_layer, second_parent_crossover_last_layer])
        copy_segment_layers_or_engines(
            segment_layers_list2, offspring_segment_layers_list, crossover_point + 1,
            len(offspring_segment_layers_list), num_segments)
        copy_segment_layers_or_engines(
            block_engines_list2, offspring_block_engines_list, crossover_point,
            len(offspring_block_engines_list), num_segments, add_offset=engine_offset)
    else:
        # print(offspring_segment_layers_list)
        last_segment_to_keep_intact = num_segments
        remaining_layers = 0
        remaining_segments = -1
        current_first_layer = num_conv_layers
        while last_segment_to_keep_intact > crossover_point and remaining_layers > remaining_segments:
            last_segment_to_keep_intact -= 1
            prev_first_layer = current_first_layer
            current_first_layer = segment_layers_list2[last_segment_to_keep_intact][0]
            remaining_layers = current_first_layer - \
                (last_layer_in_first_part + 1)
            remaining_segments = last_segment_to_keep_intact - crossover_point
            if remaining_layers < remaining_segments:
                current_first_layer = prev_first_layer
                last_segment_to_keep_intact += 1

        prefered_block_types = []
        for segment_index in range(crossover_point, last_segment_to_keep_intact):
            prefered_block_types.append(
                len(block_engines_list2[segment_index]) != 1)

        intermediate_mapping_desc = generate_random_mapping_for_sub_model(mapping_desc1,
                                                                          layers=[
                                                                              *range(last_layer_in_first_part + 1, current_first_layer)],
                                                                          num_segments=last_segment_to_keep_intact - crossover_point,
                                                                          prefered_block_types=prefered_block_types)

        intermediate_mapping_segment_layer_list = intermediate_mapping_desc.segment_layers_list
        intermediate_mapping_block_engine_list = intermediate_mapping_desc.block_engines_list
        new_first_layer = last_layer_in_first_part + 1
        # print('1', offspring_segment_layers_list)
        copy_segment_layers_or_engines(
            intermediate_mapping_segment_layer_list, offspring_segment_layers_list,
            0, crossover_point, len(intermediate_mapping_segment_layer_list), add_offset=new_first_layer)
        # print('2', offspring_segment_layers_list)
        copy_segment_layers_or_engines(
            segment_layers_list2, offspring_segment_layers_list, len(
                offspring_segment_layers_list),
            last_segment_to_keep_intact, num_segments)

        engine_offset = offspring_block_engines_list[-1][-1] + (
            1 if last_block_of_first_part_has_one_engine else 0)
        copy_segment_layers_or_engines(intermediate_mapping_block_engine_list,  offspring_block_engines_list,
                                       0, crossover_point,
                                       len(intermediate_mapping_block_engine_list), add_offset=engine_offset)
        # print('2', offspring_block_engines_list)
        if last_segment_to_keep_intact < num_segments:
            first_engine_in_second_parent_second_part = block_engines_list2[
                last_segment_to_keep_intact][0]
            expected_first_engine_in_second_part = offspring_block_engines_list[-1][-1] + (
                1 if len(offspring_block_engines_list[-1]) == 1 else 0)
            engine_offset = expected_first_engine_in_second_part - \
                first_engine_in_second_parent_second_part
            copy_segment_layers_or_engines(block_engines_list2, offspring_block_engines_list,
                                           len(offspring_block_engines_list),
                                           last_segment_to_keep_intact, num_segments,
                                           add_offset=engine_offset)
        # print('3', offspring_block_engines_list)
    # mapping
    offspring_segment_blocks_list = []
    for i in range(num_segments):
        offspring_segment_blocks_list.append(i)
    mapping_desc_dict = prepare_custom_mapping_desc(
        offspring_segment_layers_list, offspring_segment_blocks_list, offspring_block_engines_list)
    # print(mapping_desc_dict)
    if not mapping_general_utils.validate_mapping_dict(mapping_desc_dict, model_dag):
        parent1_mapping_dict = prepare_custom_mapping_desc(
            segment_layers_list1, segment_blocks_list1, block_engines_list1)
        parent2_mapping_dict = prepare_custom_mapping_desc(
            segment_layers_list2, segment_blocks_list2, block_engines_list2)
        print('mapping_desc_dict', mapping_desc_dict)
        print('parent1_mapping_dict', parent1_mapping_dict)
        print('parent1_mapping_dict', parent2_mapping_dict)
        print(crossover_point)
    mapping = custom_mapping_from_desc_dict(
        board_name, model_dag, mapping_desc_dict, timing_metric=mapping_desc1.timing_metric)
    mapping_desc = MappingDescription(board_name, model_name, model_dag, num_segments, num_blocks,
                                      offspring_segment_layers_list, offspring_block_engines_list,
                                      offspring_segment_blocks_list, mapping_desc1.timing_metric)

    return mapping, mapping_desc


def modeify_mapping_random(mapping_desc):

    board_name = mapping_desc.board_name
    model_name = mapping_desc.model_name
    model_dag = mapping_desc.model_dag
    num_segments = mapping_desc.num_segments
    num_blocks = mapping_desc.num_blocks
    segment_layers_list = mapping_desc.segment_layers_list
    block_engines_list = mapping_desc.block_engines_list

    segment_to_modify = random.randint(0, num_segments - 1)
    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))

    orig_mapping_dict = prepare_custom_mapping_desc(
        segment_layers_list, mapping_desc.segment_block_list, block_engines_list)

    # expand or shrink the segment layers depending on its adjacent segments
    segment_first_layer = segment_layers_list[segment_to_modify][0]
    segment_last_layer = segment_layers_list[segment_to_modify][-1]
    new_first_layer = segment_first_layer
    new_last_layer = segment_last_layer
    originally_one_layer = segment_last_layer == segment_first_layer
    prev_segment = segment_to_modify - 1
    prev_segment_first_layer = 0
    if prev_segment >= 0:
        prev_segment_first_layer = segment_layers_list[prev_segment][0]
    next_segment = segment_to_modify + 1
    next_segment_last_layer = num_conv_layers
    end_rand_range = next_segment_last_layer
    if next_segment < num_segments:
        next_segment_last_layer = segment_layers_list[next_segment][-1]
        if len(segment_layers_list[next_segment]) == 1:
            end_rand_range = next_segment_last_layer
        else:
            end_rand_range = min(
                next_segment_last_layer - 1, num_conv_layers - 1)

    if prev_segment >= 0 and prev_segment_first_layer + 1 < end_rand_range - 1:
        new_first_layer = random.randint(
            prev_segment_first_layer + 1, end_rand_range - 1)
    if next_segment < num_segments and new_first_layer < end_rand_range:
        new_last_layer = random.randint(
            new_first_layer, end_rand_range)

    # print(num_segments, prev_segment, segment_to_modify, next_segment)
    # print(segment_first_layer, segment_last_layer, prev_segment_first_layer,
    #      next_segment_last_layer, new_first_layer, new_last_layer)

    assert new_first_layer <= new_last_layer

    segment_layers_list[segment_to_modify] = [new_first_layer]

    # if a sigment has originally one layer, its original last layer was inclusive
    # in this case if the modification was from the previous layers,
    # then the last layer is updated to make it exclusive
    if originally_one_layer and new_last_layer == segment_last_layer and new_first_layer != segment_first_layer:
        new_last_layer += 1

    if new_last_layer <= new_first_layer + 1:
        new_last_layer = new_first_layer
    else:
        segment_layers_list[segment_to_modify].append(new_last_layer)

    # print(num_segments, prev_segment, segment_to_modify, next_segment)
    # print(segment_first_layer, segment_last_layer, prev_segment_first_layer,
    #       next_segment_last_layer, new_first_layer, new_last_layer)

    if prev_segment >= 0:
        prev_segment_new_last_layer = new_first_layer
        assert prev_segment_new_last_layer >= prev_segment_first_layer

        if prev_segment_new_last_layer <= prev_segment_first_layer + 1:
            segment_layers_list[prev_segment] = [prev_segment_first_layer]
        else:
            segment_layers_list[prev_segment] = [
                prev_segment_first_layer, prev_segment_new_last_layer]

    if next_segment < num_segments:
        if new_first_layer == new_last_layer:
            next_segment_new_first_layer = new_last_layer + 1
        else:
            next_segment_new_first_layer = new_last_layer

        assert next_segment_last_layer >= next_segment_new_first_layer

        # originally one layer
        if len(segment_layers_list[next_segment]) == 1:
            next_segment_last_layer += 1
        if next_segment_last_layer <= next_segment_new_first_layer + 1:
            segment_layers_list[next_segment] = [next_segment_new_first_layer]
        else:
            segment_layers_list[next_segment] = [
                next_segment_new_first_layer, next_segment_last_layer]

    if len(segment_layers_list[-1]) > 1:
        segment_layers_list[-1][1] = num_conv_layers
    # modify engines
    block_type = random.random() 
    new_num_layers = max(1, new_last_layer - new_first_layer)
    original_engne_count = 1
    if len(block_engines_list[segment_to_modify]) > 1:
        original_engne_count = block_engines_list[segment_to_modify][1] - \
            block_engines_list[segment_to_modify][0]

    if num_segments > 1 and block_type < 0.25:
        new_engine_count = 1
    else:
        min_engines_per_block = 2 if num_segments == 1 else 1
        new_engine_count = random.randint(
            min_engines_per_block, min(constants.MAX_ENGINES_V2, new_num_layers))

    engine_diff = new_engine_count - original_engne_count

    # adjust previous block engines to account for shrinking its layers
    overflow = 0

    if prev_segment >= 0:
        prev_segment_num_layers = 1
        if len(segment_layers_list[prev_segment]) > 1:
            prev_segment_num_layers = segment_layers_list[prev_segment][1] - \
                segment_layers_list[prev_segment][0]
        prev_block_num_engines = 1
        if len(block_engines_list[prev_segment]) > 1:
            prev_block_num_engines = block_engines_list[prev_segment][1] - \
                block_engines_list[prev_segment][0]
        overflow = max(0, prev_block_num_engines - prev_segment_num_layers)
        if overflow > 0:
            block_engines_list[prev_segment] = [
                block_engines_list[prev_segment][0]]
            if prev_block_num_engines - overflow > 1:
                block_engines_list[prev_segment].append(
                    block_engines_list[prev_segment][0] + prev_block_num_engines - overflow)

            engine_diff -= overflow

    # adjust segment_to_modify engines
    block_engines_list[segment_to_modify] = [
        block_engines_list[segment_to_modify][0] - overflow]
    if new_engine_count > 1:
        block_engines_list[segment_to_modify].append(
            block_engines_list[segment_to_modify][0] + new_engine_count)

    # adjust next block engines to account for shrinking its layers
    if next_segment < num_segments:
        block_engines_list[next_segment][0] += engine_diff
        if len(block_engines_list[next_segment]) > 1:
            block_engines_list[next_segment][1] += engine_diff
        next_segment_num_layers = 1
        if len(segment_layers_list[next_segment]) > 1:
            next_segment_num_layers = segment_layers_list[next_segment][1] - \
                segment_layers_list[next_segment][0]
        next_block_num_engines = 1
        if len(block_engines_list[next_segment]) > 1:
            next_block_num_engines = block_engines_list[next_segment][1] - \
                block_engines_list[next_segment][0]
        overflow = max(0, next_block_num_engines - next_segment_num_layers)
        if overflow > 0:
            engine_diff -= overflow
            block_engines_list[next_segment][-1] -= overflow
            if block_engines_list[next_segment][-1] <= block_engines_list[next_segment][0] + 1:
                block_engines_list[next_segment] = [
                    block_engines_list[next_segment][0]]

    # adjust the following blocks engines
    for i in range(segment_to_modify + 2, num_blocks):
        block_engines_list[i][0] += engine_diff
        if len(block_engines_list[i]) > 1:
            block_engines_list[i][1] += engine_diff

    # mapping
    segment_blocks_list = []
    for i in range(num_segments):
        segment_blocks_list.append(i)

    mapping_desc_dict = prepare_custom_mapping_desc(
        segment_layers_list, [*range(num_blocks)], block_engines_list)

    mapping = custom_mapping_from_desc_dict(
        board_name, model_dag, mapping_desc_dict, timing_metric=mapping_desc.timing_metric)
    timing_metric = mapping_desc.timing_metric
    ret_mapping_desc = MappingDescription(board_name, model_name, model_dag, num_segments, num_blocks, segment_layers_list, block_engines_list,
                                          segment_blocks_list, timing_metric)

    #print(orig_mapping_dict ,'>\n', mapping_desc_dict)

    return mapping, ret_mapping_desc


def proportional_allocation(num_pes, op_counts_list):

    engines_pes = []
    alpha = 0
    sum_roots = 0
    for i in range(len(op_counts_list)):
        sum_roots += math.sqrt(op_counts_list[i])

    alpha = (sum_roots / num_pes) ** 2

    for i in range(len(op_counts_list)):
        engines_pes.append(max(1, int(math.sqrt(op_counts_list[i] / alpha))))

    return engines_pes
