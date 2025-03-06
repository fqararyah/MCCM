import numpy as np
import copy
import math

def decide_break_points(data_point_groups, dependent_ponts_group, smaller_ratio= 0.25, larger_ratio = 0.25):
    break_point1 = 0
    break_point2 = 0
    num_groups = len(data_point_groups)
    max_num_points_in_group = max([len(data_point_groups[i]) for i in range(len(data_point_groups))])
    sorted_data_points = []
    sorted_dependent_points = []
    for i in range(num_groups):
        zipped_lists = zip(data_point_groups[i], dependent_ponts_group[i])
        zipped_lists = sorted(zipped_lists)
        sorted_points, sorted_dependent = zip(*zipped_lists)
        sorted_data_points.append(copy.copy(sorted_points))
        sorted_dependent_points.append(copy.copy(sorted_dependent))
    range_min = min([sorted_data_points[i][0] for i in range(num_groups)])
    range_max = max([sorted_data_points[i][-1] for i in range(num_groups)])
    step = (range_max - range_min) / max(1000, (max_num_points_in_group * num_groups))
    range_steps = np.arange(range_min, range_max, step)

    unique_points = {}
    num_unique_points = 0
    for i in range(num_groups):
        for j in range(max_num_points_in_group):
            if j >= len(sorted_data_points[i]):
                continue
            data_point = sorted_data_points[i][j]
            dependent_point = sorted_dependent_points[i][j]
            if data_point not in unique_points:
                unique_points[data_point] = {}
            if dependent_point not in unique_points[data_point]:
                num_unique_points += 1
                unique_points[data_point][dependent_point] = 0

    for i in range(len(range_steps)):
        points_so_far_smaller = 0
        if break_point1 != 0:
            break
        for k in range(0, max_num_points_in_group):
            at_least_one_point = False
            for j in range(num_groups): 
                if k >= len(sorted_data_points[j]):
                    continue
                data_point = sorted_data_points[j][k]
                dependent_point = sorted_dependent_points[j][k]
                if data_point <= range_steps[i]:
                    if unique_points[data_point][dependent_point] == 0:
                        points_so_far_smaller += 1
                        unique_points[data_point][dependent_point] = 1

                    at_least_one_point = True

            if not at_least_one_point:
                break
            
            if points_so_far_smaller >= num_unique_points * smaller_ratio:
                break_point1 = range_steps[i]
                break
        
        for data_point, dependent_dict_points in unique_points.items():
            for dependent_point, _ in dependent_dict_points.items():
                unique_points[data_point][dependent_point] = 0

    for data_point, dependent_dict_points in unique_points.items():
        for dependent_point, _ in dependent_dict_points.items():
            unique_points[data_point][dependent_point] = 0

    for i in range(len(range_steps) - 1, 0, -1):
        points_so_far_larger = 0
        if break_point2 != 0:
            break
        for k in range(max_num_points_in_group - 1, 0, -1):
            at_least_one_point = False
            for j in range(num_groups):  
                if k >= len(sorted_data_points[j]):
                    continue
                data_point = sorted_data_points[j][k]
                dependent_point = sorted_dependent_points[j][k]
                if data_point >= range_steps[i]:
                    if unique_points[data_point][dependent_point] == 0:
                        points_so_far_larger += 1
                        unique_points[data_point][dependent_point] = 1

                at_least_one_point = True

            if not at_least_one_point:
                break

            if points_so_far_larger >= num_unique_points * larger_ratio:
                break_point2 = range_steps[i]
                break
        
        for data_point, dependent_dict_points in unique_points.items():
            for dependent_point, _ in dependent_dict_points.items():
                unique_points[data_point][dependent_point] = 0

    return break_point1, break_point2, range_min, range_max


def decide_break_points_ratio_to_min_max(data_point_groups, 
                                         ratio_to_min= 2, ratio_to_max = 0.7):
    break_point1 = 0
    break_point2 = 0
    num_groups = len(data_point_groups)
    range_min = math.inf
    range_max = 0
    for i in range(num_groups):
        for j in range(len(data_point_groups[i])):
            range_min = min(range_min, min(data_point_groups[i]))
            range_max = max(range_max, max(data_point_groups[i]))
    
    break_point1 = range_min * ratio_to_min
    break_point2 = range_max * ratio_to_max

    return break_point1, break_point2, range_min, range_max

def decide_break_points_by_heursitic_results(data_point_groups):
    break_point1 = 0
    break_point2 = 0
    sorted_list = sorted(data_point_groups[-1])
    range_min = sorted_list[0]
    spread = sorted_list[-1] - sorted_list[0] 
    break_point1 = sorted_list[-1] + spread
    break_point2 = break_point1

    return break_point1, break_point2, range_min


def perf_dict_to_plot_dict(perf_dict):
    plot_dict = {}
    board_name_list = []
    model_name_list = []
    for board_name, model_names_dict in perf_dict.items():
        board_name_list.append(board_name)
        if len(model_name_list) == 0:
            model_name_list = list(model_names_dict.keys())
        for mapping_dict in model_names_dict.values():
            for mapping, val in mapping_dict.items():
                if mapping not in plot_dict:
                    plot_dict[mapping] = []
                plot_dict[mapping].append(val)

    return plot_dict, board_name_list, model_name_list
