import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import mapping_utils.mapping_general_utils as mapping_general_utils
import json
from matplotlib.ticker import MultipleLocator
import constants as consts
import plot_utils as plot_utils

path = os.path.dirname(__file__)


def scatter(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, model_name,
            custom_engine_counts=None, plot_line=None, figures_path=consts.FIGURES_DIR,
            markers=None, alphas=None, figure_size=None, annotate_max_min=False):

    annotate_worst = False
    annotate_custom = False
    if markers == None:
        markers = ['X', '*', '^', '.']
    save_path = figures_path + '/{}/'.format(board_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += metric + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dpi_val = 100
    # Show the major grid and style it slightly.
    legend_cols = len(xpoints_grp)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    else:
        plt.figure(figsize=(3.5, 1))

    legend_y = 1.24
    legend_x = 1
    if custom_engine_counts != None:
        dpi_val = 200

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    plt.rcParams.update({'font.size': 8})

    plt.grid(which='major', color='#555555', linewidth=1)
    # Show the minor grid as well. Style it in very light gray as a thin,
    plt.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    plt.minorticks_on()
    ax = plt.gca()
    ax.set_axisbelow(True)

    for i in range(len(xpoints_grp)):
        if i == len(xpoints_grp) - 1 and custom_engine_counts != None:
            plt.scatter(xpoints_grp[i], ypoints_grp[i],
                        label=series_labels[i], marker=markers[i], zorder=1, linewidths=0.2, edgecolor='#990000', alpha=.5)
        else:
            plt.scatter(xpoints_grp[i], ypoints_grp[i],
                        label=series_labels[i], marker=markers[i], zorder=i + 2, alpha=alphas[i])
        min_x = min(xpoints_grp[i])
        max_x = max(xpoints_grp[i])
        min_y = min(ypoints_grp[i])
        max_y = max(ypoints_grp[i])
        min_x_index = xpoints_grp[i].index(min_x)
        max_x_index = xpoints_grp[i].index(max_x)
        min_y_index = ypoints_grp[i].index(min_y)
        max_y_index = ypoints_grp[i].index(max_y)

        min_x_index_num_engines = min_x_index + 2
        max_x_index_num_engines = max_x_index + 2
        min_y_index_num_engines = min_y_index + 2
        max_y_index_num_engines = max_y_index + 2

        if annotate_max_min and (i < len(xpoints_grp) - 1 or custom_engine_counts == None or annotate_custom):
            if series_labels[i].lower() == 'custom':
                min_x_index_num_engines = custom_engine_counts[min_x_index]
                max_x_index_num_engines = custom_engine_counts[max_x_index]
                min_y_index_num_engines = custom_engine_counts[min_y_index]
                max_y_index_num_engines = custom_engine_counts[max_y_index]

            # if max_x >= 0.9 * ax.get_xlim()[1]:
            #     max_x -= 0.06 * ax.get_xlim()[1]
            # if min_y >= 0.1 * ax.get_ylim()[1]:
            #     max_y -= 0.05 * ax.get_ylim()[1]
            y_inc = 0  # ( (len(xpoints_grp) // 2) - i ) / 3
            if plot_name.lower().startswith('thr'):
                ax.annotate(max_x_index_num_engines, (max_x - (max_x_index_num_engines >=
                            10), ypoints_grp[i][max_x_index] + y_inc), weight='bold', zorder=i + 3)
                if annotate_worst and min_x_index != max_x_index:
                    ax.annotate(min_x_index_num_engines, (min_x - (min_x_index_num_engines >=
                                10), ypoints_grp[i][min_x_index] + y_inc), weight='bold', zorder=i + 3)
            else:
                ax.annotate(min_x_index_num_engines, (min_x - (min_x_index_num_engines >=
                            10), ypoints_grp[i][min_x_index] + y_inc), weight='bold', zorder=i + 3)
                if annotate_worst and min_x_index != max_x_index:
                    ax.annotate(max_x_index_num_engines, (max_x - (max_x_index_num_engines >=
                                10), ypoints_grp[i][max_x_index] + y_inc), weight='bold', zorder=i + 3)

            if (min_y_index != min_x_index or not annotate_worst) and min_y_index != max_x_index:
                ax.annotate(min_y_index_num_engines, (xpoints_grp[i][min_y_index] - (min_y_index_num_engines + 2 >= 10),
                                                      min_y + y_inc), weight='bold', zorder=i + 3)
            if annotate_worst and max_y_index != min_x_index and max_y_index != max_x_index:
                ax.annotate(max_y_index_num_engines, (xpoints_grp[i][max_y_index] - (max_y_index_num_engines + 2 >= 10),
                                                      max_y + y_inc), weight='bold', zorder=i + 3)

    if plot_line is not None:
        print(save_path + '{}.png'.format(plot_name.lower()))
        line_label = list(plot_line.keys())[0]
        line_y = list(plot_line.values())[0]
        plt.axhline(y=line_y, label=line_label, linestyle='-', color='red')

    plt.xlabel(axis_labels[0], labelpad=0.2)
    plt.ylabel(axis_labels[1], labelpad=0)

    plt.legend(loc='upper right', bbox_to_anchor=(legend_x, legend_y),
               ncol=legend_cols, handletextpad=0.1, handlelength=0.8, columnspacing=0.4,
               frameon=False, borderpad=0)
    # plt.title(board_name + '_' + model_name)
    plt.savefig(save_path + '{}.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight', dpi=dpi_val)
    plt.savefig(save_path + '{}.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight', dpi=dpi_val)
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()

def scatter_with_breaks(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels,
                        plot_name, board_name, model_name,
                        figures_path=consts.FIGURES_DIR, markers=None,
                        alphas=None, figure_size=None,legend_specs = {'location': 'lower left',
                                                                      'ncol': 1}):

    if figure_size is None:
        figure_size = (3.5, 1)

    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', figsize=figure_size)#, gridspec_kw={'width_ratios': [2, 8]})
    fig.tight_layout()

    ax.grid(which='major', color='#555555', linewidth=1)
    ax.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax2.grid(which='major', color='#555555', linewidth=1)
    ax2.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax2.minorticks_on()
    ax2.set_axisbelow(True)


    plt.subplots_adjust(wspace=0.05)

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    if markers == None:
        markers = ['X', '*', '^', '.']

    x_break_point1, x_break_point2, range_min = plot_utils.decide_break_points_by_heursitic_results(
        xpoints_grp)
    
    print('>>>', x_break_point1, x_break_point2)

    for i in range(len(xpoints_grp)):
        # plot the same data on both axes
        ax.scatter(xpoints_grp[i], ypoints_grp[i],
                   label=series_labels[i], marker=markers[i], alpha=alphas[i])
        if i != len(xpoints_grp) - 1:
            ax2.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i], alpha=alphas[i])

    ax.set_xlim(range_min - (x_break_point1 - range_min) / 40, x_break_point1)
    ax2.set_xlim(x_break_point2, ax2.get_xlim()[1])

    ax.legend(loc=legend_specs['location'], ncol = legend_specs['ncol'], handlelength=0.8,
               columnspacing=0.4, borderpad=0.1, labelspacing = 0.1)
    
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, right=False)
    ax2.tick_params(which='minor', left=False)

    d = .1  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1, 1), (-d, +d), **kwargs)
    ax.plot((1, 1), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((0, 0), (1-d, 1+d), **kwargs)
    ax2.plot((0, 0), (-d, +d), **kwargs)

    ax.set_xlabel(axis_labels[0], labelpad=0.2)
    ax.set_ylabel(axis_labels[1], labelpad=0)

    save_path = mapping_general_utils.prepare_save_path_from_list(
        figures_path, [board_name, metric, model_name])

    plt.savefig(save_path + '{}_break.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}_break.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def scatter_with_breaks_xy(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels,
                           plot_name, board_name, model_name,
                           figures_path=consts.FIGURES_DIR, markers=None,
                           alphas=None, figure_size=None,):
    if figure_size is None:
        figure_size = (10, 10)

    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                               sharey='row', facecolor='w',
                                               figsize=figure_size,
                                               gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]})

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    if markers == None:
        markers = ['X', '*', '^', '.']

    x_break_point1, x_break_point2, range_min, range_max = plot_utils.decide_break_points(
        xpoints_grp, ypoints_grp, 0.35, 0.1)
    y_break_point1, y_break_point2, y_range_min, range_max_y = plot_utils.decide_break_points(
        ypoints_grp, xpoints_grp, 0.1, 0.35)
    print('>>>', x_break_point1, x_break_point2)
    print('>>>', y_break_point1, y_break_point2)

    for i in range(len(xpoints_grp)):
        # plot the same data on both axes
        ax1.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax2.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax3.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax4.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])

    margin = 0  # (x_break_point1 - range_min) / 10
    starting_point_left = range_min - margin
    ax1.set_xlim(starting_point_left, x_break_point1 + margin)
    ax3.set_xlim(starting_point_left, x_break_point1 + margin)

    margin = 0  # (range_max - x_break_point2) / 10
    ax2.set_xlim(x_break_point2 - margin, ax2.get_xlim()[1])
    ax4.set_xlim(x_break_point2 - margin, ax2.get_xlim()[1])

    margin = 0  # (y_break_point1 - y_range_min) / 10
    starting_point_bottom = y_range_min - margin
    ax3.set_ylim(starting_point_bottom, y_break_point1 + margin)
    ax4.set_ylim(starting_point_bottom, y_break_point1 + margin)

    margin = 0  # (range_max_y - y_break_point2) / 10
    ax1.set_ylim(y_break_point2 - margin, ax2.get_ylim()[1])
    ax2.set_ylim(y_break_point2 - margin, ax2.get_ylim()[1])

    # hide the spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, right=False)

    ax3.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(left=False, right=False)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax2.tick_params(bottom=False)

    ax3.spines['top'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax3.tick_params(top=False)
    ax4.tick_params(top=False)

    d = .03  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    ax4.plot((-d, +d), (-d, +d), **kwargs)

    plt.xlabel(axis_labels[0], labelpad=0.2)
    plt.ylabel(axis_labels[1], labelpad=0)

    save_path = mapping_general_utils.prepare_save_path_from_list(
        figures_path, [board_name, metric, model_name])

    plt.savefig(save_path + '{}_break_xy.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}_break_xy.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def plot_bars_as_interleaved_bar_groups(num_groups, plot_dict, colors=[], x_ticks_dict=None,
                                        rotation='horizontal', secondary_ticks_location=0.25,
                                        y_title=None,
                                        legend_y=1, legend_x=0, bar_linewidth=1, breakdown=False,
                                        horizontal_bars=False,
                                        seperate_baseline_label=None,
                                        make_space_between_axis_and_bars=False,
                                        draw_norm_line=False,
                                        percentage=False):
    x = []
    num_bars_in_group = len(plot_dict)
    is_seperate_baseline = 0
    if seperate_baseline_label is not None:
        is_seperate_baseline = 1
    legend_n_col = num_bars_in_group + is_seperate_baseline

    print(num_bars_in_group)
    if breakdown:
        num_bars_in_group = 1

    column_width = 1 / (num_bars_in_group + 1)
    if column_width == 0.5:
        column_width = 0.8

    if x_ticks_dict is not None:
        xticks_vals = list(x_ticks_dict.values())
        xticks_repititions = list(x_ticks_dict.keys())
        num_first_level_ticks = len(xticks_vals[0])
        ticks_0 = xticks_vals[0] * xticks_repititions[0]

    for j in range(num_bars_in_group):
        # adding space between sets of groups if there are two levels on x-axis
        x.append([])
        first_level_offsex = 0
        for i in range(num_groups):
            if horizontal_bars and i % num_first_level_ticks == 0:
                first_level_offsex += column_width
            x[j].append(i + column_width * j + first_level_offsex)

    bottom = np.zeros(num_groups)

    ylim = 0
    j = 0
    color_index = 0
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    align = 'edge'
    if breakdown:
        align = 'center'
    for series, values in plot_dict.items():
        current_color = default_colors[color_index]
        color_index += 1
        ylim = max(ylim, max(values))
        if horizontal_bars:
            plt.barh(x[j], values, column_width,
                     label=series, edgecolor='black', zorder=3, linewidth=bar_linewidth, left=bottom, align=align)
        else:
            plt.bar(x[j], values, column_width,
                    label=series, edgecolor='black', color=current_color, zorder=3, linewidth=bar_linewidth, bottom=bottom, align=align)
        if breakdown:
            bottom += values
        else:
            j += 1

    if make_space_between_axis_and_bars:
        if breakdown:
            plt.xlim([- 0.5, len(x[0]) - 0.5])
        else:
            plt.xlim([- 0.1, len(x[0]) - 0.2])

    if not horizontal_bars:
        if ylim > 2:
            plt.ylim([0, int(ylim + 0.5)])
            plt.yticks(np.arange(0, int(ylim + 0.5) + 0.1, 1))
        elif percentage:
            plt.ylim([0, ylim + 0.01])
            plt.yticks(np.arange(0, ylim + 0.01, ylim / 2))

        if percentage:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(ylim / 4))
        elif ylim >= 2:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
        else:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))

    if seperate_baseline_label is not None:
        plt.axhline(y=1, label=seperate_baseline_label, linestyle='-',
                    color=default_colors[color_index], linewidth=3)
    elif draw_norm_line:
        plt.axhline(y=1, linestyle='-', color='black')

    if len(plot_dict) > 1:
        if horizontal_bars:
            plt.legend(loc='upper left', handletextpad=0.1, bbox_to_anchor=(
                legend_x, legend_y), handlelength=0.8, frameon=False, borderpad=0, columnspacing=0.4, ncol=legend_n_col)
        else:
            plt.legend(loc='upper center', bbox_to_anchor=(legend_x, legend_y),
                       ncol=legend_n_col, handletextpad=0.1, handlelength=0.8, columnspacing=0.4, frameon=False, borderpad=0)

    if x_ticks_dict is not None:
        if horizontal_bars:
            plt.yticks(x[len(x)//2], ticks_0)
        else:
            plt.xticks(x[len(x)//2], ticks_0, rotation=rotation)

    if x_ticks_dict is not None and len(xticks_vals) > 1:
        ticks_1 = xticks_vals[1]
        if len(ticks_1) > 1:
            secondary_x = []
            step = num_groups / len(ticks_1)
            offset = step / 2
            for i in range(len(ticks_1)):
                secondary_x.append(offset)
                offset += step
                if horizontal_bars:
                    offset += column_width
                else:
                    if i > 0:
                        plt.axvline(x=i * step - column_width,
                                    color='black', linewidth=1)

            if horizontal_bars:
                sec = plt.gca().secondary_yaxis(secondary_ticks_location)
                sec.set_yticks(secondary_x, ticks_1, rotation='vertical')
            else:
                sec = plt.gca().secondary_xaxis(secondary_ticks_location)
                sec.set_xticks(secondary_x, ticks_1)


def separate_bars(xpoints_grp, plot_dict, colors=[]):
    x = np.arange(len(xpoints_grp[0]))  # the label locations
    bar_width = 0.8
    width = len(x)  # the width of the bars
    multiplier = 0

    xticks_x = []
    xticks_y = []
    seperator = 2

    for i in range(len(xpoints_grp)):
        for j in range(0, len(x), 2):
            xticks_x.append(i * (len(x) + seperator) + j)
            xticks_y.append(2 + j)

    for series, values in plot_dict.items():
        offset = (width + seperator) * multiplier
        if len(colors) > 1:
            rects = plt.bar(x + offset, values, bar_width,
                            label=series, color=colors, edgecolor='black')
        else:
            rects = plt.bar(x + offset, values, bar_width,
                            label=series, edgecolor='black')

        # plt.broken_barh([(0, 15)], (2, 4))
        multiplier += 1

    plt.xticks(xticks_x, xticks_y)


def prepare_plot_dict(xpoints_grp, ypoints_grp, series_labels, normalize):
    plot_dict = {}
    for i in range(len(xpoints_grp)):
        if normalize:
            normalization_base = ypoints_grp[i][0]
            for j in range(len(ypoints_grp[i])):
                ypoints_grp[i][j] /= normalization_base
        plot_dict[series_labels[i]] = ypoints_grp[i]

    return plot_dict

# x_ticks_dict: for hierarical or multiple ticks. The format: {num_bar_groups_per_tick: [ticks]}


def plot_bar_groups(data_dict, plot_file_name,
                    x_title=None, y_title=None, x_ticks_dict=None, relative_save_path=None, interleaved_bars=True,
                    breakdown=False, horizontal_bars=False, percentage=False,
                    seperate_baseline_label=None, abs_save_path=consts.FIGURES_DATA_DIR):

    save_path = mapping_general_utils.prepare_save_path(
        abs_save_path, relative_save_path)
    num_groups = len(list(data_dict.values())[0])

    rotation = 'horizontal'
    secondary_ticks_location = -0.25
    dpi_val = 100
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['xtick.major.pad'] = 0
    plt.rcParams['ytick.major.pad'] = 0
    legend_x = 0
    legend_y = 0
    draw_norm_line = True
    make_space_between_axis_and_bars = False
    bar_linewidth = 0.1
    if horizontal_bars:
        secondary_ticks_location = -0.42
        plt.figure(figsize=(3.2, 0.5))
        bar_linewidth = 0.5
        legend_x = 0.2
        legend_y = 1.6
    elif num_groups == 2:
        secondary_ticks_location = -0.5
        plt.figure(figsize=(3.45, 1))
        legend_x = 0.5
        legend_y = 1.25
    elif num_groups == 8:
        rotation = 20
        secondary_ticks_location = -0.5
        plt.figure(figsize=(3.45, 0.8))
        bar_linewidth = 0.1
        legend_x = 0.5
        legend_y = 1.15
    elif num_groups == 20:
        rotation = 'vertical'
        secondary_ticks_location = -0.6
        plt.figure(figsize=(6.45, 1.2))
        legend_x = 0.5
        legend_y = 1.17
    elif num_groups == 11:
        draw_norm_line = False
        secondary_ticks_location = -0.6
        plt.figure(figsize=(3.45, 1))
        legend_x = 0.5
        legend_y = 1
    elif num_groups == 40:
        secondary_ticks_location = -0.4
        plt.figure(figsize=(8, 0.6))
        bar_linewidth = 0.5
        legend_x = 0.5
        legend_y = 1.4
    elif num_groups == 27:
        rotation = 'vertical'
        draw_norm_line = False
        secondary_ticks_location = -0.4
        plt.figure(figsize=(3.45, 0.6))
        bar_linewidth = 0.5
        legend_x = 0.5
        legend_y = 1.45
    elif num_groups == 7:
        draw_norm_line = False
        secondary_ticks_location = -0.4
        plt.figure(figsize=(3.45, 0.6))
        bar_linewidth = 0.5
        legend_x = 0.5
        legend_y = 1.45
    elif num_groups == 50:
        rotation = 'vertical'
        draw_norm_line = False
        secondary_ticks_location = -0.6
        plt.figure(figsize=(10, 0.5))
        bar_linewidth = 0.5
        legend_x = 0.5
        legend_y = 1.55
    if num_groups == 2 and breakdown and not horizontal_bars:
        plt.figure(figsize=(1.6, 1))
        legend_x = 0.5
        legend_y = 1
        legend_x = 0.3
        legend_y = 1.3
        make_space_between_axis_and_bars = True
    if num_groups == 4 and len(data_dict) == 2 and not horizontal_bars:
        plt.figure(figsize=(1.6, 1))
        legend_x = 0.4
        legend_y = 1.29
        make_space_between_axis_and_bars = True

    plt.margins(0)
    # plt.gca().tick_params(axis='y', which='minor', length=2)

    # plt.minorticks_on()
    if horizontal_bars:
        plt.grid(axis='x', color='#555555', which='major')
        plt.grid(axis='x', color='#555555', which='minor',
                 linewidth=0.5, linestyle='dashed')
    else:
        plt.grid(axis='y', color='#555555', which='major')
        plt.grid(axis='y', color='#555555', which='minor',
                 linewidth=0.5, linestyle='dashed')
    plt.ylabel(y_title, labelpad=0.4)
    plt.xlabel(x_title, labelpad=0.4)
    plt.autoscale(enable=True, axis='x', tight=True)

    if percentage:
        if horizontal_bars:
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        else:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    if interleaved_bars:
        plot_bars_as_interleaved_bar_groups(num_groups=num_groups, plot_dict=data_dict, x_ticks_dict=x_ticks_dict,
                                            rotation=rotation, secondary_ticks_location=secondary_ticks_location,
                                            y_title=y_title,
                                            legend_y=legend_y, legend_x=legend_x, bar_linewidth=bar_linewidth,
                                            breakdown=breakdown, horizontal_bars=horizontal_bars,
                                            seperate_baseline_label=seperate_baseline_label,
                                            make_space_between_axis_and_bars=make_space_between_axis_and_bars, draw_norm_line=draw_norm_line,
                                            percentage=percentage)
    if horizontal_bars:
        plt.gca().invert_yaxis()

    plt.savefig(save_path + '{}.png'.format(plot_file_name), format='png',
                bbox_inches='tight', dpi=dpi_val)
    plt.savefig(save_path + '{}.pdf'.format(plot_file_name), format='pdf',
                bbox_inches='tight', dpi=dpi_val)
    plt.clf()
    plt.close()


def bar_mapping_groups(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, model_name,
                       y_axis_unit, plot_mode='sep', normalized=True, normalize=False):

    save_path = os.getcwd() + '/figures/{}/'.format(board_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += metric + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plt.figure(figsize=(4, 2))

    plot_dict = prepare_plot_dict(
        xpoints_grp, ypoints_grp, series_labels, normalized, normalize)
    if plot_mode == 'sep':
        separate_bars(xpoints_grp, plot_dict)
    elif plot_mode == 'inter':
        plot_bars_as_interleaved_bar_groups(len(xpoints_grp[0]), plot_dict)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel(axis_labels[0])
    y_axis_label_postfix = ' normalized' if normalized else '(' + \
        y_axis_unit + ')'
    plt.ylabel(axis_labels[1] + y_axis_label_postfix)
    plt.title(board_name + '_' + model_name)
    # ax.set_xticks(x + width, xpoints_grp[0])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, borderpad=0, columnspacing=0.4)
    plot_name_postfix = 'normalized' if normalized else y_axis_unit
    plt.savefig(save_path + '{}_{}.png'.format(plot_name,
                plot_name_postfix), bbox_inches='tight')
    plt.clf()
    plt.close()


def bar_model_groups(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, mapping_name,
                     y_axis_unit='', plot_mode='sep', normalized=True, colors=[]):

    save_path = mapping_general_utils.prepare_save_path_from_list(
        os.getcwd() + '/figures', [board_name, metric, mapping_name])

    plt.figure(figsize=(4, 2))

    plot_dict = prepare_plot_dict(
        xpoints_grp, ypoints_grp, series_labels, normalized)
    if plot_mode == 'sep':
        separate_bars(xpoints_grp, plot_dict, colors)
    elif plot_mode == 'inter':
        plot_bars_as_interleaved_bar_groups(
            len(xpoints_grp[0]), plot_dict, colors)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel(axis_labels[0])
    y_axis_label_postfix = ''
    if normalized or y_axis_unit != '':
        y_axis_label_postfix = ' normalized' if normalized else '(' + \
            y_axis_unit + ')'
    plt.ylabel(axis_labels[1] + y_axis_label_postfix)
    plt.title(board_name + '_' + mapping_name)
    # ax.set_xticks(x + width, xpoints_grp[0])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, borderpad=0, columnspacing=0.4)
    plot_name_postfix = 'normalized' if normalized else y_axis_unit
    if plot_name_postfix != '':
        plot_name = '{}_{}.png'.format(plot_name, plot_name_postfix)
    else:
        plot_name = plot_name + '.png'
    plt.savefig(save_path + plot_name, bbox_inches='tight')
    plt.clf()
    plt.close()
