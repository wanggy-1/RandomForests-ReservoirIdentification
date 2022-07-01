import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys
from collections import Counter


def class_distribution(data=None, classname=None, titlename='Class Distribution', bar_width=0.5, make_plot=True):
    """
    Plot class distribution.
    :param data: (numpy.1darray) - Data for visualization.
    :param classname: (String) - Class names. Must match the class codes in ascending order.
                      Default is None, which is to use the class codes as class names.
    :param titlename: (String) - The title name of this figure. Default is 'Class Distribution'.
    :param bar_width: (Float) - Control the bar width, range from 0 to 1. Default is 0.5.
    :param make_plot: (Bool) - Whether to make an independent plot. Default is True.
    """
    # Sort class codes in ascending order.
    x = list(dict(sorted(Counter(data).items())).keys())
    num = list(dict(sorted(Counter(data).items())).values())
    # Plot class distribution.
    if make_plot:
        plt.figure(figsize=(8, 6))
    plt.ylabel('Amount', fontsize=20)
    if titlename is not None:
        plt.title(titlename, fontsize=24)
    if classname is not None:  # If there are class names, will be used as x ticks.
        plt.xticks(x, classname)
    plt.tick_params(labelsize=18)
    plt.bar(x, num, width=bar_width, edgecolor='k')  # Plot bars to show sample amounts of all classes.
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--')
    plt.xlim([-0.5, 1.5])
    for i in range(len(x)):  # Add annotation text of the sample amount at the top of the bar.
        plt.text(x[i]-0.08, num[i]+2, str(num[i]), fontsize=18)


def plot_log(df_log=None, depth_name=None, log_name=None, xlim=None, ylim=None, xlabel='Value', ylabel='Depth(m)',
             title=None, fill_log=False, cmap='rainbow', colors=None, categorical=False):
    """
    Plot logging data.
    :param df_log: (pandas.DataFrame) - The well log data frame which contains a depth column and log columns.
    :param depth_name: (String) - The depth column name.
    :param log_name: (String) - The log column name.
    :param xlim: (List of floats) - Range of x-axis, e.g. [0, 150]. Default is to infer from data.
                                    For numerical logs only.
    :param ylim: (List of floats) - Range of y-axis, e.g. [0, 150]. Default is to infer from data.
    :param xlabel: (String) - The label of x-axis. Default is 'Value'. For numerical logs only.
    :param ylabel: (String) - The label of y-axis. Default is 'Depth(m)'.
    :param title: (String) - The title of this plot.
    :param fill_log: (Bool) - Whether to fill the area between y-axis and the log curve. Default is True.
                              For numerical logs only.
    :param cmap: (String) - The color map to fill the area between y-axis and the log curve. Default is 'rainbow'.
                            For numerical logs only.
    :param colors: (List of Strings) - Colors for different logging data classes. For categorical logs only.
                   For example, ['red', 'green', 'blue'] for classes ['sandstone', 'shale', 'limestone'].
    :param categorical: (Bool) - Whether the logging data is categorical. Default is False.
    """
    # Get depth and logging data.
    depth = df_log[depth_name].values
    log = df_log[log_name].values
    # Categorical logs.
    if categorical:
        plt.figure(figsize=[2, 8])
        plt.xticks([])
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tick_params(axis='y', labelsize=13)
        plt.xlim([0, 1])
        if ylim is None:
            plt.ylim(depth.min(), depth.max())
        else:
            plt.ylim(ylim[0], ylim[1])
        ax = plt.gca()
        ax.invert_yaxis()
        classlabel = list(dict(sorted(Counter(log).items())).keys())
        for i in range(len(classlabel)):
            plt.fill_betweenx(depth, 0, (log+1) * 10, where=(log == classlabel[i]), color=colors[i], interpolate=True)
    # Numerical logs.
    else:
        plt.style.use('bmh')  # Plot style.
        plt.figure(figsize=(7, 10))
        ax = plt.axes()
        plt.plot(log, depth, c='black', lw=0.5)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is None:
            ax.set_ylim(depth.min(), depth.max())
        else:
            ax.set_ylim(ylim[0], ylim[1])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_title(title, fontsize=16, fontweight='bold')
        if fill_log:
            # Get x axis range.
            left_value, right_value = ax.get_xlim()
            span = abs(left_value - right_value)
            # Assign color map.
            cmap = plt.get_cmap(cmap)
            # Create array of values to divide up the area under curve.
            color_index = np.arange(left_value, right_value, span / 100)
            # Loop through each value in the color_index.
            for index in sorted(color_index):
                index_value = (index - left_value) / span
                color = cmap(index_value)  # Obtain color for color index value.
                plt.fill_betweenx(depth, 0, log, where=log >= index, color=color)
    plt.tight_layout()


def cross_plot2D(df=None, x=None, y=None, c=None, cmap='rainbow',
                 xlabel=None, ylabel=None, title=None, colorbar=None,
                 xlim=None, ylim=None, show=True):
    """
    Make 2D cross-plot.
    https://towardsdatascience.com/scatterplot-creation-and-visualisation-with-matplotlib-in-python-7bca2a4fa7cf
    :param df: (pandas.DataFrame) - Well log data frame.
    :param x: (String) - Log name in data frame, which will be the x-axis of the cross-plot.
    :param y: (String) - Log name in data frame, which will be the y-axis of the cross-plot.
    :param c: (String) - Log name in data frame, which will be the color of the scatters in cross-plot.
    :param cmap: (String) - Default is 'rainbow', the color map of scatters.
    :param xlabel: (String) - X-axis name.
    :param ylabel: (String) - Y-axis name.
    :param title: (String) - Title of the figure.
    :param colorbar: (String) - Name of the color-bar.
    :param xlim: (List of floats) - Default is to infer from data. Range of x-axis, e.g. [0, 150]
    :param ylim: (List of floats) - Default is to infer from data. Range of y-axis, e.g. [0, 150]
    :param show: (Bool) - Default is True. Whether to show the figure.
    """
    plt.figure(figsize=(13, 9))
    # Set style sheet to bmh.
    plt.style.use('bmh')
    # Set up the scatter plot.
    plt.scatter(x=x, y=y, data=df, c=c, cmap=cmap)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(labelsize=16)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontsize=20)
    cbar = plt.colorbar()
    cbar.set_label(colorbar, size=18)
    cbar.ax.tick_params(labelsize=16)
    if show:
        plt.show()


def visualize_horizon(df=None, x_name=None, y_name=None, value_name=None, deltax=25.0, deltay=25.0, cmap='seismic_r',
                      vmin=None, vmax=None, nominal=False, class_code=None, class_label=None, fig_name=None,
                      cbar_label=None, axe_aspect='equal', show=True):
    """
    Visualize horizon data.
    :param df: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 'value1', 'value2', '...'] columns.
    :param x_name: (String) - x-coordinate column name.
    :param y_name: (String) - y-coordinate column name.
    :param value_name: (String) - Column name of the values over which the plot is drawn.
    :param deltax: (Float) - Default is 25.0. x-coordinate interval of the regular grid.
    :param deltay: (Float) - Default is 25.0. y-coordinate interval of the regular grid.
    :param cmap: (String) - Default is 'seismic_r'. The color map.
    :param vmin: (Float) - Minimum value of the continuous color bar.
    :param vmax: (Float) - Maximum value of the continuous color bar.
    :param nominal: (Bool) - Default is False. Whether the data is nominal (discrete).
    :param class_code: (List of integers) - The class codes. (e.g. [0, 1, 2])
    :param class_label: (List of strings) - The class labels. (e.g. ['Mudstone', 'Shale', 'Sandstone']).
    :param fig_name: (String) - Default is 'Result'. Figure name.
    :param cbar_label: (String) - The color bar label. Default is to use value_name as color bar label.
    :param axe_aspect: (String) - Default is 'equal'. The aspect ratio of the axes.
                       'equal': ensures an aspect ratio of 1. Pixels will be square (unless pixel sizes are explicitly
                       made non-square in data coordinates using extent).
                       'auto': The axes is kept fixed and the aspect is adjusted so that the data fit in the axes. In
                       general, this will result in non-square pixels.
    :param show: (Bool) - Default is True. Whether to show plot.
    """
    # Construct regular grid to show horizon.
    x = df[x_name].values
    xmin, xmax = np.amin(x), np.amax(x)
    y = df[y_name].values
    ymin, ymax = np.amin(y), np.amax(y)
    xnew = np.linspace(xmin, xmax, int((xmax - xmin) / deltax) + 1)
    ynew = np.linspace(ymin, ymax, int((ymax - ymin) / deltay) + 1)
    xgrid, ygrid = np.meshgrid(xnew, ynew)
    xgrid = xgrid.ravel(order='F')
    ygrid = ygrid.ravel(order='F')
    df_grid = pd.DataFrame({x_name: xgrid, y_name: ygrid})
    # Merge irregular horizon on regular grid.
    df_horizon = pd.merge(left=df_grid, right=df, how='left', on=[x_name, y_name])
    data = df_horizon[value_name].values
    x = df_horizon[x_name].values
    y = df_horizon[y_name].values
    # Plot horizon.
    plt.figure(figsize=(12, 6))
    if fig_name is None:
        plt.title('Result', fontsize=20)
    else:
        # plt.title(fig_name, fontsize=22)
        pass
    plt.ticklabel_format(style='plain')
    extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]
    plt.imshow(data.reshape([len(ynew), len(xnew)], order='F'),
               origin='lower', aspect=axe_aspect, vmin=vmin, vmax=vmax,
               cmap=plt.cm.get_cmap(cmap), extent=extent)
    plt.xlabel(x_name, fontsize=22)
    plt.ylabel(y_name, fontsize=22)
    plt.tick_params(labelsize=20)
    if cbar_label is None:
        cbar_label = value_name
    if nominal:
        class_code.sort()
        tick_min = (max(class_code) - min(class_code)) / (2 * len(class_code)) + min(class_code)
        tick_max = max(class_code) - (max(class_code) - min(class_code)) / (2 * len(class_code))
        tick_step = (max(class_code) - min(class_code)) / len(class_code)
        ticks = np.arange(start=tick_min, stop=tick_max + tick_step, step=tick_step)
        cbar = plt.colorbar(ticks=ticks, fraction=0.05)
        cbar.set_label(cbar_label, fontsize=22)
        cbar.ax.set_yticklabels(class_label)
        cbar.ax.tick_params(axis='y', labelsize=20)
    else:
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(axis='y', labelsize=20)
        cbar.set_label(cbar_label, fontsize=22)
    if show:
        plt.show()


def horizon_log(df_horizon=None, df_well_coord=None, log_file_path=None, sep='\t', well_name_col=None,
                horizon_x_col=None, horizon_y_col=None, horizon_z_col=None,
                log_x_col=None, log_y_col=None, log_z_col=None, log_value_col=None, log_abnormal_value=None,
                log_file_suffix='.txt', print_progress=False, w_x=25.0, w_y=25.0, w_z=2.0):
    """
    Mark log values on horizon. Only for vertical well now.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 'z'] columns.
    :param df_well_coord: (pandas.DataFrame) - Default is None, which means not to use a well coordinates data frame and
                          the log files must contain well x and y columns. If not None, then this is a well coordinates
                          data frame which contains ['well_name', 'well_x', 'well_y'] columns.
    :param log_file_path: (String) - Time domain well log file directory.
    :param sep: (String) - Default is '\t'. Column delimiter in log files.
    :param well_name_col: (String) - Well name column name in df_well_coord.
    :param horizon_x_col: (String) - Horizon x-coordinate column name in df_horizon.
    :param horizon_y_col: (String) - Horizon y-coordinate column name in df_horizon.
    :param horizon_z_col: (String) - Horizon z-coordinate column name in df_horizon.
    :param log_x_col: (String) - Well x-coordinate column name.
    :param log_y_col: (String) - Well y-coordinate column name.
    :param log_z_col: (String) - Well log two-way time column name.
    :param log_value_col: (String) - Well log value column name.
    :param log_abnormal_value: (Float) - Abnormal value in log value column.
    :param log_file_suffix: (String) - Default is '.txt'. Well log file suffix.
    :param print_progress: (Bool) - Default is False. Whether to print progress.
    :param w_x:  (Float) - Default is 25.0.
                 Size of x window in which the well xy-coordinates and horizon xy-coordinates will be matched.
    :param w_y: (Float) - Default is 25.0.
                Size of y window in which the well xy-coordinates and horizon xy-coordinates will be matched.
    :param w_z: (Float) - Default is 2.0.
                Size of z window in which the well z-coordinates and horizon z-coordinates will be matched.
    :return: df_out: (pandas.DataFrame) - Output data frame which contains ['x', 'y', 'z', 'log value', 'well name']
                     columns.
    """
    # Initialize output data frame as a copy of horizon data frame.
    df_out = df_horizon.copy()
    # List of well log files.
    log_file_list = os.listdir(log_file_path)
    # Mark log values on horizon.
    for log_file in log_file_list:
        # Load well log data.
        df_log = pd.read_csv(os.path.join(log_file_path, log_file), delimiter=sep)
        if log_abnormal_value is not None:
            drop_ind = [x for x in range(len(df_log)) if df_log.loc[x, log_value_col] == log_abnormal_value]
            df_log.drop(index=drop_ind, inplace=True)
            df_log.reset_index(drop=True, inplace=True)
        # Get well name.
        well_name = log_file[:-len(log_file_suffix)]
        # Print progress.
        if print_progress:
            sys.stdout.write('\rMatching well %s' % well_name)
        if df_well_coord is not None:
            # Get well coordinates from well coordinate file.
            [well_x, well_y] = \
                np.squeeze(df_well_coord[df_well_coord[well_name_col] == well_name][[log_x_col, log_y_col]].values)
        else:
            well_x, well_y = df_log.loc[0, log_x_col], df_log.loc[0, log_y_col]
        # Get horizon coordinates.
        horizon_x = df_horizon[horizon_x_col].values
        horizon_y = df_horizon[horizon_y_col].values
        # Compute distance map between well and horizon.
        xy_dist = np.sqrt((horizon_x - well_x) ** 2 + (horizon_y - well_y) ** 2)
        # Get array index of minimum distance in distance map. This is the horizon coordinate closest to the well.
        idx_xy = np.argmin(xy_dist)
        if xy_dist[idx_xy] < math.sqrt(w_x ** 2 + w_y ** 2):
            # Get horizon two-way time at the closest point to the well.
            horizon_t = df_horizon.loc[idx_xy, horizon_z_col]
            # Get well log two-way time.
            log_t = df_log[log_z_col].values
            # Compute distances between well log two-way time and horizon two-way time at the closest point to the well.
            t_dist = np.abs(log_t - horizon_t)
            # Get array index of the minimum distance. This the vertically closest point of the well log to the horizon.
            idx_t = np.argmin(t_dist)
            if t_dist[idx_t] < w_z:
                # Get log value.
                log_value = df_log.loc[idx_t, log_value_col]
                # Mark log value on horizon.
                df_out.loc[idx_xy, log_value_col] = log_value
                # Mark well name.
                df_out.loc[idx_xy, 'WellName'] = well_name
    # Drop NaN.
    df_out.dropna(axis='index', how='any', subset=[log_value_col], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def plot_markers(df=None, x_col=None, y_col=None, class_col=None, wellname_col=None,
                 class_code=None, class_label=None, colors=None,
                 annotate=True, anno_color='k', anno_fontsize=12, anno_shift=None, show=False):
    """
    Plot markers on horizon. For nominal markers only.
    :param df: (pandas.Dataframe) - Marker data frame which contains ['x', 'y', 'class', 'well name'] columns.
    :param x_col: (String) - x-coordinate column name.
    :param y_col: (String) - y-coordinate column name.
    :param class_col: (String) - Class column name.
    :param wellname_col: (String) - Well name column name.
    :param class_code: (List of integers) - Class codes of the markers.
    :param class_label: (List of strings) - Label names of the markers.
    :param colors: (List of strings) - Color names of the markers.
    :param annotate: (Bool) - Default is True. Whether to annotate well names beside well markers.
    :param anno_color: (String) - Annotation text color.
    :param anno_fontsize: (Integer) - Default is 12. Annotation text font size.
    :param anno_shift: (List of floats) - Default is [0, 0]. Annotation text coordinates.
    :param show: (Bool) - Default is False. Whether to show markers.
    """
    # Get class codes.
    marker_class = df[class_col].copy()
    marker_class.drop_duplicates(inplace=True)
    marker_class = marker_class.values
    marker_class = marker_class.astype('int')
    marker_class.sort()
    # Plot markers.
    for i in range(len(marker_class)):
        idx = [x for x in range(len(df)) if df.loc[x, class_col] == marker_class[i]]
        sub_frame = df.loc[idx, [x_col, y_col]]
        x, y = sub_frame.values[:, 0], sub_frame.values[:, 1]
        idx = np.squeeze(np.argwhere(class_code == marker_class[i]))
        plt.scatter(x, y, c=colors[idx], edgecolors='k', label=class_label[idx],
                    s=100)
    if annotate:
        # Annotate well names.
        for i in range(len(df)):
            well_name = df.loc[i, wellname_col]
            x_anno = df.loc[i, x_col]
            y_anno = df.loc[i, y_col]
            if anno_shift is None:
                anno_shift = [0, 0]
            plt.annotate(text=well_name, xy=(x_anno, y_anno), xytext=(x_anno - anno_shift[0], y_anno + anno_shift[1]),
                         color=anno_color, fontsize=anno_fontsize)
    plt.legend(loc='upper right', fontsize=20)
    if show:
        plt.show()
