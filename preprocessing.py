from visualization import class_distribution
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from collections import Counter
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import math
import scipy.interpolate
import scipy.spatial
import segyio


def resample_log(df_log=None, delta=None, depth_col='depth', log_col=None, method='average', abnormal_value=None,
                 nominal=False, delete_nan=True, fill_nan=None):
    """
    Re-sample well log by a certain depth interval (Small sampling interval to large sampling interval).
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['depth', 'log'] columns.
    :param delta: (Float) - Depth interval.
    :param depth_col: (String) - Default is 'depth'. Depth column name.
    :param log_col: (String or list of strings) - Well log column name(s). E.g. 'gamma' or ['Vp', 'porosity'].
    :param method: (String) - Default is 'average'. Re-sampling method.
                   'nearest' - Take the nearest log value.
                   'average' - Take the mean of log values in a depth window (length = delta)
                               centered at the re-sampled depth point.
                   'median' - Take the median of log values in a depth window (length = delta)
                              centered at the re-sampled depth point.
                   'rms' - Take the root-mean-square of log values in a depth window (length = delta)
                           centered at the re-sampled depth point.
                   'most_frequent' - Take the most frequent log values in a depth window (length = delta)
                                     centered at the re-sampled depth point.
    :param abnormal_value: (Float) - Abnormal value in log column. If abnormal value is defined, will remove the whole
                                     row with abnormal value. If None, means no abnormal value in log column.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal (e.g. [0, 1, 2, 0, 2]). If True, the log
                    value will be integer, else the log value will be float.
    :param delete_nan: (Bool) - Default is True. Whether to delete rows with NaN value.
    :param fill_nan: (Integer or float) - Default is not to fill NaN. Fill NaN with this value.
    :return: df_out: (pandas.Dataframe) - Re-sampled well log data frame.
    """
    df_log_copy = df_log.copy()
    if abnormal_value is not None:
        df_log_copy.replace(abnormal_value, np.nan)
        df_log_copy.dropna(axis='index', how='any', inplace=True)
        df_log_copy.reset_index(drop=True, inplace=True)
    depth = df_log_copy[depth_col].values  # Original depth array.
    log = df_log_copy[log_col].values  # Original log array.
    new_depth = np.arange(start=math.ceil(np.amin(depth) // delta * delta),
                          stop=np.amax(depth) // delta * delta + delta * 2,
                          step=delta)  # New depth array.
    if log.ndim > 1:
        new_log = np.full([len(new_depth), log.shape[1]], fill_value=np.nan)  # Initiate new log array (fill with nan).
    else:
        new_log = np.full(len(new_depth), fill_value=np.nan)
    for i in range(len(new_depth)):
        # Choose the depth and log values that meet the condition.
        condition = (depth > new_depth[i] - delta / 2) & (depth <= new_depth[i] + delta / 2)
        index = np.argwhere(condition)  # Find index in the window.
        temp_log = log[index]  # Log in the window.
        temp_depth = depth[index]  # Depth in the window.
        if len(temp_log):  # If there are log values in the window.
            # Re-sample log by different methods.
            if method == 'nearest' and len(temp_log):  # Nearest neighbor.
                ind_nn = np.argmin(np.abs(temp_depth - new_depth[i]))  # The nearest neighbor index.
                new_log[i] = temp_log[ind_nn]
            if method == 'average' and len(temp_log):  # Take average value.
                new_log[i] = np.average(temp_log, axis=0)
            if method == 'median' and len(temp_log):  # Take median value.
                new_log[i] = np.median(temp_log, axis=0)
            if method == 'rms' and len(temp_log):  # Root-mean-square value.
                new_log[i] = np.sqrt(np.mean(temp_log ** 2, axis=0))
            if method == 'most_frequent' and len(temp_log):  # Choose the most frequent log value (nominal log only).
                if temp_log.ndim > 1 and temp_log.shape[1] > 1:
                    for j in range(temp_log.shape[1]):
                        values, counts = np.unique(temp_log[:, j], return_counts=True)
                        ind_mf = np.argmax(counts)
                        new_log[i, j] = values[ind_mf]
                else:
                    values, counts = np.unique(temp_log, return_counts=True)
                    ind_mf = np.argmax(counts)
                    new_log[i] = values[ind_mf]
    # Output result to new data-frame.
    df_out = pd.DataFrame(data=np.c_[new_depth, new_log], columns=df_log_copy.columns)
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_out.columns)
    sub_col.remove(depth_col)
    df_out.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    if fill_nan:
        # Fill NaN.
        df_out.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with any NaN.
        df_out.dropna(axis='index', how='any', inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def log_interp(df=None, step=0.125, log_col=None, depth_col='Depth',
               top_col=None, bottom_col=None, nominal=True, mode='segmented', method='slinear',
               delete_nan=False, fill_nan=None):
    """
    Interpolate well log between top depth and bottom depth.
    :param df: (pandas.DataFrame) - Well log data frame which contains ['top depth', 'bottom depth', 'log'] column.
    :param step: (Float) - Default is 0.125m. Interpolation interval.
    :param log_col: (String or list of strings) - Column name(s) of the well log.
    :param depth_col: (String) - Default is 'Depth'. Column name of measured depth.
    :param top_col: (String) - Column name of the top boundary depth.
    :param bottom_col: (String) - Column name of the bottom boundary depth.
    :param nominal: (Bool) - Default is True. Whether the log value is nominal.
    :param mode: (String) - Default is 'segmented'. When is 'segmented', interpolate segmented well log with top depth
                 and bottom depth column. When is 'continuous', interpolate continuous well log with depth column.
    :param method: (String) - Default is 'slinear'. Interpolation method when processing continuous well log values.
                   Optional methods are: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                   'previous', and 'next'. 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
                   zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value
                   of the point; 'nearest-up' and 'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5) in
                   that 'nearest-up' rounds up and 'nearest' rounds down.
    :param delete_nan: (Bool) - Default is True. Whether to delete rows with NaN.
    :param fill_nan: (Integer or float) - Default is not to fill NaN. Fil NaN with this value.
    :return df_out: (pandas.DataFrame) - Interpolated well log data frame.
    """
    if mode == 'segmented':
        depth_min = df[top_col].min()  # Get minimum depth.
        depth_min = math.ceil(depth_min)
        depth_max = df[bottom_col].max()  # Get maximum depth.
        depth_array = np.arange(depth_min, depth_max + step, step, dtype=np.float32)  # Create depth column.
        # Initiate well log column with NaN.
        if isinstance(log_col, str):
            log_array = np.full(len(depth_array), np.nan)
        elif isinstance(log_col, list) and len(log_col) > 1:
            log_array = np.full([len(depth_array), len(log_col)], np.nan)
        else:
            raise ValueError('Log column name must either be string type for 1 column or list type for 2 or more '
                             'columns.')
        # Create new data frame.
        df_out = pd.DataFrame({depth_col: depth_array})
        df_out[log_col] = log_array
        # Assign log values to new data frame.
        for i in range(len(df)):
            idx = (df_out.loc[:, depth_col] <= df.loc[i, bottom_col]) & \
                    (df_out.loc[:, depth_col] >= df.loc[i, top_col])
            df_out.loc[idx, log_col] = df.loc[i, log_col]
    elif mode == 'continuous':
        log_depth = df[depth_col].values  # Get well log depth array.
        log_value = df[log_col].values  # Get well log value array.
        depth_min = np.amin(log_depth)  # Get minimum well log value.
        depth_max = np.amax(log_depth)  # Get maximum well log value.
        log_depth[log_depth == depth_min] = round(depth_min)
        new_depth = np.arange(start=depth_min, stop=depth_max, step=step)  # New depth array.
        f = scipy.interpolate.interp1d(log_depth, log_value, axis=0, kind=method)  # Interpolator.
        new_value = f(new_depth)  # Interpolated log value.
        data = np.c_[new_depth, new_value]
        df_out = pd.DataFrame(data, columns=df.columns)
    else:
        raise ValueError("Mode can only be 'segmented' or 'continuous'.")
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_out.columns)
    sub_col.remove(depth_col)
    df_out.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    if fill_nan is not None:
        # Fill NaN.
        df_out.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with NaN.
        df_out.dropna(axis='index', how='any', inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def time_log(df_dt=None, df_log=None, log_depth_col='Depth', dt_depth_col='Depth',
             time_col='TWT', log_col=None, fill_nan=None, delete_nan=False, nominal=False):
    """
    Match well logs with a detailed time-depth relation.
    :param df_dt: (pandas.DataFrame) - Depth-time relation data frame which contains ['Depth', 'Time'] columns.
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['Depth', 'log'] columns.
    :param log_depth_col: (String) - Default is 'Depth'. Column name of depth in well log file.
    :param dt_depth_col: (String) - Default is 'Depth'. Column name of depth in depth-time relation file.
    :param time_col: (String) - Default is 'TWT'. Column name of two-way time.
    :param log_col: (String or list of strings) - Column name(s) of well log.
    :param fill_nan: (Float or integer) - Default is not to fill NaN. Fill NaN with this number.
    :param delete_nan: (Bool) - Default is False. Whether to delete rows with NaN.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal.
    :return: df_log_time: (pandas.DataFrame) - Time domain well log data frame.
    """
    df_log_time = df_log.copy()
    if log_col == 'infer':
        # Infer log columns.
        log_col = list(df_log_time.columns).remove(log_depth_col)
    # Remove rows in well log whose depth is out of range of depth-time relation.
    ind = [i for i in range(len(df_log_time)) if df_log_time.loc[i, log_depth_col] < df_dt[dt_depth_col].min() or
           df_log_time.loc[i, log_depth_col] > df_dt[dt_depth_col].max()]
    df_log_time.drop(ind, inplace=True)
    df_log_time.reset_index(drop=True, inplace=True)
    # Get depth and time arrays in df_dt.
    depth = df_dt[dt_depth_col].values
    time = df_dt[time_col].values
    # Fit interpolator.
    f = scipy.interpolate.interp1d(depth, time)
    # Transform depth-domain well log to time-domain.
    df_log_time[time_col] = f(df_log_time[log_depth_col].values)
    # Drop depth column.
    df_log_time.drop(columns=log_depth_col, inplace=True)
    # Re-arrange columns in df_log_copy.
    if isinstance(log_col, str):
        log_col = [log_col]
    new_col = [time_col] + log_col
    df_log_time = df_log_time[new_col]
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_log_time.columns)
    sub_col.remove(time_col)
    df_log_time.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_log_time.reset_index(drop=True, inplace=True)
    if fill_nan is not None:
        # Fill NaN.
        df_log_time.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with NaN.
        df_log_time.dropna(axis='index', how='any', inplace=True)
        df_log_time.reset_index(drop=True, inplace=True)
    # Change data type.
    df_log_time = df_log_time.astype('float32')
    if nominal:
        df_log_time[log_col] = df_log_time[log_col].astype('int32')
    return df_log_time


def cube2well(cube_file=None, cube_name=None, header_x=73, header_y=77, scl_x=1, scl_y=1,
              df=None, x_col=None, y_col=None, z_col=None, well_coord=None,
              w_x=25.0, w_y=25.0, w_z=2.0,
              mute=False):
    """
    Get up-hole trace data from cube and add them to well log data frame.
    :param cube_file: (String) - Cube file name. SEG-Y format.
    :param cube_name: (String) - Cube data name, which will be a new column name in well log data frame.
    :param header_x: (Integer) - Default is 73. Trace x coordinate's byte position in trace header.
                     73: source X, 181: X cdp.
    :param header_y: (Integer) - Default is 77. Trace y coordinate's byte position in trace header.
                     77: source Y, 185: Y cdp.
    :param scl_x: (Float) - The trace x coordinates will multiply this parameter.
                  Default is 1, which means not to scale the trace x coordinates read from trace header.
                  For example, if scl_x=0.1, the trace x coordinates from trace header will multiply 0.1.
    :param scl_y: (Float) - The trace y coordinates will multiply this parameter.
                  Default is 1, which means no to scale the trace y coordinates read from trace header.
                  For example, if scl_y=0.1, the trace y coordinates from trace header will multiply 0.1.
    :param df: (Pandas.Dataframe) - Time domain well log data frame, which has to contain x, y and z coordinate column,
                                    or at least z coordinate column but requires manually input well_coord.
    :param x_col: (String) - X-coordinate column name.
    :param y_col: (String) - Y-coordinate column name.
    :param z_col: (String) - Z-coordinate column name.
    :param well_coord: (List of floats) - Well location coordinates in the form of [well_x, well_y] (e.g. [10.0, 50.0]).
                       If the well log data frame does not have x and y columns, then all well log samples' coordinates
                       will be well_coord.
                       This parameter has no effect when x_col or y_col is not None.
    :param w_x: (Float) - Default is 25.0. The window's x size in which the well xy coordinates and cube xy coordinates
                will be matched.
    :param w_y: (Float) - Default is 25.0. The window's y size in which the well xy coordinates and cube xy coordinates
                will be matched.
    :param w_z: (Float) - Default is 2.0. The window's z size in which the well z coordinates and cube z coordinates
                will be matched.
    :param mute: (Bool) - If True, will not print progress and info on screen. Default is False.
    :return: df: (Pandas.Dataframe) - Well log data frame with a new column of up-hole trace data from cube.
    """
    # Load cube.
    with segyio.open(cube_file) as f:
        f.mmap()  # Memory map file for faster reading.
        if not mute:
            sys.stdout.write('Loading cube data...')
        cube = segyio.tools.cube(f)  # Load cube data.
        if not mute:
            sys.stdout.write('Done\n')
        x = np.zeros(shape=(f.tracecount,), dtype='float32')  # Initiate trace x-coordinates.
        y = np.zeros(shape=(f.tracecount,), dtype='float32')  # Initiate trace y-coordinates.
        for i in range(f.tracecount):
            if not mute:
                sys.stdout.write('\rLoading trace coordinates: %.2f%%' % ((i+1) / f.tracecount * 100))
            x[i] = f.header[i][header_x] * scl_x  # Get x-coordinates from trace header.
            y[i] = f.header[i][header_y] * scl_y  # Get y-coordinates from trace header.
        if not mute:
            sys.stdout.write('\n')
        x = x.reshape([len(f.ilines), len(f.xlines)], order='C')  # Re-shape x-coordinates array to match the cube.
        y = y.reshape([len(f.ilines), len(f.xlines)], order='C')  # Re-shape y-coordinates array to match the cube.
        t = f.samples  # Get sampling time.
        if not mute:
            print('Cube info:')
            print('X range: %.2f-%.2f [%d samples]' % (np.amin(x), np.amax(x), len(f.ilines)))
            print('Y range: %.2f-%.2f [%d samples]' % (np.amin(y), np.amax(y), len(f.xlines)))
            print('Z range: %.1fms-%.1fms [%d samples]' % (np.amin(t), np.amax(t), len(t)))
    f.close()
    # Get well log's 3D coordinates.
    if x_col is None and y_col is None and well_coord is not None:
        well_x = well_coord[0] * np.ones(len(df))
        well_y = well_coord[1] * np.ones(len(df))
    else:
        well_x = df[x_col].values
        well_y = df[y_col].values
    well_z = df[z_col].values
    if not mute:
        print('Well info:')
        print('X range: %.2f-%.2f' % (np.amin(well_x), np.amax(well_x)))
        print('Y range: %.2f-%.2f' % (np.amin(well_y), np.amax(well_y)))
        print('Z range: %.1fms-%.1fms [%d samples]' % (np.amin(well_z), np.amax(well_z), len(well_z)))
    # Match well log coordinates with cube data coordinates.
    dist_xy = scipy.spatial.distance.cdist(np.c_[well_x, well_y],
                                           np.c_[x.ravel(order='C'), y.ravel(order='C')],
                                           metric='euclidean')  # xy plane distance map.
    dist_z = scipy.spatial.distance.cdist(np.reshape(well_z, (-1, 1)),
                                          np.reshape(t, (-1, 1)),
                                          metric='minkowski', p=1)  # z-direction distance.
    indx, indy = np.unravel_index(np.argmin(dist_xy, axis=1), x.shape, order='C')
    indz = np.argmin(dist_z, axis=1)
    dist_xy_min = np.amin(dist_xy, axis=1)
    dist_z_min = np.amin(dist_z, axis=1)
    ixy = np.squeeze(np.argwhere(dist_xy_min < math.sqrt(w_x ** 2 + w_y ** 2)))
    iz = np.squeeze(np.argwhere(dist_z_min < w_z))
    ind = np.intersect1d(ixy, iz)
    if len(ind) == 0:
        raise ValueError('The well log has no sample can match the cube, please check the coordinates.')
    # Get data from the cube.
    df.loc[ind, cube_name] = cube[indx[ind], indy[ind], indz[ind]]
    return df


def feature_selection(X=None, y=None, estimator='RFC', n_tree=300, max_feature='auto', random_state=None,
                      auto=True, cv=5, step=1, feature_to_select=None, scoring=None,
                      show=True, **kwargs):
    """
    Select most informative features by recursive feature elimination (RFE) or RFE in a cross-validation loop (RFECV).
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html?highlight=rfe#sklearn.feature_selection.RFE
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html?highlight=rfe#sklearn.feature_selection.RFECV
    The RFE selects features by recursively considering smaller and smaller sets of features until it reaches the pre-
    defined number. The RFECV performs RFE in a cross-validation loop to find the optimal number of features.
    Here the Random Forests classifier and the Random Forests regressor are used as base estimators for classification
    and regression respectively.
    :param X: (pandas.DataFrame or numpy.2darray) - Sample features of shape (n_samples, n_features).
    :param y: (pandas.DataFrame or numpy.1darray) - Sample labels of shape (n_samples, ).
    :param estimator: (String) - Define using Random Forest classifier or regressor.
                      If 'RFC', which is the default option, will use the Random Forest classifier as the estimator.
                      If 'RFR', will use the Random Forest regressor as the estimator.
    :param n_tree: (Integer) - The amount of decision trees in Random Forest.
    :param max_feature: (Integer, float or string) - The number of features to consider when looking for the best split.
                        If integer, then consider the 'max_feature' features at each split.
                        If float, then max_features is a fraction and round(max_features * n_features) features are
                        considered at each split.
                        If 'auto', then max_features=sqrt(n_features). Default is 'auto'.
                        If 'sqrt', then max_features=sqrt(n_features) (same as 'auto').
                        If 'log2', then max_features=log2(n_features).
                        If None, then max_features=n_features.
    :param auto: (Bool) - Default is True. Whether to automatically select the optimal number of features. If true, will
                 use the RFECV, otherwise will use the RFE and require to input the desired number of selected features.
    :param cv: (Integer) - Determines the cross-validation splitting strategy. Default is 5-folds.
                           Only effective when auto is True.
    :param step: (Integer) - The number of features to remove at each iteration. Default is 1.
    :param feature_to_select: (Integer, float or None) - The number of features to select.
                              When auto is True, will use RFECV, and this parameter will be the minimum number of
                              features to be selected. Default is 1.
                              When auto is False, will use RFE, and this parameter will be the number of features to
                              select. Default is half of the features.
                              If integer, the parameter is the absolute number of features to select.
                              If float between 0 and 1, it is the fraction of features to select.
    :param scoring: (String) - The evaluation metric for cross-validation. Default is 'accuracy'.
    :param random_state: (Integer) - Controls both the randomness of the bootstrapping of the samples used when building
                                     trees (if bootstrap=True) and the sampling of the features to consider when looking
                                     for the best split at each node (if max_features < n_features). Default is None.
    :param show: (Bool) - Default is True.
                 Whether to show the feature rank, feature importance and cross-validation (CV) accuracy curve.
                 To show CV accuracy curve, RFECV must be used.
    :param kwargs: (Dictionary) - Key word arguments of the RandomForestClassifier or RandomForestRegressor from the
                                  scikit-learn.
                                  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#
                                  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#
    :return: X_new: (pandas.DataFrame or numpy.ndarray) - Selected sample features.
             y: (pandas.DataFrame or numpy.ndarray) - Sample classes or target values.
             selector: (object) - The selector object. For detailed information about its attributes, see
             https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html?highlight=rfe#sklearn.feature_selection.RFE
             https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html?highlight=rfe#sklearn.feature_selection.RFECV
    """
    # Check input info.
    if isinstance(X, pd.DataFrame):
        X_is_df = 1
        n_sample = X.values.shape[0]
        n_feature = X.values.shape[1]
    elif isinstance(X, np.ndarray):
        X_is_df = 0
        if X.ndim != 2:
            raise ValueError('The sample features X must be a 2-dimensional array.')
        n_sample = X.shape[0]
        n_feature = X.shape[1]
        feature_names = [f'Feature{i}' for i in range(n_feature)]
        X = pd.DataFrame(data=X, columns=feature_names)
    else:
        raise TypeError('The input X can either be pandas.DataFrame or numpy.2darray. Can not accept %s.' % type(X))
    if isinstance(y, pd.DataFrame):
        y_ndim = np.squeeze(y.values).ndim
        if y_ndim != 1:
            raise ValueError('The target y must be 1-d array-like, but got %d-d array-like instead.' % y_ndim)
    elif isinstance(y, np.ndarray):
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError('The target y must be 1-d array, but got %d-d array instead.' % y.ndim)
    else:
        raise TypeError('The input y can either be pandas.DataFrame or numpy.ndarray. Can not accept %s' % type(y))
    if n_sample != len(y):
        raise ValueError('The length of sample feature array is %d, but the length of sample label array is %d' %
                         (n_sample, len(y)))
    print('Data set information:')
    print('Sample number:%d' % n_sample)
    print('Feature number:%d' % n_feature)
    if estimator == 'RFC':
        if isinstance(y, pd.DataFrame):
            print('Class distribution:\n', dict(sorted(Counter(np.squeeze(y.values)).items())))
        else:
            print('Class distribution:\n', dict(sorted(Counter(y).items())))
    elif estimator == 'RFR':
        if isinstance(y, pd.DataFrame):
            y_min, y_max, y_ave, y_std = y.min(axis=0), y.max(axis=0), y.mean(axis=0), y.std(axis=0)
        else:
            y_min, y_max, y_ave, y_std = np.amin(y), np.amax(y), np.mean(y), np.std(y)
        tb = PrettyTable()
        tb.field_names = ['Minimum', 'Maximum', 'Average', 'Standard Deviation']
        tb.add_row([y_min, y_max, y_ave, y_std])
        tb.float_format = '.2'
        print(tb)
    else:
        raise ValueError("The estimator must be 'RFC' or 'RFR', can not accept '%s'" % estimator)
    # Initialize the estimator.
    if estimator == 'RFC':
        estimator = RandomForestClassifier(n_estimators=n_tree, max_features=max_feature, random_state=random_state,
                                           **kwargs)
    else:
        estimator = RandomForestRegressor(n_estimators=n_tree, max_features=max_feature, random_state=random_state,
                                          **kwargs)
    # If auto is True, select the optimal number of features by RFE in cross-validation loop.
    if auto:
        if feature_to_select is None:
            feature_to_select = 1
        selector = RFECV(estimator, min_features_to_select=feature_to_select, step=step, cv=cv, scoring=scoring)
    # If auto is False, select the user-defined number of features by recursive feature elimination (RFE).
    else:
        selector = RFE(estimator, n_features_to_select=feature_to_select, step=step)
    # Fit selector with features and targets, and generate a new dataset with selected features.
    sys.stdout.write('Selecting features...')
    if isinstance(y, pd.DataFrame):
        X_new = selector.fit_transform(X, np.squeeze(y.values))
    else:
        X_new = selector.fit_transform(X, y)
    if X_is_df:
        selected_feature = list(selector.get_feature_names_out())
        X_new = pd.DataFrame(data=X_new, columns=selected_feature)
    sys.stdout.write('Done\n')
    # Print selected features.
    print('%d features are selected by RFE:' % selector.n_features_)
    print(selected_feature)
    # Print the feature rank.
    df_rank = pd.DataFrame({'Rank': selector.ranking_}, index=selector.feature_names_in_)
    df_rank.sort_values(by=['Rank'], inplace=True)
    # Print feature importance.
    feature_importance = selector.estimator_.feature_importances_
    df_importance = pd.DataFrame({'Importance': feature_importance}, index=selected_feature)
    df_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
    if show:
        print('Feature importance:\n', df_importance)
    # If automatically select the optimal number of features, draw a curve of cross-validation accuracy.
    if auto:
        plt.figure(figsize=[14, 8])
        plt.style.use('bmh')
        plt.title('RFECV Result', fontsize=20)
        plt.xlabel('Number of feature selected', fontsize=18)
        plt.ylabel('CV score', fontsize=18)
        plt.tick_params(labelsize=17)
        x_axis = range(feature_to_select, selector.n_features_in_ + 1)
        plt.xticks(x_axis)
        y_axis = selector.cv_results_['mean_test_score']
        yerr = selector.cv_results_['std_test_score']
        plt.plot(x_axis, y_axis, 'ro--', lw=2, markeredgecolor='k', ms=8)
        plt.fill_between(x_axis, y_axis + yerr, y_axis - yerr, alpha=0.25)
        line = plt.axvline(x=selector.n_features_, ls='--', c='k', lw=3,
                           label='n_features = {}\nscore = {:0.3f}'.format(selector.n_features_, y_axis.max()))
        plt.legend(handles=[line], loc='best', fontsize=18)
        # plt.savefig('cv_curve.jpg', dpi=600)
        if show:
            plt.show()
    return X_new, y, selector


def balance_class(X=None, y=None, smote_strategy='not majority', enn_strategy='all', k_smote=5, k_enn=3, kind_sel='all',
                  random_state=None, plot_distribution=True):
    """
    Use SMOTE-ENN to balance the class distribution of the dataset.
    :param X: (pandas.DataFrame or numpy.ndarray) - Sample features of shape (n_samples, n_features).
    :param y: (pandas.DataFrame or numpy.1darray) - Sample target values of shape (n_samples, ).
                                                    Class labels in classification, real numbers in regression.
    :param smote_strategy: (String, float or dictionary) - Determine how to use SMOTE to resample the dataset.
                            Default is 'not majority'.
                            'minority': resample only the minority class;
                            'not minority': resample all classes but the minority class;
                            'not majority': resample all classes but the majority class;
                            'all': resample all classes;
                            'auto': equivalent to 'not majority'.
                            For more detailed information, see
                            https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    :param enn_strategy: (String, float or dictionary) - Determine how to use ENN to resample the dataset.
                         Default is 'all'.
                         'majority': resample only the majority class;
                         'not minority': resample all classes but the minority class;
                         'not majority': resample all classes but the majority class;
                         'all': resample all classes;
                         'auto': equivalent to 'not minority'.
                         For more detailed information, see
                         https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html
    :param k_smote: (Integer) - Number of nearest neighbours to used to construct synthetic samples. Default is 5.
    :param k_enn: (Integer) - Size of the neighbourhood to consider to compute the nearest neighbors. Default is 3.
    :param kind_sel: (String) - Strategy to use in order to exclude samples when using ENN. Default is 'all'.
                     If 'all', all neighbours will have to agree with the samples of interest to not be excluded.
                     If 'mode', the majority vote of the neighbours will be used in order to exclude a sample.
                     The strategy "all" will be less conservative than 'mode'. Thus, more samples will be removed when
                     kind_sel="all" generally.
    :param random_state: (Integer) - Control the randomization of the algorithm. Default is None.
    :param plot_distribution: (Bool) - Determines whether to plot the class distribution before and after class-
                              balancing. Default is True.
    :return: X_new: (pandas.DataFrame or numpy.ndarray) - Sample features after class-balancing.
             y_new: (pandas.DataFrame or numpy.ndarray) - Sample class labels after class-balancing.
    """
    # Check input info.
    if isinstance(X, pd.DataFrame):
        n_sample = X.values.shape[0]
        n_feature = X.values.shape[1]
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError('The sample features X must be a 2-dimensional array.')
        n_sample = X.shape[0]
        n_feature = X.shape[1]
    else:
        raise TypeError('The input X can either be pandas.DataFrame or numpy.2darray. Can not accept %s.' % type(X))
    if isinstance(y, pd.DataFrame):
        y_ndim = np.squeeze(y.values).ndim
        if y_ndim != 1:
            raise ValueError('The target y must be 1-d array-like, but got %d-d array-like instead.' % y_ndim)
    elif isinstance(y, np.ndarray):
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError('The target y must be a 1-d array, but got a %d-d array instead.' % y.ndim)
    print('Data set information:')
    print('Sample number:%d' % n_sample)
    print('Feature number:%d' % n_feature)
    if isinstance(y, pd.DataFrame):
        print('Class distribution (before balancing):\n', dict(sorted(Counter(np.squeeze(y.values)).items())))
    else:
        print('Class distribution (before balancing):\n', dict(sorted(Counter(y).items())))
    ys = y.copy()
    # Scale all features.
    scaler = MinMaxScaler(feature_range=(0, 1))
    Xs = scaler.fit_transform(X)
    # Class-balancing.
    if smote_strategy is not None:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_smote, random_state=random_state)
        Xs, ys = smote.fit_resample(Xs, ys)
    if enn_strategy is not None:
        enn = EditedNearestNeighbours(sampling_strategy=enn_strategy, n_neighbors=k_enn, kind_sel=kind_sel)
        Xs, ys = enn.fit_resample(Xs, ys)
    if isinstance(ys, pd.DataFrame):
        print('Class distribution (after balancing):\n', dict(sorted(Counter(np.squeeze(ys.values)).items())))
    else:
        print('Class distribution (after balancing):\n', dict(sorted(Counter(ys).items())))
    # Inverse transform the features.
    Xs = scaler.inverse_transform(Xs)
    if isinstance(X, pd.DataFrame):
        Xs = pd.DataFrame(data=Xs, columns=X.columns)
    # Visualize sample class distribution before and after class-balancing.
    if plot_distribution:
        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        if isinstance(y, pd.DataFrame):
            class_distribution(data=np.squeeze(y.values), titlename='Class distribution before balancing',
                               make_plot=False, bar_width=0.2)
        else:
            class_distribution(data=y, titlename='Class distribution before balancing', make_plot=False, bar_width=0.2)
        plt.subplot(122)
        if isinstance(ys, pd.DataFrame):
            class_distribution(data=np.squeeze(ys.values), titlename='Class distribution after balancing',
                               make_plot=False, bar_width=0.2)
        else:
            class_distribution(data=ys, titlename='Class distribution after balancing', make_plot=False,
                               bar_width=0.2)
        plt.tight_layout()
        plt.show()
    return Xs, ys
