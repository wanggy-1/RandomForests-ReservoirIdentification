import pandas as pd
import numpy as np
import sys
import joblib
import segyio
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter
from prettytable import PrettyTable


def rf_train(X=None, y=None, estimator='RFC', n_tree=300, max_feature=None, random_state=None, model_name=None,
             **kwargs):
    """
    Train a Random Forest classifier.
    :param X: (pandas.DataFrame or numpy.2darray) - Sample features of shape (n_samples, n_features).
    :param y: (pandas.DataFrame or numpy.1darray) - Sample target values of shape (n_samples, ).
                                                    Class labels in classification, real numbers in regression.
    :param estimator: (String) - Determines to use Random Forest classifier or regressor as the estimator.
                                 If 'RFC', which is the default option, will use the RF classifier as the estimator.
                                 If 'RFR', will use the RF regressor as the estimator.
    :param n_tree: (Integer) - The number of trees in the forest.
    :param max_feature: (Integer, float or string) - The number of features to consider when looking for the best split.
                        If integer, then consider the 'max_feature' features at each split.
                        If float, then max_features is a fraction and round(max_features * n_features) features are
                        considered at each split.
                        If 'auto', then max_features=sqrt(n_features). Default is 'auto'.
                        If 'sqrt', then max_features=sqrt(n_features) (same as 'auto').
                        If 'log2', then max_features=log2(n_features).
                        If None, then max_features=n_features.
    :param random_state: (Integer) - Controls both the randomness of the bootstrapping of the samples used when building
                                     trees (if bootstrap=True) and the sampling of the features to consider when looking
                                     for the best split at each node (if max_features < n_features). Default is None.
    :param model_name: (String) - File name to save the RF model. Default is 'RFC_Model.model'.
    :param kwargs: (Dictionary) - Key word arguments of the RandomForestClassifier or RandomForestRegressor from the
                                  scikit-learn.
                                  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#
                                  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#
    :return: rf: (Object) - The trained Random Forest estimator.
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
        feature_names = [f'Feature{i}' for i in range(n_feature)]
        X = pd.DataFrame(data=X, columns=feature_names)
    else:
        raise TypeError('The input X can either be pandas.DataFrame or numpy.2darray. Can not accept %s.' % type(X))
    if isinstance(y, pd.DataFrame):
        y = np.squeeze(y.values)
        if y.ndim != 1:
            raise ValueError('The target y must be 1-d array-like, but got %d-d array-like instead.' % y.ndim)
    elif isinstance(y, np.ndarray):
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError('The target y must be a 1-d array, but got a %d-d array instead.' % y.ndim)
    else:
        raise TypeError('The input y can either be pandas.DataFrame or numpy.ndarray. Can not accept %s' % type(y))
    if n_sample != len(y):
        raise ValueError('The length of sample feature array is %d, but the length of sample label array is %d' %
                         (n_sample, len(y)))
    print('Data set information:')
    print('Sample number:%d' % n_sample)
    print('Feature number:%d' % n_feature)
    if estimator == 'RFC':
        print('Class distribution:\n', dict(sorted(Counter(y).items())))
    elif estimator == 'RFR':
        y_min, y_max, y_ave, y_std = np.amin(y), np.amax(y), np.mean(y), np.std(y)
        tb = PrettyTable()
        tb.field_names = ['Minimum', 'Maximum', 'Average', 'Standard Deviation']
        tb.add_row([y_min, y_max, y_ave, y_std])
        tb.float_format = '.2'
        print(tb)
    else:
        raise ValueError("The estimator must be 'RFC' or 'RFR', can not accept '%s'" % estimator)
    # The default mode of feature selection for node splitting.
    if max_feature is None:
        max_feature = 'auto'
    # Initialize the estimator.
    if estimator == 'RFC':
        rf = RandomForestClassifier(n_estimators=n_tree, max_features=max_feature, random_state=random_state, **kwargs)
    else:
        rf = RandomForestRegressor(n_estimators=n_tree, max_features=max_feature, random_state=random_state, **kwargs)
    # Train the estimator.
    sys.stdout.write('Training...')
    rf.fit(X, y)
    sys.stdout.write('Done.\n')
    # Output feature importance.
    df_importance = pd.DataFrame({'Feature names': rf.feature_names_in_, 'Importance': rf.feature_importances_})
    df_importance.sort_values(by='Importance', axis='index', ascending=False, inplace=True)
    print('Feature Importance:\n', df_importance)
    # Save the estimator.
    if model_name is None:
        if estimator == 'RFC':
            model_name = 'RFC_Model.model'
        else:
            model_name = 'RFR_Model.model'
    joblib.dump(rf, model_name)
    # Return the trained estimator.
    return rf


def rf_predict(estimator=None, X=None, sgy_filelist=None, mode='slice', iline_range=None, xline_range=None,
               t_range=None, proba=False):
    """
    Use a trained Random Forest estimator to predict the sample values (class labels in classification and real numbers
    in regression).
    :param estimator: (object) - The trained Random Forest estimator.
    :param sgy_filelist: (List of strings) - The SEG-Y file list. If the feature data are in SEG-Y file, then input all
                         SEG-Y file names as a list. In this case, the program will load feature data from SEG-Y files
                         and the parameter X will have no effect. Notice that all SEG-Y files must have the same
                         structure (inline, cross-line and t range).
    :param X: (pandas.DataFrame or numpy.2darray) - Sample features of shape (n_samples, n_features).
    :param mode: (String) - Determine how to predict the sample values. Only effective when the inputs are SEG-Y files.
                            If 'slice', will read SEG-Y files repeatedly and do prediction slice by slice, memory-saving
                            but time-consuming. Useful when have to input many features and the files are big.
                            If 'cube', will read all SEG-Y files do prediction for the whole cube at once, time-saving
                            but memory-consuming.
    :param iline_range: (List of integers) - Define the inline range for prediction. Default is to use all inlines.
    :param xline_range: (List of integers) - Define the cross-line range for prediction. Default is to use all cross-
                        lines.
    :param t_range: (List of integers) - Define the time range for prediction. Default is to use all time samples.
    :param proba: (Bool) - Whether to predict the class probabilities of samples.
                           Default is False, which is to predict the classes of samples.
                           If True, will predict the class probabilities of samples.
                           Notice that can not predict class probabilities in regression.
    :return: p: (numpy.2darray) - When proba is True, output the predicted class probabilities of samples,
                                  with the shape of (n_samples, n_classes).
             v: (numpy.1darray) - When proba is False, output the predicted values of samples (class labels in
                                  classification and real numbers in regression), with the shape of (n_samples, ).
    """
    warnings.simplefilter('ignore')
    # When input sample features are in SEG-Y files.
    if sgy_filelist is not None:
        # Check SEG-Y file structures.
        sys.stdout.write('Checking SEG-Y file info...')
        for i, file in zip(range(len(sgy_filelist)), sgy_filelist):
            with segyio.open(file) as f:
                temp_il = f.ilines
                temp_xl = f.xlines
                temp_t = f.samples
                if i == 0:
                    iline, xline, t = temp_il.copy(), temp_xl.copy(), temp_t.copy()
            f.close()
            if ((temp_il == iline).all()) & ((temp_xl == xline).all()) & ((temp_t == t).all()):
                continue
            else:
                raise ValueError('The structure of SEG-Y file %d is different from SEG-Y file 1' % (i+1))
        sys.stdout.write('Clear\n')
        # Customize prediction area.
        if iline_range is None:
            il_ind1, il_ind2 = 0, len(iline) - 1
        else:
            il_ind1 = np.squeeze(np.argwhere(iline == iline_range[0]))
            il_ind2 = np.squeeze(np.argwhere(iline == iline_range[1]))
        if xline_range is None:
            xl_ind1, xl_ind2 = 0, len(xline) - 1
        else:
            xl_ind1 = np.squeeze(np.argwhere(xline == xline_range[0]))
            xl_ind2 = np.squeeze(np.argwhere(xline == xline_range[1]))
        if t_range is None:
            t_ind1, t_ind2 = 0, len(t) - 1
        else:
            t_ind1 = np.squeeze(np.argwhere(t == t_range[0]))
            t_ind2 = np.squeeze(np.argwhere(t == t_range[1]))
        iline = iline[il_ind1: il_ind2 + 1]
        xline = xline[xl_ind1: xl_ind2 + 1]
        t = t[t_ind1: t_ind2 + 1]
        print('Customized prediction area:')
        print('Inline: [%d-%d]' % (iline[0], iline[-1]))
        print('Xline: [%d-%d]' % (xline[0], xline[-1]))
        print('Time: [%d-%d]' % (t[0], t[-1]))
        # Do prediction slice by slice to save memory, but will take more time.
        if mode == 'slice':
            # Initialize a cube to store the prediction result.
            if proba:
                p = np.zeros(shape=(len(iline), len(xline), len(t), estimator.n_classes_), dtype=np.float32)
            else:
                v = np.zeros(shape=(len(iline), len(xline), len(t)), dtype=np.float32)
            # If inline number is smaller than (or equal to) cross-line number, do prediction along inline direction.
            # Initialize a feature data container.
            X = np.zeros(shape=(len(xline) * len(t), len(sgy_filelist)), dtype=np.float32)
            if len(iline) <= len(xline):
                # Load feature data from SEG-Y files.
                for i, n_il in zip(range(len(iline)), iline):
                    for j, file in zip(range(len(sgy_filelist)), sgy_filelist):
                        sys.stdout.write('\rPredicting...[Line%d, File%d]%.2f%%' %
                                         (i+1, j+1, (i*len(sgy_filelist)+j+1)/(len(iline)*len(sgy_filelist))*100))
                        with segyio.open(file) as f:
                            f.mmap()
                            data = f.iline[n_il][xl_ind1: xl_ind2+1, t_ind1: t_ind2+1]
                            data = np.ravel(data, order='C')
                        f.close()
                        X[:, j] = data
                    # Do prediction and transfer the prediction result to the cube.
                    if proba:
                        s = estimator.predict_proba(X)  # Predict class probabilities.
                        s = np.reshape(s, newshape=(len(xline), len(t), estimator.n_classes_), order='C')
                        p[i, :, :, :] = s
                    else:
                        s = estimator.predict(X)  # Predict class labels or regression values.
                        s = np.reshape(s, newshape=(len(xline), len(t)), order='C')
                        v[i, :, :] = s
                sys.stdout.write('\n')
            # If cross-line number is smaller than inline number, do prediction along cross-line direction.
            else:
                # Initialize a feature data container.
                X = np.zeros(shape=(len(iline) * len(t), len(sgy_filelist)), dtype=np.float32)
                for i, n_xl in zip(range(len(xline)), xline):
                    for j, file in zip(range(len(sgy_filelist)), sgy_filelist):
                        sys.stdout.write('\rPredicting...[Line%d, File%d]%.2f%%' %
                                         (i+1, j+1, (i*len(sgy_filelist)+j+1)/(len(xline)*len(sgy_filelist))*100))
                        with segyio.open(file) as f:
                            f.mmap()
                            data = f.xline[n_xl][il_ind1: il_ind2+1, t_ind1: t_ind2+1]
                            data = np.ravel(data, order='C')
                        f.close()
                        X[:, j] = data
                    # Do prediction and transfer the prediction result to the cube.
                    if proba:
                        s = estimator.predict_proba(X)  # Predict class probabilities.
                        s = np.reshape(s, newshape=(len(iline), len(t), estimator.n_classes_), order='C')
                        p[:, i, :, :] = s
                    else:
                        s = estimator.predict(X)  # Predict class labels or regression values.
                        s = np.reshape(s, newshape=(len(iline), len(t)), order='C')
                        v[:, i, :] = s
                sys.stdout.write('\n')
            if proba:
                return p
            else:
                return v
        # Do prediction to the whole cube at once, time-saving but will consume more memory.
        elif mode == 'cube':
            # Initialize a feature data container.
            X = np.zeros(shape=(len(iline) * len(xline) * len(t), len(sgy_filelist)), dtype=np.float32)
            # Load feature data from SEG-Y files.
            for i, file in zip(range(len(sgy_filelist)), sgy_filelist):
                sys.stdout.write('\rLoading data from file %d/%d' % (i+1, len(sgy_filelist)))
                data = segyio.tools.cube(file)
                data = data[il_ind1: il_ind2+1, xl_ind1: xl_ind2+1, t_ind1: t_ind2+1]
                data = np.ravel(data, order='C')
                X[:, i] = data
            sys.stdout.write('\n')
            # Do prediction.
            if proba:  # Predict class probabilities.
                sys.stdout.write('Predicting...')
                p = estimator.predict_proba(X)
                p = np.reshape(p, newshape=(len(iline), len(xline), len(t), estimator.n_classes_), order='C')
                sys.stdout.write('Done\n')
                return p
            else:  # Predict class labels or regression values.
                sys.stdout.write('Predicting...')
                v = estimator.predict(X)
                v = np.reshape(v, newshape=(len(iline), len(xline), len(t)), order='C')
                sys.stdout.write('Done\n')
                return v
        else:
            raise ValueError("No such mode as %s. Options are 'slice' or 'cube'" % mode)
    else:
        # Check input data info.
        if isinstance(X, pd.DataFrame):
            n_sample = X.shape[0]
            n_feature = X.shape[1]
        elif isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError('The sample features X must be a 2-dimensional array.')
            n_sample = X.shape[0]
            n_feature = X.shape[1]
        else:
            raise TypeError('The input X can either be pandas.DataFrame or numpy.2darray. Can not accept %s.' % type(X))
        print('Sample number:%d' % n_sample)
        print('Feature number:%d' % n_feature)
    # Predict the classes or class probabilities of X.
    if proba:  # Predict class probabilities.
        sys.stdout.write('Predicting...')
        p = estimator.predict_proba(X)
        sys.stdout.write('Done\n')
        return p
    else:  # Predict class labels or regression values.
        sys.stdout.write('Predicting...')
        v = estimator.predict(X)
        sys.stdout.write('Done\n')
        return v
