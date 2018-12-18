# Contains utility functions for neural network code, with documentation for each.
import numpy as np


def z_standardize_and_quantile_scaling(train_pandas_df, test_pandas_df, columns, scaling_factor,
                                       lower_quantile, upper_quantile):
    """
    Non-destructive: makes new 'proc_' columns rather than replacing data.
    Z-score data using training set.
    Then center to median and scale data in lower quantile to upper quantile as
        [ -1/2 scaling_factor, 1/2 scaling_factor ]
    Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :param scaling_factor: desired interquartile range of re-scaled data
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        new_col = 'proc_{}'.format(col)
        # calculate z-score on training data
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[new_col] = (train[col] - train_m) / train_sd
        test[new_col] = (test[col] - train_m) / train_sd
        # re-scale using median & interquartile distance
        train_median = train[new_col].median()
        train_lower_quant = train[new_col].quantile(lower_quantile, interpolation='midpoint')
        train_upper_quant = train[new_col].quantile(upper_quantile, interpolation='midpoint')
        train_iq_dist = train_upper_quant - train_lower_quant
        train[new_col] = scaling_factor * (train[new_col] - train_median) / train_iq_dist
        test[new_col] = scaling_factor * (test[new_col] - train_median) / train_iq_dist
    return train, test


def z_standardize_and_norm_with_scaled_recenter(train_pandas_df, test_pandas_df, columns, scaling_factor):
    """
    Non-destructive: makes new 'proc_' columns rather than replacing data.
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage).
    Then normalize to training set range from -1 to 1.
    Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        new_col = 'proc_{}'.format(col)
        # calculate z-score on training data
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[new_col] = (train[col] - train_m) / train_sd
        test[new_col] = (test[col] - train_m) / train_sd
        # normalize using z-scored training data
        train_m = train[new_col].mean()
        train_max = train[new_col].max()
        train_min = train[new_col].min()
        train_range = train_max - train_min
        train[new_col] = scaling_factor * (train[new_col] - train_m) / train_range
        test[new_col] = scaling_factor * (test[new_col] - train_m) / train_range
        # re-center to the midpoint of the rescaled values
        #   because the data may be skewed from the mean
        train_rescale_max = train[new_col].max()
        train_rescale_min = train[new_col].min()
        train_rescale_range = train_rescale_max - train_rescale_min
        train_rescale_midpoint = train_rescale_max - (train_rescale_range / 2)
        train[new_col] = train[new_col] - train_rescale_midpoint
        test[new_col] = test[new_col] - train_rescale_midpoint
    return train, test


def z_standardize_and_norm_with_scaling(train_pandas_df, test_pandas_df, columns, scaling_factor):
    """
    Non-destructive: makes new 'proc_' columns rather than replacing data.
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage).
    Then normalize to training set range from -1 to 1.
    Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        new_col = 'proc_{}'.format(col)
        # calculate z-score on training data
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[new_col] = (train[col] - train_m) / train_sd
        test[new_col] = (test[col] - train_m) / train_sd
        # normalize using z-scored training data
        train_m = train[new_col].mean()
        train_max = train[new_col].max()
        train_min = train[new_col].min()
        train[new_col] = scaling_factor * (train[new_col] - train_m) / (train_max - train_min)
        test[new_col] = scaling_factor * (test[new_col] - train_m) / (train_max - train_min)
    return train, test


def z_standardize_and_norm(train_pandas_df, test_pandas_df, columns):
    """
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage).
    Then normalize to training set range from -1 to 1.
    Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[col] = (train[col] - train_m) / train_sd
        test[col] = (test[col] - train_m) / train_sd
        train_m = train[col].mean()
        train_max = train[col].max()
        train_min = train[col].min()
        train[col] = (train[col] - train_m) / (train_max - train_min)
        test[col] = (test[col] - train_m) / (train_max - train_min)
    return train, test


def z_standardize(train_pandas_df, test_pandas_df, columns):
    """
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage). Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[col] = (train[col] - train_m) / train_sd
        test[col] = (test[col] - train_m) / train_sd
    return train, test


def normalize(train_pandas_df, test_pandas_df, columns):
    """
    Mean center and normalize using training set mean, min, max.
    Results in approximate range from -1 to 1. Test set may exceed training set range.
    Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame
    :param columns: list of columns to standardize (typically in + out columns in the network)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        train_m = train[col].mean()
        train_max = train[col].max()
        train_min = train[col].min()
        train[col] = (train[col] - train_m) / (train_max - train_min)
        test[col] = (test[col] - train_m) / (train_max - train_min)
    return train, test


def normalize_zero_to_value(train_pandas_df, test_pandas_df, columns, norm_value, na_fill=0):
    """
    Normalize in range [0, 1] by dividing max value
        NOTE: Assumes data min is 0 (semantically or actually)
    :param train_pandas_df: train dataset to calculate statistics on and apply norm
    :param test_pandas_df: test dataset to apply norm
    :param columns: columns to normalize
    :param na_fill: value to replace NA with -- default 0
    :return:
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        new_col = 'proc_{}'.format(col)
        train[new_col] = train[col] / norm_value
        test[new_col] = test[col] / norm_value
        train[new_col] = train[new_col].fillna(na_fill)
        test[new_col] = test[new_col].fillna(na_fill)
    return train, test


def normalize_zero_to_max(train_pandas_df, test_pandas_df, columns, na_fill=0):
    """
    Normalize in range [0, 1] by dividing max value
        NOTE: Assumes data min is 0 (semantically or actually)
    :param train_pandas_df: train dataset to calculate statistics on and apply norm
    :param test_pandas_df: test dataset to apply norm
    :param columns: columns to normalize
    :param na_fill: value to replace NA with -- default 0
    :return:
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy()
    for col in columns:
        new_col = 'proc_{}'.format(col)
        train_max = train[col].max()
        train[new_col] = train[col] / train_max
        test[new_col] = test[col] / train_max
        train[new_col] = train[new_col].fillna(na_fill)
        test[new_col] = test[new_col].fillna(na_fill)
    return train, test


def make_sequences(pandas_df,
                   in_columns,
                   primary_id_col=None,
                   secondary_id_col=None,
                   sequence_len=10,
                   min_valid_prop=.7,
                   missing_fill=0,
                   overlap=1):
    """
    Create sequences of data of a specific length using a sliding window that moves by "overlap".
    Sequences are necessary for training sequential models like LSTMs/GRUs. Also finds the indices
    of data points that can possibly be predicted, so targets (i.e. y) can be extracted.

    :param pandas_df: pandas DataFrame object with sequential data
    :param in_columns: list of columns to include in the resulting input sequences (i.e., X)
    :param primary_id_col: if specified, sequences will not overlap different primary IDs;
                               i.e., no sequence will include data from more than one primary ID
    :param secondary_id_col: if specified, sequences will not overlap on secondary IDs;
                                i.e., no sequence will include data from more than one secondary ID
    :param sequence_len: number of rows to include in each sequence
    :param min_valid_prop: minimum proportion of non-missing data points per sequence; the sequence
                           will not be included in the result if this constraint is not met
    :param missing_fill: replace missing values with this (iff min_valid_prop is satisfied)
    :param overlap: number of rows to move the sliding window forward by (usually 1)
    :returns: tuple of (ndarray of sequences [i.e. inputs], list of target indices)
    """
    seqs = []
    indices = []
    for i in range(sequence_len, len(pandas_df) + 1, overlap):
        seq = pandas_df.iloc[i - sequence_len:i][in_columns]
        if primary_id_col and \
                        len(pandas_df.iloc[i - sequence_len:i][primary_id_col].unique()) > 1:
            continue  # Cannot have sequences spanning multiple primary IDs
        if secondary_id_col and \
                        len(pandas_df.iloc[i - sequence_len:i][secondary_id_col].unique()) > 1:
            continue  # Cannot have sequences spanning multiple secondary IDs
        if seq.count().sum() / float(sequence_len * len(in_columns)) < min_valid_prop:
            continue  # Not enough valid data in this sequence.
        seqs.append(seq.fillna(missing_fill).values)
        indices.append(i - 1)
    return np.array(seqs), indices

# def make_sequences2(pandas_df,
#                    in_columns,
#                    primary_id_col=None,
#                    secondary_id_col=None,
#                    sequence_len=10,
#                    min_valid_prop=.7,
#                    missing_fill=0,
#                    overlap=1):
#     """
#     Create sequences of data of a specific length using a sliding window that moves by "overlap".
#     Sequences are necessary for training sequential models like LSTMs/GRUs. Also finds the indices
#     of data points that can possibly be predicted, so targets (i.e. y) can be extracted.
#
#     :param pandas_df: pandas DataFrame object with sequential data
#     :param in_columns: list of columns to include in the resulting input sequences (i.e., X)
#     :param primary_id_col: if specified, sequences will not overlap different primary IDs;
#                                i.e., no sequence will include data from more than one primary ID
#     :param secondary_id_col: if specified, sequences will not overlap on secondary IDs;
#                                 i.e., no sequence will include data from more than one secondary ID
#     :param sequence_len: number of rows to include in each sequence
#     :param min_valid_prop: minimum proportion of non-missing data points per sequence; the sequence
#                            will not be included in the result if this constraint is not met
#     :param missing_fill: replace missing values with this (iff min_valid_prop is satisfied)
#     :param overlap: number of rows to move the sliding window forward by (usually 1)
#     :returns: tuple of (ndarray of sequences [i.e. inputs], list of target indices)
#     """
#     seqs = []
#     indices = []
#
#     sort_cols = in_columns + [primary_id_col] + [secondary_id_col]
#     seq = pandas_df[sort_cols].groupby([primary_id_col, secondary_id_col]).values
#     sort_cols = out_columns + [primary_id_col] + [secondary_id_col]]
#     y_label = pandas_df[sort_cols].groupby([primary_id_col, secondary_id_col]).first()
#     return

def make_instances(pandas_df,
                   in_columns,
                   primary_id_col=None,
                   secondary_id_col=None,):
    """
    creates instance data from sequence data that was repeated (demographics)
    assumes models that use same final ouput label so this returns same length
    sequence aligned to match the original time series sequence
    :param pandas_df: (dataframe) contains only relevant data from k folds
    :param in_columns: (list) list of columns to extract
    :param primary_id_col: (string) id of primary column to group by
    :param secondary_id_col: (string) id of secondary column to group by
    :return: extracted sequence as a numpy array, indices to labels
    """
    sort_cols = in_columns + [primary_id_col] + [secondary_id_col]
    #seq = pandas_df[sort_cols].groupby(primary_id_col,secondary_id_col).first()
    seq = pandas_df[sort_cols].groupby([primary_id_col, secondary_id_col]).first()
    return seq.values, seq.index.values.tolist()  # .values returns as a numpy array



if __name__ == '__main__':
    # Do some testing.
    import pandas as pd

    df = pd.DataFrame.from_records([{'pid': 'p1', 'a': 1, 'b': 2, 'c': 'x'},
                                    {'pid': 'p1', 'a': 4.5},
                                    {'pid': 'p2', 'a': 2, 'b': 1, 'c': 'y'},
                                    {'pid': 'p2', 'a': 3, 'b': 0, 'c': 'x'},
                                    {'pid': 'p2', 'a': 4, 'b': 0, 'c': 'z'}])
    print(df)
    print('Sequences with strict min_valid_prop requirement:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.9))
    print('Sequences with less strict min valid data:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.5))
    print('Sequences with string column:')
    print(make_sequences(df, ['c'], sequence_len=3, min_valid_prop=.5, missing_fill=0))
    print('Sequences bounded by participant id:')
    print(make_sequences(df, ['a'], primary_id_col='pid', sequence_len=2))

    print('Standardized with pid=p2 as train, pid=p1 as test:')
    a, b = z_standardize(df[df.pid == 'p2'], df[df.pid == 'p1'], ['a', 'b'])
    print(a)
    print(b)
