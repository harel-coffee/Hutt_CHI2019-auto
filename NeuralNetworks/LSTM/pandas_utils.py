import pandas as pd
import numpy as np
from collections import OrderedDict

# get min/max values for all columns

def get_min_max(df):
    """
    returns dictionary with min and max values for each numeric column,
    :param df: (pandas dataframe): any pandas data frame
    :return: dictionary: dictionary with columns as keys and list of
         [min,max]  as values.
    """

    min_df = df.min() # this drops some columns for some reason
    max_df = df.max()
    min_max_dict = OrderedDict()

    for col in min_df.index:
        # return as np.array so can use dtype on it to check for non-numeric
        tmp = np.array([min_df[col],max_df[col]])
        min_max_dict[col] = tmp

    return min_max_dict

def get_cols_string_or_not(df,string=True, include_cols=None ):
    """
    :param df: (dataframe) dataframe with columns to extract
    :param string: (boolean) if True return list of columns with strings,
                if False, return list of numeric columns
    :param include_cols: restrict search to this list if included
    :return: (list) list of columns with only numeric or only non-numeric values
    """
    if include_cols:
        # get new pd frame with included columns
        df_of_col = df.loc[:, include_cols]
        all_numeric = df_of_col.select_dtypes(include=np.number).columns.values.tolist()
    else:
        include_cols = df.columns.values.tolist()
        all_numeric = df.select_dtypes(include=np.number).columns.values.tolist()
    if string:
        tmp = [col for col in include_cols if col not in all_numeric]
        return tmp
    else:
        return all_numeric

def get_corr_from_predictions(preds,y,metrics='mse',
                              corr = ['spearman','kendall'],
                              round = False):
    """
    given predictions and labels, returns rank order correlations
    :param preds: (numpy array): predictions from mse or one hot softmax
    :param y: y (numpy array): labels
    :param metrics: (string): metric used in loss function, 'mse' or 'accuracy'
            'accuracy' assumes one hot encoding, 'mse' assumes regression
    :param corr: (string): list of corr methods to use
    :param round: (boolean) if set round the regression values
    :return: dictionary:  = dictionary with keys for methods, values are
             coefficients
    """
    corr_values = {}
    if metrics == 'accuracy':
        y_out = np.argmax(y, axis=1)
        x_out = np.argmax(preds, axis=1)
    elif metrics == 'mse':
        y_out = np.reshape(y, (y.shape[0],))
        x_out = np.reshape(preds, (preds.shape[0],))
        if round:  #  option for rounding output of regression to whole values
            x_out = np.round(x_out)
    else:
        print('get_corr_from_predictions(): option for metrics = ',metrics,
              ' not supported, ' + 'returning zeros')
        for item in corr:
            corr_values[item] = 0.0
        return corr_values

    df = pd.DataFrame(data = {'x_out': x_out,'y_out': y_out})
    # df = pd.DataFrame(data=y_out, columns=['y_out'])
    # df['x_out'] = pd.Series(x_out, index=df.index)
    for item in corr:
        corr_values[item] = df.loc[:, ['x_out', 'y_out']].corr(method=item).iloc[0][1]
    return corr_values

def accuracy_from_round(preds,y):
    """
    given predictions and labels from mse regression, return rounded output
    predictions
    :param preds: (numpy array): predictions from mse
    :param y: (numpy array): labels
    :return: accuracy in percent
    """
    y_out = np.reshape(y, (y.shape[0],))
    x_out = np.reshape(preds, (preds.shape[0],))
    x_out = np.round(x_out)
    correct = np.sum((x_out-y_out)==0)
    return correct/y_out.shape[0]

def get_unique_counts(df,column_name):
    """
    Returns a series object containing counts of all unique valuess in column_name
    :param df: (pandas dataframe)
    :param column_name: (string) column name to search
    :return: (series object) contains counts of all unique values and appends
             an index 'max' to show the max count of all indices
    """
    count_unique = df[column_name].value_counts()
    count_unique['max'] = count_unique.max()

    return count_unique


