import csv
import datetime
import itertools
import multiprocessing
import os
import pickle
import psutil
import sys
from time import sleep
import numpy as np
import pandas as pd
# PyTables required for HDFStore functionality of pandas,
# so just putting in an explicit import here
import tables
import nn_util
from keras import models, layers, optimizers, callbacks
from keras import backend as K
# from scipy import stats
# from experiment_list import experiments
from sklearn.model_selection import GroupKFold

# ZAK add utilities used for correlations
import pandas_utils


# from sklearn.model_selection import LeaveOneGroupOut
# import imblearn
# from imblearn.over_sampling import ADASYN
# from imblearn.over_sampling import SMOTE


# Logger class:
# http://stackoverflow.com/questions/20898212/how-to-automatically-direct-print-statements-outputs-to-a-file-in-python
class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


def append_data(out_data, outfile_str):
    outfile = open(outfile_str, 'a', newline='')
    writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(out_data)
    outfile.close()
    # for row in out_data:
    #     del row
    # del out_data
    return


def process_report(start_time, process_id):
    cur_full_mem = process_id.memory_full_info()
    print('Unique process memory (uss) MB: ' + str(cur_full_mem.uss / 1024 / 1024))
    print('Elapsed time: ' + str(datetime.datetime.utcnow() - start_time))
    return

# ZAK add for LSTM + FF model
def lstm_dense(lstm_in_cols, lstm_num_neurons, sequence_len,
               dense_in_cols, dense_num_neurons, loss_func,
               out_columns, out_activation=None, model_def=None,
               metrics = 'mse'
               ):
    # create inputs
    input_lstm = layers.Input(shape=(sequence_len, len(lstm_in_cols),))
    input_dense = layers.Input(shape=(len(dense_in_cols),))
    
    # build lstm section 
    x_lstm = layers.LSTM(lstm_num_neurons, name='lstm_1')(input_lstm)
    x_lstm = layers.LeakyReLU(name='lstm_1_LR')(x_lstm)
    x_lstm = layers.BatchNormalization(name='lstm_1_batch')(x_lstm)
    
    # build dense section
    x_dense = layers.Dense(dense_num_neurons, name='dense_1')(input_dense)
    x_dense = layers.LeakyReLU(name='dense_1_LR')(x_dense)
    x_dense = layers.BatchNormalization(name='dense_1_batch')(x_dense)

    # concatenate outputs
    x = layers.concatenate([x_lstm, x_dense])
    # final dense layers
    x = layers.Dense(dense_num_neurons, name='dense_2')(x)
    x = layers.LeakyReLU(name='merge_2_LR')(x)
    x = layers.BatchNormalization(name='dense_2_batch')(x)
    output = layers.Dense(len(out_columns), name='dense_2_out')(x)

    # create final model
    final_model = models.Model(inputs=[input_lstm, input_dense], outputs=output)
    final_model.compile(loss=loss_func, optimizer=optimizers.Nadam(),
                        metrics=[metrics])
    return final_model

# def run_recurrent(model_type, in_columns, out_columns, orig_in_cols, orig_out_cols, demo_cols,
#                   num_layers, num_neurons, size_batches, num_epochs, stop_patience, dropout_perc,
#                   model_name, data_list, start_time, sequence_len, rec_structure, activation,
#                   out_activation, loss_func, func_name, func_dict):

# ZAK change to put inside of pool loop, add iter_num and change data_list to
# data_dict
def run_recurrent(iter_num, model_type, in_columns, out_columns, orig_in_cols,
                  orig_out_cols, demo_cols,
                  num_layers, num_neurons, size_batches, num_epochs,
                  stop_patience, dropout_perc,
                  model_name, data_list, start_time, sequence_len,
                  rec_structure, activation,
                  out_activation, loss_func, func_name, func_dict):
    # grab current timestamp in UTC timezone -- use this for individual model outputs
    cur_time = start_time.strftime("%m%d%Y-%H%M%S")
    model_time = datetime.datetime.utcnow().strftime("%m%d%Y-%H%M%S")

    logging_dir = 'output/{}_{}/logging/'.format(model_type, cur_time)
    data_dump_dir = 'output/{}_{}/dump/'.format(model_type, cur_time)
    models_dir = data_dump_dir + 'models/'
    epochs_dir = data_dump_dir + 'epochs/'
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(data_dump_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(epochs_dir, exist_ok=True)

    debugging_output_file = logging_dir + "{}_DEBUG-OUTPUT_{}.txt".format(model_type, cur_time)
    debugging_error_file = logging_dir + "{}_DEBUG-ERROR_{}.txt".format(model_type, cur_time)
    test_preds_file = 'output/{}_{}/model_test_preds.csv'.format(model_type, cur_time)
    train_val_preds_file = 'output/{}_{}/model_train_val_preds.csv'.format(model_type, cur_time)
    test_inputs_file = 'output/{}_{}/model_test_inputs.csv'.format(model_type, cur_time)
    train_val_inputs_file = 'output/{}_{}/model_train_val_inputs.csv'.format(model_type, cur_time)
    model_overview_file = models_dir + 'models_overview.csv'

    # ZAK add output file for misc stats and other data per fold
    model_preds_file_misc = 'output/{}_{}/model_preds_misc.csv'.format(
        model_type,
        cur_time)

    # this sets up output to both stdout and a logging file
    # adapted from
    # http://stackoverflow.com/questions/20898212/how-to-automatically-direct-print-statements-outputs-to-a-file-in-python
    logging_output = open(debugging_output_file, 'a')
    logging_error = open(debugging_error_file, 'a')
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Logger(logging_output, sys.stdout)
    sys.stderr = Logger(logging_error, sys.stderr)

    current_process = psutil.Process(os.getpid())

    time_col = "sequence_id"

    print("---MODEL PARAMETERS---")
    print("model_type: {}".format(model_type))
    print("in_columns: {}".format(in_columns))
    print("out_columns: {}".format(out_columns))
    print("orig_in: {}".format(orig_in_cols))
    print("orig_out: {}".format(orig_out_cols))
    print("num_layers: {}".format(num_layers))
    print("num_neurons: {}".format(num_neurons))
    print("size_batches: {}".format(size_batches))
    print("num_epochs: {}".format(num_epochs))
    print("stop_patience: {}".format(stop_patience))
    print("dropout_perc: {}".format(dropout_perc))
    print("model_name: {}".format(model_name))
    print("sequence_len: {}".format(sequence_len))
    print("rec_structure: {}".format(rec_structure))
    print("activation: {}".format(activation))
    print("out_activation: {}".format(out_activation))
    print("loss_func: {}".format(loss_func))
    print("----------------------")

    # ZAK add list and dictionaries to hold accuracy scores and correlations
    y_test_stack = []
    test_preds_stack = []
    metrics = func_dict['metrics']

    corr_list = ['spearman', 'kendall', 'pearson']
    if metrics == 'mse':
        pred_list = ['train', 'val', 'test', 'train_rnd', 'test_rnd', 'val_rnd']
    else:
        pred_list = ['train', 'val', 'test']
    
    scores_dict = {}
    corr_dict = {}
    for pred in pred_list:
        scores_dict[pred] = []
        corr_dict[pred] = {}
        for corr in corr_list:
            corr_dict[pred][corr] = []
            

    # ZAK part of major change for subsets and pooling
    for data_dict in data_list:
        set_name = data_dict['set_name']
        fold = data_dict['k']
        pid_col = data_dict['primary_col']
        secondary_col = data_dict['secondary_col']

        iter_start_time = datetime.datetime.utcnow()

        print('Memory before training set {} type {} fold {} model {}'.format(set_name, model_type, fold, iter_num))

        process_report(start_time, current_process)

        train_data = data_dict['train'].copy()
        val_data = data_dict['val'].copy()
        test_data = data_dict['test'].copy()

        # prepare training data
        x_train, y_i_train = nn_util.make_sequences(
            train_data, in_columns,
            sequence_len=sequence_len,
            primary_id_col=pid_col,
            secondary_id_col=secondary_col,
            min_valid_prop=1)
        y_train = train_data[out_columns].iloc[y_i_train].values
        # ZAK get instance x values for LSTM + FF
        # x_train2 = train_data[demo_cols].iloc[y_i_train].values

        # # ZAK original way I was getting second input
        # # get training data for demographics columns
        # x_train2a, y_i = nn_util.make_instances(
        #             train_data,
        #             demo_cols,
        #             primary_id_col=pid_col,
        #             secondary_id_col=secondary_col,)
        # # #############################################################

        # prepare validation data
        x_val, y_i_val = nn_util.make_sequences(
            val_data, in_columns,
            sequence_len=sequence_len,
            primary_id_col=pid_col,
            secondary_id_col=secondary_col,
            min_valid_prop=1)
        y_val = val_data[out_columns].iloc[y_i_val].values

        # ZAK get validation data for demographics columns (LSTM + FF)
        # x_val2 = val_data[demo_cols].iloc[y_i_val].values

        # prepare test data
        x_test, y_i_test = nn_util.make_sequences(
            test_data, in_columns,
            sequence_len=sequence_len,
            primary_id_col=pid_col,
            secondary_id_col=secondary_col,
            min_valid_prop=1)
        y_test = test_data[out_columns].iloc[y_i_test].values

        # ZAK get test data for demographics columns (LSTM + FF)
        # x_test2 = test_data[demo_cols].iloc[y_i_test].values

        train_pid = train_data[pid_col].iloc[y_i_train].values
        train_time = train_data[time_col].iloc[y_i_train].values
        val_pid = val_data[pid_col].iloc[y_i_val].values
        val_time = val_data[time_col].iloc[y_i_val].values
        test_pid = test_data[pid_col].iloc[y_i_test].values
        test_time = test_data[time_col].iloc[y_i_test].values

        model_train_x = train_data[in_columns].iloc[y_i_train].values
        model_val_x = val_data[in_columns].iloc[y_i_val].values
        model_test_x = test_data[in_columns].iloc[y_i_test].values

        orig_train_x = train_data[orig_in_cols].iloc[y_i_train].values
        orig_val_x = val_data[orig_in_cols].iloc[y_i_val].values
        orig_test_x = test_data[orig_in_cols].iloc[y_i_test].values

        orig_train_y = train_data[orig_out_cols].iloc[y_i_train].values
        orig_val_y = val_data[orig_out_cols].iloc[y_i_val].values
        orig_test_y = test_data[orig_out_cols].iloc[y_i_test].values

        model_start_time = datetime.datetime.utcnow()

        print('Training set {} type {} fold {} model {} ({})'.format(
            set_name, model_type, fold, iter_num, model_time))
        print('Train on batches of size {} for {} epochs'.format(size_batches, num_epochs))
        print('Train len: {}  Val len: {}'.format(len(x_train), len(x_val)))
        print('Data prep elapsed time: ' + str(model_start_time - iter_start_time))
        process_report(start_time, current_process)

        # LSTM MODEL BEGIN
        model = models.Sequential()

        model.add(layers.LSTM(num_neurons, activation=activation, input_shape=(sequence_len, len(in_columns),)))
        model.add(layers.LeakyReLU())
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(len(out_columns), activation=out_activation))

        optimizer = optimizers.Nadam()
        model.compile(loss=loss_func, optimizer=optimizer)
        # LSTM MODEL END

        # ZAK new way to instantiate and compile model for LSTM + FF
        # model = func_name(**func_dict)

        os.makedirs(models_dir + "{}_{}_{}_{}/".format(model_time, set_name, fold, iter_num), exist_ok=True)
        better_model_file = models_dir + "{}_{}_{}_{}/".format(model_time, set_name, fold,
                                                               iter_num) + "{val_loss:.7f}_{epoch:04d}.hdf5"
        best_weights_file = models_dir + "{}_{}_{}_{}/best_weights.hdf5".format(model_time, set_name, fold, iter_num)

        # EarlyStopping checks whether model improves across epochs
        #   -- if not satisfied for patience # of epochs, then stop training early
        # val_mon = callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience)
        tf_csv_logger = callbacks.CSVLogger(
            epochs_dir + "c={}_m={}_f={}_i={}_{}".format(set_name, model_name, fold, iter_num, model_time) + ".txt")
        model_improve = callbacks.ModelCheckpoint(better_model_file)
        best_model_weights = callbacks.ModelCheckpoint(best_weights_file, save_best_only=True, save_weights_only=True)
        model_history = model.fit(x=x_train, y=y_train,
                                  verbose=0,  # 2 = one log line per epoch, 0 = none, 1 = progress bar
                                  batch_size=size_batches,
                                  # Max number of training epochs, if early stopping doesn't happen
                                  epochs=num_epochs,
                                  validation_data=[x_val, y_val],
                                  callbacks=[tf_csv_logger, model_improve, best_model_weights])
                                  # callbacks=[tf_csv_logger, model_improve, best_model_weights, val_mon])
        # # ZAK modify for two inputs for (LSTM + FF)
        # model_history = model.fit([x_train, x_train2], y_train,
        #                           verbose=0,
        #                           # 2 = one log line per epoch, 0 = none, 1 = progress bar
        #                           batch_size=size_batches,
        #                           # Max number of training epochs, if early stopping doesn't happen
        #                           epochs=num_epochs,
        #                           validation_data=[[x_val, x_val2], y_val],
        #                           callbacks=[tf_csv_logger, model_improve,
        #                                      best_model_weights])
        # callbacks=[tf_csv_logger, model_improve, best_model_weights, val_mon])

        model.summary()

        model_end_time = datetime.datetime.utcnow()

        print('Memory after training set {} type {} fold {} model {}'.format(set_name, model_type, fold, iter_num))
        print('Model training elapsed time: ' + str(model_end_time - model_start_time))
        process_report(start_time, current_process)

        # LOAD THE BEST MODEL WEIGHTS
        #        best_model = models.load_model(best_model_file)
        # need to wait a second to make sure that the model weights were output to file system
        sleep(1)
        print("Model weights before loading best: " + str(model.get_weights()[0][0][:4]))
        model.load_weights(best_weights_file)
        print("Model weights after loading best: " + str(model.get_weights()[0][0][:4]))

        # SAVE MODEL
        model_filename = str(iter_num) + "_" + str(fold) + '_' + set_name + '_' + model_type + '_' + str(num_neurons) + '_' + model_time + '.h5'
        model_overview_rows = []
        if not os.path.exists(model_overview_file):
            model_overview_rows.append(
                ["model_type", "model_name", "num_neurons", "set_name", "fold", "in_columns", "out_columns",
                 "model_filename", "model_time"])
        model_overview_rows.append(
            [model_type, model_name, num_neurons, set_name, fold, in_columns, out_columns, model_filename, model_time])
        model.save(models_dir + model_filename)
        append_data(model_overview_rows, model_overview_file)

        # PREDICT
        # ZAK  needs to be updated to handle single or multi-input
        # currently only dual input
        # ZAK add for (LSTM + FF)
        overfit_preds = model.predict(x_train)
        val_preds = model.predict(x_val)
        test_preds = model.predict(x_test)

        #######################################################
        # ZAK add to compute correlations across all folds at end of loop
        y_test_stack.append(y_test)
        test_preds_stack.append(test_preds)

        # ZAK get model evaluations
        # ZAK modified for LSTM + FF
        scores_test = model.evaluate(x_test, y_test, verbose=0)
        scores_train = model.evaluate(x_train, y_train, verbose=0)
        scores_val = model.evaluate(x_val, y_val, verbose=0)

        # # PREDICT
        # # ZAK  needs to be updated to handle single or multi-input
        # # currently only dual input
        # # ZAK add for (LSTM + FF)
        # overfit_preds = model.predict([x_train, x_train2])
        # val_preds = model.predict([x_val, x_val2])
        # test_preds = model.predict([x_test, x_test2])
        #
        # #######################################################
        # # ZAK add to compute correlations across all folds at end of loop
        # y_test_stack.append(y_test)
        # test_preds_stack.append(test_preds)
        #
        # # ZAK get model evaluations
        # # ZAK modified for LSTM + FF
        # scores_test = model.evaluate([x_test, x_test2], y_test)
        # scores_train = model.evaluate([x_train, x_train2],y_train)
        # scores_val = model.evaluate([x_val, x_val2], y_val)

        # ZAK modified for scores  to use dictionary
        # print('########################################################')
        print("Calculating model performance")
        if metrics == 'accuracy':
            # print("%s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))
            scores_dict['test'].append(scores_test * 100)
            scores_dict['train'].append(scores_train * 100)
            scores_dict['val'].append(scores_val * 100)
        elif metrics == 'mse':
            # print('%s: %.3f  mse' % (model.metrics_names[1], scores_test[1]))
            scores_dict['test'].append(scores_test)
            scores_dict['test_rnd'].append(
                pandas_utils.accuracy_from_round(test_preds, y_test))
            scores_dict['train'].append(scores_train)
            scores_dict['train_rnd'].append(
                pandas_utils.accuracy_from_round(overfit_preds, y_train))
            scores_dict['val'].append(scores_val)
            scores_dict['val_rnd'].append(
                pandas_utils.accuracy_from_round(val_preds, y_val))
        # print('########################################################')

        # # ZAK modified for scores  to use dictionary
        # # print('########################################################')
        # print("Calculating model performance")
        # if metrics == 'accuracy':
        #     # print("%s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))
        #     scores_dict['test'].append(scores_test[1] * 100)
        #     scores_dict['train'].append(scores_train[1] * 100)
        #     scores_dict['val'].append(scores_val[1] * 100)
        # elif metrics == 'mse':
        #     # print('%s: %.3f  mse' % (model.metrics_names[1], scores_test[1]))
        #     scores_dict['test'].append(scores_test[1])
        #     scores_dict['test_rnd'].append(
        #         pandas_utils.accuracy_from_round(test_preds, y_test))
        #     scores_dict['train'].append(scores_train[1])
        #     scores_dict['train_rnd'].append(
        #         pandas_utils.accuracy_from_round(overfit_preds, y_train))
        #     scores_dict['val'].append(scores_val[1])
        #     scores_dict['val_rnd'].append(
        #         pandas_utils.accuracy_from_round(val_preds, y_val))
        #     # print('########################################################')

        # ZAK modified to use dictionary for correlations
        tmp = pandas_utils.get_corr_from_predictions(
            test_preds, y_test, metrics=metrics, corr=corr_list)
        for key in corr_dict['test']:
            corr_dict['test'][key].append(tmp[key])
        if metrics == 'mse':
            tmp = pandas_utils.get_corr_from_predictions(
                test_preds, y_test, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['test_rnd']:
                corr_dict['test_rnd'][key].append(tmp[key])
            
        tmp = pandas_utils.get_corr_from_predictions(
            val_preds, y_val, metrics=metrics, corr=corr_list)
        for key in corr_dict['val']:
            corr_dict['val'][key].append(tmp[key])       
        if metrics == 'mse':            
            tmp = pandas_utils.get_corr_from_predictions(
                val_preds, y_val, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['val_rnd']:
                corr_dict['val_rnd'][key].append(tmp[key])
            
        tmp = pandas_utils.get_corr_from_predictions(
            overfit_preds, y_train, metrics=metrics, corr=corr_list)
        for key in corr_dict['train']:
            corr_dict['train'][key].append(tmp[key])       
        if metrics == 'mse':            
            tmp = pandas_utils.get_corr_from_predictions(
                overfit_preds, y_train, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['train_rnd']:
                corr_dict['train_rnd'][key].append(tmp[key])

        #############################################################################

        model_train_val_output_rows = []
        model_test_output_rows = []
        model_train_val_input_rows = []
        model_test_input_rows = []

        # query input length
        in_len = len(x_train[0])
        # query input width
        in_width = len(x_train[0][0])
        # query output length
        out_len = len(y_train[0])

        proc_input_header = []
        orig_input_header = []
        pred_out_header = []
        proc_out_header = []
        orig_out_header = []

        input_features = []
        out_targets = []

        in_feature_names = []
        out_target_names = []

        # generate headers from data dimensions
        for in_dim in range(in_width):
            input_features = input_features + ["in_feature_" + str(in_dim)]
            proc_input_header = proc_input_header + ["proc_input_" + str(in_dim)]
            orig_input_header = orig_input_header + ["orig_input_" + str(in_dim)]
            in_feature_names = in_feature_names + [in_columns[in_dim]]
        for out_dim in range(out_len):
            pred_out_header = pred_out_header + ["pred_out_" + str(out_dim)]
            proc_out_header = proc_out_header + ["proc_out_" + str(out_dim)]
            out_targets = out_targets + ["out_target_" + str(out_dim)]
            out_target_names = out_target_names + [out_columns[out_dim]]
            orig_out_header = orig_out_header + ["orig_out_" + str(out_dim)]

        # prepare prediction files
        if not os.path.exists(train_val_preds_file):
            model_train_val_output_rows.append(
                ["model_type", "set_name", "time", "model_name", "student_id", "fold_type", "fold_num",
                 "data_index"] + input_features + proc_input_header + orig_input_header + out_targets + pred_out_header + proc_out_header + orig_out_header)
        if not os.path.exists(test_preds_file):
            model_test_output_rows.append(
                ["model_type", "set_name", "time", "model_name", "student_id", "fold_type", "fold_num",
                 "data_index"] + input_features + proc_input_header + orig_input_header + out_targets + pred_out_header + proc_out_header + orig_out_header)

        # prepare inputs files
        if not os.path.exists(train_val_inputs_file):
            model_train_val_input_rows.append(
                ["model_type", "set_name", "time", "model_name", "student_id", "fold_type", "fold_num", "data_index",
                 "seq_index"] + input_features + proc_input_header)
        if not os.path.exists(test_inputs_file):
            model_test_input_rows.append(
                ["model_type", "set_name", "time", "model_name", "student_id", "fold_type", "fold_num", "data_index",
                 "seq_index"] + input_features + proc_input_header)

        print("Preparing training predictions")

        for train_input, pred, proc_out, index, pid, time, orig_out_vals, proc_in, orig_in_vals \
                in zip(x_train,
                       overfit_preds,
                       y_train,
                       y_i_train,
                       train_pid,
                       train_time,
                       orig_train_y,
                       model_train_x,
                       orig_train_x):
            model_proc_in_vals = []
            model_orig_in_vals = []
            model_pred_out_vals = []
            model_proc_out_vals = []
            model_orig_out_vals = []
            for in_dim in range(in_width):
                model_proc_in_vals = model_proc_in_vals + [proc_in[in_dim]]
                model_orig_in_vals = model_orig_in_vals + [orig_in_vals[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_proc_out_vals = model_proc_out_vals + [proc_out[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig_out_vals[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "train", fold,
                          index] + in_feature_names + model_proc_in_vals + model_orig_in_vals + out_target_names + model_pred_out_vals + model_proc_out_vals + model_orig_out_vals  # change this for each fold type
            for seq_index in range(in_len):
                model_seq_inputs = []
                cur_input = train_input[seq_index]  # change this for each fold type
                for in_dim in range(in_width):
                    model_seq_inputs = model_seq_inputs + [cur_input[in_dim]]
                input_row = [model_type, set_name, time, model_name, pid, "train", fold, index,
                             seq_index] + in_feature_names + model_seq_inputs  # change this for each fold type
                model_train_val_input_rows.append(input_row)  # change this for each fold type
            model_train_val_output_rows.append(output_row)  # change this for each fold type

        try:
            print("Preparing validation predictions")
        except:
            pass
        for val_input, pred, proc_out, index, pid, time, orig_out_vals, proc_in, orig_in_vals \
                in zip(x_val, val_preds,
                       y_val, y_i_val,
                       val_pid,
                       val_time,
                       orig_val_y,
                       model_val_x,
                       orig_val_x):

            model_proc_in_vals = []
            model_orig_in_vals = []
            model_pred_out_vals = []
            model_proc_out_vals = []
            model_orig_out_vals = []
            for in_dim in range(in_width):
                model_proc_in_vals = model_proc_in_vals + [proc_in[in_dim]]
                model_orig_in_vals = model_orig_in_vals + [orig_in_vals[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_proc_out_vals = model_proc_out_vals + [proc_out[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig_out_vals[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "val", fold,
                          index] + in_feature_names + model_proc_in_vals + model_orig_in_vals + out_target_names + model_pred_out_vals + model_proc_out_vals + model_orig_out_vals  # change this for each fold type
            for seq_index in range(in_len):
                model_seq_inputs = []
                cur_input = val_input[seq_index]  # change this for each fold type
                for in_dim in range(in_width):
                    model_seq_inputs = model_seq_inputs + [cur_input[in_dim]]
                input_row = [model_type, set_name, time, model_name, pid, "val", fold, index,
                             seq_index] + in_feature_names + model_seq_inputs  # change this for each fold type
                model_train_val_input_rows.append(input_row)  # change this for each fold type
            model_train_val_output_rows.append(output_row)  # change this for each fold type

        try:

            print("Preparing testing predictions")
        except:
            pass
        for test_input, pred, proc_out, index, pid, time, orig_out_vals, proc_in, orig_in_vals \
                in zip(x_test,
                       test_preds,
                       y_test,
                       y_i_test,
                       test_pid,
                       test_time,
                       orig_test_y,
                       model_test_x,
                       orig_test_x):

            model_proc_in_vals = []
            model_orig_in_vals = []
            model_pred_out_vals = []
            model_proc_out_vals = []
            model_orig_out_vals = []
            for in_dim in range(in_width):
                model_proc_in_vals = model_proc_in_vals + [proc_in[in_dim]]
                model_orig_in_vals = model_orig_in_vals + [orig_in_vals[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_proc_out_vals = model_proc_out_vals + [proc_out[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig_out_vals[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "test", fold,
                          index] + in_feature_names + model_proc_in_vals + model_orig_in_vals + out_target_names + model_pred_out_vals + model_proc_out_vals + model_orig_out_vals  # change this for each fold type
            for seq_index in range(in_len):
                model_seq_inputs = []
                cur_input = test_input[seq_index]  # change this for each fold type
                for in_dim in range(in_width):
                    model_seq_inputs = model_seq_inputs + [cur_input[in_dim]]
                input_row = [model_type, set_name, time, model_name, pid, "test", fold, index,
                             seq_index] + in_feature_names + model_seq_inputs  # change this for each fold type
                model_test_input_rows.append(input_row)  # change this for each fold type
            model_test_output_rows.append(output_row)  # change this for each fold type

        append_data(model_train_val_output_rows, train_val_preds_file)
        append_data(model_test_output_rows, test_preds_file)
        append_data(model_train_val_input_rows, train_val_inputs_file)
        append_data(model_test_input_rows, test_inputs_file)

        #############################################################
        # ZAK add for output of correlations and other data
        # ADD add per fold for training, validation
        # ZAK modified to use dictionaries for metrics and correlations
        model_misc_rows = []
        if not os.path.exists(model_preds_file_misc):
            model_misc_rows.append(
                ["model_type", "set_name", "time", "model_name", "fold_type",
                 "fold_num", 'metrics_type','metrics_length', 'metrics_value',
                 'pearson', 'kendall', 'spearman'])

        for pred in pred_list:
            fold_type = pred.split('_')[0]
            if 'rnd' in pred:
                t_metrics = 'round'
            else:
                t_metrics = metrics
            model_misc_rows.append(
                [model_type, set_name, time, model_name, fold_type, fold, t_metrics,
                 y_test.shape[0], scores_dict[pred][fold], corr_dict[pred]['pearson'][fold],
                 corr_dict[pred]['kendall'][fold], corr_dict[pred]['spearman'][fold]])

        append_data(model_misc_rows, model_preds_file_misc)
        #############################################################

        iter_end_time = datetime.datetime.utcnow()

        print("Finished output of fold predictions")
        print('Model prediction and output elapsed time: ' + str(iter_end_time - model_end_time))
        print('Total elapsed time: ' + str(iter_end_time - start_time))

        K.clear_session()

    # ZAK add to compute correlations across all folds at end of loop
    # need this before all the mean/std to get total # samples
    y_test_stack = np.vstack(y_test_stack)
    test_preds_stack = np.vstack(test_preds_stack)

    # ZAK add method to use dictionaries for mean and std of
    # correlations and metrics
    stat_list = ['mean', 'std']

    corr_stats = {}
    for pred in pred_list:
        corr_stats[pred] = {}
        for corr in corr_list:
            corr_stats[pred][corr] = {}

    for pred in pred_list:
        for corr in corr_list:
            for stat in stat_list:
                if stat == 'mean':
                    corr_stats[pred][corr][stat] = np.mean(
                        np.array(corr_dict[pred][corr]))
                elif stat == 'std':
                    corr_stats[pred][corr][stat] = np.std(
                        np.array(corr_dict[pred][corr]))
                else:
                    print('run_reccurrent(): bad stat option, filling zeros')
                    corr_stats[pred][corr][stat] = 0.0

    metrics_stats = {}
    for pred in pred_list:
        metrics_stats[pred] = {}
    for pred in pred_list:
        for stat in stat_list:
            if stat == 'mean':
                metrics_stats[pred][stat] = np.mean(np.array(scores_dict[pred]))
            elif stat == 'std':
                metrics_stats[pred][stat] = np.std(np.array(scores_dict[pred]))
            else:
                print('run_reccurrent(): bad stat option, filling zeros')
                corr_stats[pred][corr][stat] = 0.0

    # ZAK write mean and std's of correlations and metrics to output file
    model_preds_file_stats = 'output/{}_{}/model_preds_stats.csv'.format(
        model_type,
        cur_time)
    model_stat_rows = []
    if not os.path.exists(model_preds_file_stats):
        model_stat_rows.append(
            ["model_type", 'set_name', "time", "model_name", "fold_type",
             "num_folds", 'metrics_type', 'metrics_length', 'stats_type',
             'metrics_value', 'pearson', 'kendall', 'spearman'])
    for pred in pred_list:
        fold_type = pred.split('_')[0]
        for stat in stat_list:
            if 'rnd' in pred:
                t_metrics = 'round'
            else:
                t_metrics = metrics
            model_stat_rows.append(
                [model_type, set_name, time, model_name, fold_type,
                 len(scores_dict[pred]),
                 t_metrics, test_preds_stack.shape[0], stat,
                 metrics_stats[pred][stat],
                 corr_stats[pred]['pearson'][stat], 
                 corr_stats[pred]['kendall'][stat], 
                 corr_stats[pred]['spearman'][stat]])

    append_data(model_stat_rows, model_preds_file_stats)

    # ZAK modified to use dictionary
    print('########################################')
    if metrics == 'accuracy':
        print("%s %.2f%% (+/- %.2f%%)" % ('Total ' + metrics + ': ',
                                          metrics_stats['test']['mean'],
                                          metrics_stats['test']['std']))
    elif metrics == 'mse':
        print("%s %.3f (+/- %.3f)" % ('Total ' + metrics + ': ',
                                      metrics_stats['test']['mean'],
                                      metrics_stats['test']['std']))
    print('#########################################')

    # ZAK add to compute correlations across all folds at end of loop
    # get correlations across all folds
    corr_dict = pandas_utils.get_corr_from_predictions(test_preds_stack,
                y_test_stack, metrics=metrics,
                corr=['spearman', 'kendall', 'pearson'])
    # write correlations to output file
    model_preds_file_corr = 'output/{}_{}/model_preds_corr.csv'.format(
        model_type,
        cur_time)
    model_corr_rows = []
    if not os.path.exists(model_preds_file_corr):
        model_corr_rows.append(
            ["model_type", 'set_name',"time", "model_name", "fold_type",
             "num_folds", 'metrics_type', 'metrics_length','metrics_test_mean',
             'metrics_test_std', 'pearson', 'kendall', 'spearman'])
    model_corr_rows.append(
        [model_type, set_name, time, model_name, "test",
         len(scores_dict['test']),
         metrics, test_preds_stack.shape[0], metrics_stats['test']['mean'],
         metrics_stats['test']['std'], corr_dict['pearson'], corr_dict['kendall'],
         corr_dict['spearman']])

    if metrics == 'mse':  # write one more row for round version
        corr_dict_round = pandas_utils.get_corr_from_predictions(
            test_preds_stack, y_test_stack, metrics=metrics,
            corr=['spearman', 'kendall', 'pearson'], round=True)
        model_corr_rows.append(
            [model_type, set_name, time, model_name, "test",
             len(scores_dict['test']),
             'round', test_preds_stack.shape[0], metrics_stats['test_rnd']['mean'],
             metrics_stats['test_rnd']['std'], corr_dict_round['pearson'],
             corr_dict_round['kendall'], corr_dict_round['spearman']])

    append_data(model_corr_rows, model_preds_file_corr)

    print('**************************************************')
    print('FINAL CORRELATIONS = ', corr_dict)
    print('**************************************************')

    sys.stdout = stdout
    sys.stderr = stderr
    logging_output.close()
    logging_error.close()

    return

#    1) missing data are zeroed in joint model (so that we can train on all data)
#    2) model input/output has additional column iteration to account for joint features
def run_joint_ff(iter_num, model_type, in_columns, out_columns,
                 orig_in_cols, orig_out_cols,
                 num_layers, num_neurons, size_batches, num_epochs,
                 stop_patience, dropout_perc, model_name, data_list,
                 start_time, activation, out_activation, loss_func, func_dict):
    # grab current timestamp in UTC timezone -- use this for individual model outputs
    cur_time = start_time.strftime("%m%d%Y-%H%M%S")
    model_time = datetime.datetime.utcnow().strftime("%m%d%Y-%H%M%S")

    logging_dir = 'output/{}_{}/logging/'.format(model_type, cur_time)
    data_dump_dir = 'output/{}_{}/dump/'.format(model_type, cur_time)
    models_dir = data_dump_dir + 'models/'
    epochs_dir = data_dump_dir + 'epochs/'
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(data_dump_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(epochs_dir, exist_ok=True)

    debugging_output_file = logging_dir + "{}_DEBUG-OUTPUT_{}.txt".format(model_type, cur_time)
    debugging_error_file = logging_dir + "{}_DEBUG-ERROR_{}.txt".format(model_type, cur_time)
    test_preds_file = 'output/{}_{}/model_test_preds.csv'.format(model_type, cur_time)
    train_val_preds_file = 'output/{}_{}/model_train_val_preds.csv'.format(model_type, cur_time)
    model_overview_file = models_dir + 'models_overview.csv'

    # ZAK add output file for misc stats and other data per fold
    model_preds_file_misc = 'output/{}_{}/model_preds_misc.csv'.format(
        model_type,
        cur_time)

    # this sets up output to both stdout and a logging file
    # adapted from:
    #   http://stackoverflow.com/questions/20898212/how-to-automatically-direct-print-statements-outputs-to-a-file-in-python
    logging_output = open(debugging_output_file, 'a')
    logging_error = open(debugging_error_file, 'a')
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Logger(logging_output, sys.stdout)
    sys.stderr = Logger(logging_error, sys.stderr)

    current_process = psutil.Process(os.getpid())

    print("---MODEL PARAMETERS---")
    print("model_type: {}".format(model_type))
    print("in_columns: {}".format(in_columns))
    print("out_columns: {}".format(out_columns))
    print("orig_in: {}".format(orig_in_cols))
    print("orig_out: {}".format(orig_out_cols))
    print("num_layers: {}".format(num_layers))
    print("num_neurons: {}".format(num_neurons))
    print("size_batches: {}".format(size_batches))
    print("num_epochs: {}".format(num_epochs))
    print("stop_patience: {}".format(stop_patience))
    print("dropout_perc: {}".format(dropout_perc))
    print("model_name: {}".format(model_name))
    print("activation: {}".format(activation))
    print("out_activation: {}".format(out_activation))
    print("loss_func: {}".format(loss_func))
    print("----------------------")

    # ZAK add list and dictionaries to hold accuracy scores and correlations
    y_test_stack = []
    test_preds_stack = []
    metrics = func_dict['metrics']

    corr_list = ['spearman', 'kendall', 'pearson']
    if metrics == 'mse':
        pred_list = ['train', 'val', 'test', 'train_rnd', 'test_rnd', 'val_rnd']
    else:
        pred_list = ['train', 'val', 'test']

    scores_dict = {}
    corr_dict = {}
    for pred in pred_list:
        scores_dict[pred] = []
        corr_dict[pred] = {}
        for corr in corr_list:
            corr_dict[pred][corr] = []

    for data_dict in data_list:

        set_name = data_dict['set_name']
        fold = data_dict['k']
        primary_col = data_dict['primary_col']
        secondary_col = data_dict['secondary_col']

        iter_start_time = datetime.datetime.utcnow()

        print('Memory before training set {} type {} fold {} model {}'.format(set_name, model_type, fold, iter_num))
        process_report(start_time, current_process)

        train_data = data_dict['train'].copy()
        val_data = data_dict['val'].copy()
        test_data = data_dict['test'].copy()

        orig_train_x = train_data[orig_in_cols].values
        orig_val_x = val_data[orig_in_cols].values
        orig_test_x = test_data[orig_in_cols].values

        orig_train_y = train_data[orig_out_cols].values
        orig_val_y = val_data[orig_out_cols].values
        orig_test_y = test_data[orig_out_cols].values

        # NOTE: no missing data in VLL -- replacement not necessary
        # # Replace missing values in output with zeroes (i.e., mean value of training set)
        # #   This allows joint model training and prediction despite missing output values
        # #       -- however, we will correlate against original values
        # for col in out_columns:
        #     train_data[col] = train_data[col].fillna(value=0)
        #     val_data[col] = val_data[col].fillna(value=0)
        #     test_data[col] = test_data[col].fillna(value=0)
        #
        # # Fill missing values in input -- use zeroes
        # for col in in_columns:
        #     train_data[col] = train_data[col].fillna(value=0)
        #     val_data[col] = val_data[col].fillna(value=0)
        #     test_data[col] = test_data[col].fillna(value=0)

        x_train, y_train = train_data[in_columns].values, train_data[out_columns].values
        x_val, y_val = val_data[in_columns].values, val_data[out_columns].values
        x_test, y_test = test_data[in_columns].values, test_data[out_columns].values
        train_pid = train_data[primary_col].values
        val_pid = val_data[primary_col].values
        test_pid = test_data[primary_col].values
        train_time = train_data[secondary_col].values
        val_time = val_data[secondary_col].values
        test_time = test_data[secondary_col].values
        y_i_train = train_data.index.values
        y_i_val = val_data.index.values
        y_i_test = test_data.index.values

        model_start_time = datetime.datetime.utcnow()

        print('Training set {} type {} fold {} model {}'.format(set_name, model_type, fold, iter_num))
        print('Train on batches of size {} for {} epochs'.format(size_batches, num_epochs))
        print('Train len: {}  Val len: {}'.format(len(x_train), len(x_val)))
        print('Data prep elapsed time: ' + str(model_start_time - iter_start_time))
        process_report(start_time, current_process)

        model = models.Sequential()

        # input layer
        # model.add(layers.Dropout(dropout_perc, input_shape=(len(in_columns),)))

        model.add(layers.Dense(num_neurons, input_shape=(len(in_columns),)))
        model.add(layers.LeakyReLU())
        model.add(layers.BatchNormalization())

        # output layer
        # model.add(layers.Dropout(dropout_perc))
        model.add(layers.Dense(len(out_columns), activation=out_activation))

        # Adam with Nesterov momentum
        optimizer = optimizers.Nadam()
        model.compile(loss=loss_func, optimizer=optimizer)

        os.makedirs(models_dir + "{}_{}_{}_{}/".format(model_time, set_name, fold, iter_num), exist_ok=True)
        better_model_file = models_dir + "{}_{}_{}_{}/".format(model_time, set_name, fold,
                                                               iter_num) + "{val_loss:.7f}_{epoch:04d}.hdf5"
        best_weights_file = models_dir + "{}_{}_{}_{}/best_weights.hdf5".format(model_time, set_name, fold, iter_num)

        # EarlyStopping checks whether absolute improvement is above min_delta
        #   -- if not satisfied for patience # of epochs, then stop training early
        # val_mon = callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience)
        tf_csv_logger = callbacks.CSVLogger(
            epochs_dir + "c={}_m={}_f={}_i={}_{}".format(set_name, model_name, fold, iter_num, model_time) + ".txt")
        model_improve = callbacks.ModelCheckpoint(better_model_file)
        best_model_weights = callbacks.ModelCheckpoint(best_weights_file, save_best_only=True, save_weights_only=True)
        model_history = model.fit(x=x_train, y=y_train,
                                  verbose=0,  # 2 = one log line per epoch, 0 = none, 1 = progress bar
                                  batch_size=size_batches,
                                  epochs=num_epochs,  # Max number of training epochs if not stopped early
                                  validation_data=[x_val, y_val],
                                  callbacks=[tf_csv_logger, model_improve, best_model_weights])
                                  # callbacks=[tf_csv_logger, model_improve, best_model_weights, val_mon])

        model.summary()

        model_end_time = datetime.datetime.utcnow()

        print('Memory after training set {} type {} fold {} model {}'.format(set_name, model_type, fold, iter_num))
        print('Model training elapsed time: ' + str(model_end_time - model_start_time))
        process_report(start_time, current_process)

        # LOAD THE BEST MODEL WEIGHTS
        # wait a moment to ensure that model weights have written to file system
        sleep(1)
        print("Model weights before loading best: " + str(model.get_weights()[0][0][:4]))
        model.load_weights(best_weights_file)
        print("Model weights after loading best: " + str(model.get_weights()[0][0][:4]))

        # SAVE MODEL
        model_filename = str(iter_num) + "_" + str(fold) + "_" + set_name + "_" + model_type + '_' + str(num_neurons) + '_' + model_time + '.h5'
        model_overview_rows = []
        if not os.path.exists(model_overview_file):
            model_overview_rows.append(
                ["model_type", "model_name", "num_neurons", "set_name", "fold", "in_columns", "out_columns",
                 "model_filename", "model_time"])
        model_overview_rows.append(
            [model_type, model_name, num_neurons, set_name, fold, in_columns, out_columns, model_filename, model_time])
        model.save(models_dir + model_filename)
        append_data(model_overview_rows, model_overview_file)

        # PREDICT
        overfit_preds = model.predict(x_train)
        val_preds = model.predict(x_val)
        test_preds = model.predict(x_test)

        #######################################################
        # ZAK add to compute correlations across all folds at end of loop
        y_test_stack.append(y_test)
        test_preds_stack.append(test_preds)

        # ZAK get model evaluations
        # ZAK modified for LSTM + FF
        scores_test = model.evaluate(x_test, y_test, verbose=0)
        scores_train = model.evaluate(x_train, y_train, verbose=0)
        scores_val = model.evaluate(x_val, y_val, verbose=0)

        # ZAK modified for scores  to use dictionary
        # print('########################################################')
        print("Calculating model performance")
        if metrics == 'accuracy':
            # print("%s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))
            scores_dict['test'].append(scores_test * 100)
            scores_dict['train'].append(scores_train * 100)
            scores_dict['val'].append(scores_val * 100)
        elif metrics == 'mse':
            # print('%s: %.3f  mse' % (model.metrics_names[1], scores_test[1]))
            scores_dict['test'].append(scores_test)
            scores_dict['test_rnd'].append(
                pandas_utils.accuracy_from_round(test_preds, y_test))
            scores_dict['train'].append(scores_train)
            scores_dict['train_rnd'].append(
                pandas_utils.accuracy_from_round(overfit_preds, y_train))
            scores_dict['val'].append(scores_val)
            scores_dict['val_rnd'].append(
                pandas_utils.accuracy_from_round(val_preds, y_val))
            # print('########################################################')

        # ZAK modified to use dictionary for correlations
        tmp = pandas_utils.get_corr_from_predictions(
            test_preds, y_test, metrics=metrics, corr=corr_list)
        for key in corr_dict['test']:
            corr_dict['test'][key].append(tmp[key])
        if metrics == 'mse':
            tmp = pandas_utils.get_corr_from_predictions(
                test_preds, y_test, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['test_rnd']:
                corr_dict['test_rnd'][key].append(tmp[key])

        tmp = pandas_utils.get_corr_from_predictions(
            val_preds, y_val, metrics=metrics, corr=corr_list)
        for key in corr_dict['val']:
            corr_dict['val'][key].append(tmp[key])
        if metrics == 'mse':
            tmp = pandas_utils.get_corr_from_predictions(
                val_preds, y_val, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['val_rnd']:
                corr_dict['val_rnd'][key].append(tmp[key])

        tmp = pandas_utils.get_corr_from_predictions(
            overfit_preds, y_train, metrics=metrics, corr=corr_list)
        for key in corr_dict['train']:
            corr_dict['train'][key].append(tmp[key])
        if metrics == 'mse':
            tmp = pandas_utils.get_corr_from_predictions(
                overfit_preds, y_train, metrics=metrics, corr=corr_list, round=True)
            for key in corr_dict['train_rnd']:
                corr_dict['train_rnd'][key].append(tmp[key])

        #############################################################################

        # we use predict() to make model predictions
        #   -- predict_proba() is the same, but checks whether preds are within 0 to 1 range

        model_train_val_output_rows = []
        model_test_output_rows = []

        # query input length
        in_len = len(x_train[0])
        # query output length
        out_len = len(y_train[0])

        proc_input_vals = []
        orig_input_vals = []
        pred_out_vals = []
        orig_out_vals = []
        raw_out_vals = []
        cor_vals = []

        input_features = []
        out_targets = []

        in_feature_names = []
        out_target_names = []

        for in_dim in range(in_len):
            proc_input_vals = proc_input_vals + ["proc_input_" + str(in_dim)]
            orig_input_vals = orig_input_vals + ["orig_input_" + str(in_dim)]
            input_features = input_features + ["in_feature_" + str(in_dim)]
            in_feature_names = in_feature_names + [in_columns[in_dim]]
        for out_dim in range(out_len):
            pred_out_vals = pred_out_vals + ["pred_out_" + str(out_dim)]
            orig_out_vals = orig_out_vals + ["proc_out_" + str(out_dim)]
            cor_vals = cor_vals + ["cor_pred_orig_" + str(out_dim)]
            out_targets = out_targets + ["out_target_" + str(out_dim)]
            out_target_names = out_target_names + [out_columns[out_dim]]
            raw_out_vals = raw_out_vals + ["orig_out_" + str(out_dim)]

        # prepare prediction files
        if not os.path.exists(train_val_preds_file):
            model_train_val_output_rows.append(
                ["model_type", "set_name", "time", "model_name", "couple_id", "fold_type", "fold_num",
                 "data_index"] + input_features + proc_input_vals + orig_input_vals + out_targets + pred_out_vals + orig_out_vals + raw_out_vals)
        if not os.path.exists(test_preds_file):
            model_test_output_rows.append(
                ["model_type", "set_name", "time", "model_name", "couple_id", "fold_type", "fold_num",
                 "data_index"] + input_features + proc_input_vals + orig_input_vals + out_targets + pred_out_vals + orig_out_vals + raw_out_vals)

        print("Preparing training predictions")

        for train_input, pred, orig, index, pid, time, orig_input, orig_output \
                in itertools.zip_longest(x_train,
                                         overfit_preds,
                                         y_train,
                                         y_i_train,
                                         train_pid,
                                         train_time,
                                         orig_train_x,
                                         orig_train_y):
            model_proc_input_vals = []
            model_orig_input_vals = []
            model_pred_out_vals = []
            model_orig_out_vals = []
            model_raw_out_vals = []
            for in_dim in range(in_len):
                model_proc_input_vals = model_proc_input_vals + [train_input[in_dim]]
                model_orig_input_vals = model_orig_input_vals + [orig_input[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig[out_dim]]
                model_raw_out_vals = model_raw_out_vals + [orig_output[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "train", fold,
                          index] + in_feature_names + model_proc_input_vals + model_orig_input_vals + out_target_names + model_pred_out_vals + model_orig_out_vals + model_raw_out_vals
            model_train_val_output_rows.append(output_row)

        print("Preparing validation predictions")

        for val_input, pred, orig, index, pid, time, orig_input, orig_output \
                in itertools.zip_longest(x_val,
                                         val_preds,
                                         y_val,
                                         y_i_val,
                                         val_pid,
                                         val_time,
                                         orig_val_x,
                                         orig_val_y):

            model_proc_input_vals = []
            model_orig_input_vals = []
            model_pred_out_vals = []
            model_orig_out_vals = []
            model_raw_out_vals = []
            for in_dim in range(in_len):
                model_proc_input_vals = model_proc_input_vals + [val_input[in_dim]]
                model_orig_input_vals = model_orig_input_vals + [orig_input[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig[out_dim]]
                model_raw_out_vals = model_raw_out_vals + [orig_output[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "val", fold,
                          index] + in_feature_names + model_proc_input_vals + model_orig_input_vals + out_target_names + model_pred_out_vals + model_orig_out_vals + model_raw_out_vals
            model_train_val_output_rows.append(output_row)

        print("Preparing testing predictions")

        for test_input, pred, orig, index, pid, time, orig_input, orig_output \
                in itertools.zip_longest(x_test,
                                         test_preds,
                                         y_test,
                                         y_i_test,
                                         test_pid,
                                         test_time,
                                         orig_test_x,
                                         orig_test_y):

            model_proc_input_vals = []
            model_orig_input_vals = []
            model_pred_out_vals = []
            model_orig_out_vals = []
            model_raw_out_vals = []
            for in_dim in range(in_len):
                model_proc_input_vals = model_proc_input_vals + [test_input[in_dim]]
                model_orig_input_vals = model_orig_input_vals + [orig_input[in_dim]]
            for out_dim in range(out_len):
                model_pred_out_vals = model_pred_out_vals + [pred[out_dim]]
                model_orig_out_vals = model_orig_out_vals + [orig[out_dim]]
                model_raw_out_vals = model_raw_out_vals + [orig_output[out_dim]]
            output_row = [model_type, set_name, time, model_name, pid, "test", fold,
                          index] + in_feature_names + model_proc_input_vals + model_orig_input_vals + out_target_names + model_pred_out_vals + model_orig_out_vals + model_raw_out_vals
            model_test_output_rows.append(output_row)

        append_data(model_train_val_output_rows, train_val_preds_file)
        append_data(model_test_output_rows, test_preds_file)

        #############################################################
        # ZAK add for output of correlations and other data
        # ADD add per fold for training, validation
        # ZAK modified to use dictionaries for metrics and correlations
        model_misc_rows = []
        if not os.path.exists(model_preds_file_misc):
            model_misc_rows.append(
                ["model_type", "set_name", "time", "model_name", "fold_type",
                 "fold_num", 'metrics_type','metrics_length', 'metrics_value',
                 'pearson', 'kendall', 'spearman'])

        for pred in pred_list:
            fold_type = pred.split('_')[0]
            if 'rnd' in pred:
                t_metrics = 'round'
            else:
                t_metrics = metrics
            model_misc_rows.append(
                [model_type, set_name, time, model_name, fold_type, fold, t_metrics,
                 y_test.shape[0], scores_dict[pred][fold], corr_dict[pred]['pearson'][fold],
                 corr_dict[pred]['kendall'][fold], corr_dict[pred]['spearman'][fold]])

        append_data(model_misc_rows, model_preds_file_misc)
        #############################################################

        iter_end_time = datetime.datetime.utcnow()

        print("Finished output of fold predictions")
        print('Model prediction and output elapsed time: ' + str(iter_end_time - model_end_time))
        print('Total elapsed time: ' + str(iter_end_time - start_time))

        K.clear_session()

    # ZAK add to compute correlations across all folds at end of loop
    # need this before all the mean/std to get total # samples
    y_test_stack = np.vstack(y_test_stack)
    test_preds_stack = np.vstack(test_preds_stack)

    # ZAK add method to use dictionaries for mean and std of
    # correlations and metrics
    stat_list = ['mean', 'std']

    corr_stats = {}
    for pred in pred_list:
        corr_stats[pred] = {}
        for corr in corr_list:
            corr_stats[pred][corr] = {}

    for pred in pred_list:
        for corr in corr_list:
            for stat in stat_list:
                if stat == 'mean':
                    corr_stats[pred][corr][stat] = np.mean(
                        np.array(corr_dict[pred][corr]))
                elif stat == 'std':
                    corr_stats[pred][corr][stat] = np.std(
                        np.array(corr_dict[pred][corr]))
                else:
                    print('run_reccurrent(): bad stat option, filling zeros')
                    corr_stats[pred][corr][stat] = 0.0

    metrics_stats = {}
    for pred in pred_list:
        metrics_stats[pred] = {}
    for pred in pred_list:
        for stat in stat_list:
            if stat == 'mean':
                metrics_stats[pred][stat] = np.mean(np.array(scores_dict[pred]))
            elif stat == 'std':
                metrics_stats[pred][stat] = np.std(np.array(scores_dict[pred]))
            else:
                print('run_reccurrent(): bad stat option, filling zeros')
                corr_stats[pred][corr][stat] = 0.0

    # ZAK write mean and std's of correlations and metrics to output file
    model_preds_file_stats = 'output/{}_{}/model_preds_stats.csv'.format(
        model_type,
        cur_time)
    model_stat_rows = []
    if not os.path.exists(model_preds_file_stats):
        model_stat_rows.append(
            ["model_type", 'set_name', "time", "model_name", "fold_type",
             "num_folds", 'metrics_type', 'metrics_length', 'stats_type',
             'metrics_value', 'pearson', 'kendall', 'spearman'])
    for pred in pred_list:
        fold_type = pred.split('_')[0]
        for stat in stat_list:
            if 'rnd' in pred:
                t_metrics = 'round'
            else:
                t_metrics = metrics
            model_stat_rows.append(
                [model_type, set_name, time, model_name, fold_type,
                 len(scores_dict[pred]),
                 t_metrics, test_preds_stack.shape[0], stat,
                 metrics_stats[pred][stat],
                 corr_stats[pred]['pearson'][stat],
                 corr_stats[pred]['kendall'][stat],
                 corr_stats[pred]['spearman'][stat]])

    append_data(model_stat_rows, model_preds_file_stats)

    # ZAK modified to use dictionary
    print('########################################')
    if metrics == 'accuracy':
        print("%s %.2f%% (+/- %.2f%%)" % ('Total ' + metrics + ': ',
                                          metrics_stats['test']['mean'],
                                          metrics_stats['test']['std']))
    elif metrics == 'mse':
        print("%s %.3f (+/- %.3f)" % ('Total ' + metrics + ': ',
                                      metrics_stats['test']['mean'],
                                      metrics_stats['test']['std']))
    print('#########################################')

    # ZAK add to compute correlations across all folds at end of loop
    # get correlations across all folds
    corr_dict = pandas_utils.get_corr_from_predictions(test_preds_stack,
                y_test_stack, metrics=metrics,
                corr=['spearman', 'kendall', 'pearson'])
    # write correlations to output file
    model_preds_file_corr = 'output/{}_{}/model_preds_corr.csv'.format(
        model_type,
        cur_time)
    model_corr_rows = []
    if not os.path.exists(model_preds_file_corr):
        model_corr_rows.append(
            ["model_type", 'set_name',"time", "model_name", "fold_type",
             "num_folds", 'metrics_type', 'metrics_length','metrics_test_mean',
             'metrics_test_std', 'pearson', 'kendall', 'spearman'])
    model_corr_rows.append(
        [model_type, set_name, time, model_name, "test",
         len(scores_dict['test']),
         metrics, test_preds_stack.shape[0], metrics_stats['test']['mean'],
         metrics_stats['test']['std'], corr_dict['pearson'], corr_dict['kendall'],
         corr_dict['spearman']])

    if metrics == 'mse':  # write one more row for round version
        corr_dict_round = pandas_utils.get_corr_from_predictions(
            test_preds_stack, y_test_stack, metrics=metrics,
            corr=['spearman', 'kendall', 'pearson'], round=True)
        model_corr_rows.append(
            [model_type, set_name, time, model_name, "test",
             len(scores_dict['test']),
             'round', test_preds_stack.shape[0], metrics_stats['test_rnd']['mean'],
             metrics_stats['test_rnd']['std'], corr_dict_round['pearson'],
             corr_dict_round['kendall'], corr_dict_round['spearman']])

    append_data(model_corr_rows, model_preds_file_corr)

    print('**************************************************')
    print('FINAL CORRELATIONS = ', corr_dict)
    print('**************************************************')

    sys.stdout = stdout
    sys.stderr = stderr
    logging_output.close()
    logging_error.close()

    return


def prep_data_ff(start_time):
    id_col = 'student_id'
    secondary_col = "survey_id"
    subset_col = 'survey_question'
    subsets = ["Anxiety", "Arousal", "Boredom", "Confusion", "Contentment", "Curiosity", "Disappointment", "Engagement",
               "Frustration", "Happiness", "Hopefulness", "Interest", "Mind Wandering", "Pleasantness", "Pride",
               "Relief", "Sadness", "Surprise"]
    cv_k_fold = 10

    # divide the training set by this to produce validation set
    train_denom_for_val = 3

    random_seed = 100

    # scaling factor is equivalent to the new range of values (on training set)
    #   though actual extremes depend on the data
    data_scaling_factor = 6.0
    current_process = psutil.Process(os.getpid())

    # grab current timestamp in UTC timezone -- use this for individual model outputs
    cur_time = start_time.strftime("%m%d%Y-%H%M%S")
    process_label = 'input_prep'

    logging_dir = 'output/{}_{}/logging/'.format(process_label, cur_time)
    data_dump_dir = 'output/{}_{}/data_dump/'.format(process_label, cur_time)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(data_dump_dir, exist_ok=True)

    debugging_output_file = logging_dir + "{}_DEBUG-OUTPUT_{}.txt".format(process_label, cur_time)
    debugging_error_file = logging_dir + "{}_DEBUG-ERROR_{}.txt".format(process_label, cur_time)

    # this sets up output to both stdout and a logging file
    # adapted from http://stackoverflow.com/questions/20898212/how-to-automatically-direct-print-statements-outputs-to-a-file-in-python
    logging_output = open(debugging_output_file, 'a')
    logging_error = open(debugging_error_file, 'a')
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Logger(logging_output, sys.stdout)
    sys.stderr = Logger(logging_error, sys.stderr)

    input_dirs = []
    for subset in subsets:
        input_dir = 'input/{}/set_{}/'.format(cur_time, subset)
        os.makedirs(input_dir, exist_ok=True)
        input_dirs.append(input_dir)

    pickle_dir = 'input/{}'.format(cur_time)
    pickle_input_data = pickle_dir + '/input_data_list.pickle'
    pickle_models_list = pickle_dir + '/models_list.pickle'

    # ZAK save most recent pickle_dir to use if needed for reproducible results
    # there may be a more elegant way to use the most recently created
    # directory that also has a pickle file in it...
    with open('most_recent_pickle_dir.pickle', 'wb') as pickle_file:
        pickle.dump(pickle_dir, pickle_file)

    # INITIAL DATA PRE-PROCESSING
    raw_data = pd.read_csv(
        'data/Alg1_5min_SEQUENCE_features.csv',
        sep=',')

    raw_data.index.rename('df_index', True)  # True makes it rename in place

    action_cols = [
        "bio_video_watch", "karma_awarded", "leaderboard_load", "tys_answer",
        "tys_finish", "tys_load", "video_completed", "video_pause",
        "video_play", "video_seek", "video_watch", "wall_load_more",
        "wall_page_load", "quiz_attempted", "returned_to_previous"
    ]

    video_col = "video_viewed_percent_duration"

    # # NOMINAL RE-MAPPING
    # browser_col = "browser"
    # browser_mapping = {
    #     "Chrome": "Chrome",
    #     "Firefox": "Firefox",
    #     "Internet Explorer": "other",
    #     "mobile-browser": "other",
    #     "Mozilla": "other"
    # }
    # raw_data[browser_col] = raw_data[browser_col].map(browser_mapping)
    #
    # os_col = "os"
    # os_mapping = {
    #     "Unknown Windows OS": "Windows",
    #     "Windows Longhorn": "Windows",
    #     "Windows XP": "Windows",
    #     "Mac OS X": "Mac",
    #     "android": "other",
    #     "Linux": "other",
    #     "Unknown Platform": "other"
    # }
    # raw_data[os_col] = raw_data[os_col].map(os_mapping)
    #
    # # NOMINAL DUMMY-CODING
    # label_demographic_cols = [
    #     "free_lunch", "hispanic_ethnicity", "race", "gender"
    # ]
    # demo_dummies = raw_data[label_demographic_cols].copy()
    # demo_dummies = pd.get_dummies(demo_dummies, prefix=label_demographic_cols, dummy_na=True)
    # demo_dummy_cols = demo_dummies.columns.values.tolist()
    # raw_data[demo_dummy_cols] = demo_dummies[demo_dummy_cols]
    #
    # label_context_cols = [
    #     "browser", "os"
    # ]
    # context_dummies = raw_data[label_context_cols].copy()
    # context_dummies = pd.get_dummies(context_dummies, prefix=label_context_cols, dummy_na=True)
    # context_dummy_cols = context_dummies.columns.values.tolist()
    # raw_data[context_dummy_cols] = context_dummies[context_dummy_cols]
    #
    # numeric_demographic_cols = [
    #     "grade", "age"
    # ]
    #
    # numeric_context_cols = [
    #     "survey_hour", "school_size_by_surveyed"
    # ]

    reg_output_cols = [
        "survey_answer"
    ]

    # CLIP ACTION COLUMNS
    raw_data[action_cols] = raw_data[action_cols].clip(lower=0, upper=10)
    raw_data[video_col] = raw_data[video_col].clip(lower=0, upper=100)

    # DECLARE MODEL COLUMNS
    numeric_cols = action_cols + [video_col] # + numeric_demographic_cols + numeric_context_cols
    # label_cols = label_demographic_cols + label_context_cols
    all_model_cols = [id_col] + [secondary_col] + [subset_col] + numeric_cols + reg_output_cols

    # REDUCE SEQUENTIAL DATA TO INSTANCES
    raw_data = raw_data[all_model_cols].groupby([id_col, secondary_col, subset_col]).mean().reset_index()

    # numeric processing cols -- to be processed during k-fold iterations
    #   NOTE: these columns do not exist, but will be added during processing
    proc_num_cols = []
    for proc_num_col in numeric_cols:
        proc_num_cols.append("proc_" + str(proc_num_col))

    # proc_label_cols = demo_dummy_cols + context_dummy_cols

    # original input columns for record-keeping
    orig_input_cols = numeric_cols  # + proc_label_cols

    # input columns for model
    input_cols = proc_num_cols  # + proc_label_cols

    # keep_cols = ['couple', 'convo', 'convo_time'] + data_cols

    # remove_pids = []
    #
    # print('removing pids: ' + str(remove_pids))
    #
    # # drop participants with missing data
    # for pid in remove_pids:
    #     raw_data = raw_data.drop(raw_data[raw_data[id_col] == pid].index)

    # drop "non-conversation" data -- first 15 seconds and anything beyond the subsequent 6 minutes
    # raw_data = raw_data.drop(raw_data[raw_data[time_col] < data_start_time].index)
    # raw_data = raw_data.drop(raw_data[raw_data[time_col] > data_end_time].index)

    models_input_output_list = []
    vll_reg_model = {
        "input": input_cols,
        "output": reg_output_cols,
        "model_name": "vll_reg_ff_action",
        "orig_input": orig_input_cols,
        "orig_output": reg_output_cols
    }

    models_input_output_list.append(vll_reg_model)

    # write models and input / output features to pickle
    with open(pickle_models_list, 'wb') as pickle_file:
        pickle.dump(models_input_output_list, pickle_file)

    # create input data structures
    input_data_list = []

    np.random.seed(random_seed)

    for set_num, set_name in enumerate(subsets):

        # ZAK add set_list, part of major change for subsets and pooling
        set_list = []

        print('Starting ' + str(set_num) + ' ' + str(set_name))
        process_report(start_time, current_process)

        pandas_df = raw_data[raw_data[subset_col] == set_name]
        # pandas_df = raw_data[raw_data.session_num == convo]

        # data = raw_data
        data = pandas_df.copy().reset_index(drop=True)  # Make a copy and reset index so it is easy to modify.

        for iter_num, fold_indices in enumerate(GroupKFold(cv_k_fold).split(X=data, groups=data[id_col])):
            train_indices = fold_indices[0]
            test_indices = fold_indices[1]

            # perform train / val split
            initial_train_data = data.iloc[train_indices]

            val_fold_indices = GroupKFold(train_denom_for_val).split(initial_train_data, initial_train_data,
                                                                     groups=initial_train_data[id_col])
            revised_train_indices, val_indices = next(val_fold_indices)

            # data processing -- normalize actions to [0, 1]
            action_train_data, action_val_data = nn_util.normalize_zero_to_value(
                train_pandas_df=initial_train_data.iloc[revised_train_indices],
                test_pandas_df=initial_train_data.iloc[val_indices],
                columns=action_cols,
                norm_value=10
            )
            action_redundant_train_data, action_test_data = nn_util.normalize_zero_to_value(
                train_pandas_df=initial_train_data.iloc[revised_train_indices],
                test_pandas_df=data.iloc[test_indices],
                columns=action_cols,
                norm_value=10
            )
            # print(train_data.columns.values)

            # data processing -- normalize video viewing to [0, 1]
            train_data, val_data = nn_util.normalize_zero_to_value(
                train_pandas_df=action_train_data,
                test_pandas_df=action_val_data,
                columns=[video_col],
                norm_value=100
            )
            redundant_train_data, test_data = nn_util.normalize_zero_to_value(
                train_pandas_df=action_redundant_train_data,
                test_pandas_df=action_test_data,
                columns=[video_col],
                norm_value=100
            )

            # # no data processing -- just do cross-validation partitions
            # train_data = initial_train_data.iloc[revised_train_indices]
            # val_data = initial_train_data.iloc[val_indices]
            # redundant_train_data = initial_train_data.iloc[revised_train_indices]
            # test_data = data.iloc[test_indices]

            # get unique participant IDs for processing and output
            train_pids = train_data[id_col].unique()
            red_train_pids = redundant_train_data[id_col].unique()
            val_pids = val_data[id_col].unique()
            test_pids = test_data[id_col].unique()
            num_train_pids = len(train_pids)
            num_red_train_pids = len(red_train_pids)
            num_val_pids = len(val_pids)
            num_test_pids = len(test_pids)

            # prepare unique pids for debugging output (list needed for appendData)
            train_pids_list = [train_pids.tolist()]
            red_train_pids_list = [red_train_pids.tolist()]
            val_pids_list = [val_pids.tolist()]
            test_pids_list = [test_pids.tolist()]

            # retrieve lengths of data sets
            len_train_data = len(train_data)
            len_red_train_data = len(redundant_train_data)
            len_val_data = len(val_data)
            len_test_data = len(test_data)

            # dump train/test pids for debugging
            append_data(train_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_train_pids) + "_rows=" + str(len_train_data) + "_train_pids.txt")
            append_data(red_train_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_red_train_pids) + "_rows=" + str(len_red_train_data) + "_red_train_pids.txt")
            append_data(val_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_val_pids) + "_rows=" + str(len_val_data) + "_val_pids.txt")
            append_data(test_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_test_pids) + "_rows=" + str(len_test_data) + "_test_pids.txt")

            # dump z-score train/test data for input
            redundant_train_data.to_csv(
                input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                    num_red_train_pids) + "_rows=" + str(len_red_train_data) + "_seed=" + str(
                    random_seed) + "_red_train.txt")
            train_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_train_pids) + "_rows=" + str(len_train_data) + "_seed=" + str(random_seed) + "_train.txt"
            val_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_val_pids) + "_rows=" + str(len_val_data) + "_seed=" + str(random_seed) + "_val.txt"
            test_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_test_pids) + "_rows=" + str(len_test_data) + "_seed=" + str(random_seed) + "_test.txt"
            train_data.to_csv(train_filename)
            val_data.to_csv(val_filename)
            test_data.to_csv(test_filename)

            # create input dictionary for current iteration
            cur_input_dict = {'set_name': set_name,
                              'k': iter_num,
                              'train': train_filename,
                              'val': val_filename,
                              'test': test_filename,
                              "primary_col": id_col,
                              "secondary_col": secondary_col
                              }
            # ZAK mod to run on set_names, part of major change for subsets and pooling
            set_list.append(cur_input_dict)

        # ZAK mod to run on set_names, part of major change for subsets and pooling
        input_data_list.append(set_list)  # items = # of set_names

    # write the input list of dictionaries as a pickle
    with open(pickle_input_data, 'wb') as pickle_out:
        pickle.dump(input_data_list, pickle_out)

    print('Done processing input data')
    process_report(start_time, current_process)

    sys.stdout = stdout
    sys.stderr = stderr
    logging_output.close()
    logging_error.close()

    return pickle_dir


# ZAK added categorical_output flag
def prep_data_lstm(start_time, categorical_output = False):
    """
    prep data for lstm application. Added categorical_output option for
    using categorical instead of mse. Note that if we want to eventually
    have option do do some subsets categorical and some mse, we need to
    replace this flag with a dictionary or matching list
    :param start_time:
    :param categorical_output: if True, convert output columns to
               one hot
    :return:
    """
    id_col = 'student_id'
    secondary_col = "survey_id"
    subset_col = 'survey_question'
    subsets = ["Anxiety", "Arousal", "Boredom", "Confusion", "Contentment", "Curiosity", "Disappointment", "Engagement",
               "Frustration", "Happiness", "Hopefulness", "Interest", "Mind Wandering", "Pleasantness", "Pride",
               "Relief", "Sadness", "Surprise"]
    cv_k_fold = 10

    # divide the training set by this to produce validation set
    train_denom_for_val = 3

    random_seed = 100

    # scaling factor is equivalent to the new range of values (on training set)
    #   though actual extremes depend on the data
    data_scaling_factor = 6.0
    current_process = psutil.Process(os.getpid())

    # grab current timestamp in UTC timezone -- use this for individual model outputs
    cur_time = start_time.strftime("%m%d%Y-%H%M%S")
    process_label = 'input_prep'

    logging_dir = 'output/{}_{}/logging/'.format(process_label, cur_time)
    data_dump_dir = 'output/{}_{}/data_dump/'.format(process_label, cur_time)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(data_dump_dir, exist_ok=True)

    debugging_output_file = logging_dir + "{}_DEBUG-OUTPUT_{}.txt".format(process_label, cur_time)
    debugging_error_file = logging_dir + "{}_DEBUG-ERROR_{}.txt".format(process_label, cur_time)

    # this sets up output to both stdout and a logging file
    # adapted from http://stackoverflow.com/questions/20898212/how-to-automatically-direct-print-statements-outputs-to-a-file-in-python
    logging_output = open(debugging_output_file, 'a')
    logging_error = open(debugging_error_file, 'a')
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Logger(logging_output, sys.stdout)
    sys.stderr = Logger(logging_error, sys.stderr)

    input_dirs = []
    for subset in subsets:
        input_dir = 'input/{}/set_{}/'.format(cur_time, subset)
        os.makedirs(input_dir, exist_ok=True)
        input_dirs.append(input_dir)

    pickle_dir = 'input/{}'.format(cur_time)
    pickle_input_data = pickle_dir + '/input_data_list.pickle'
    pickle_models_list = pickle_dir + '/models_list.pickle'

    # ZAK save most recent pickle_dir to use if needed for reproducible results
    # there may be a more elegant way to use the most recently created
    # directory that also has a pickle file in it...
    with open('most_recent_pickle_dir.pickle', 'wb') as pickle_file:
        pickle.dump(pickle_dir, pickle_file)

    # INITIAL DATA PRE-PROCESSING
    raw_data = pd.read_csv(
        'data/Alg1_5min_SEQUENCE_features.csv',
        sep=',',
        low_memory=False
    )

    raw_data.index.rename('df_index', True)  # True makes it rename in place

    action_cols = [
        "bio_video_watch", "karma_awarded", "leaderboard_load", "tys_answer", "tys_finish", "tys_load",
        "video_completed", "video_pause", "video_play", "video_seek", "video_watch", "wall_load_more",
        "wall_page_load", "quiz_attempted", "returned_to_previous"
    ]

    video_col = "video_viewed_percent_duration"

    # # NOMINAL RE-MAPPING
    # browser_col = "browser"
    # browser_mapping = {
    #     "Chrome": "Chrome",
    #     "Firefox": "Firefox",
    #     "Internet Explorer": "other",
    #     "mobile-browser": "other",
    #     "Mozilla": "other"
    # }
    # raw_data[browser_col] = raw_data[browser_col].map(browser_mapping)
    #
    # os_col = "os"
    # os_mapping = {
    #     "Unknown Windows OS": "Windows",
    #     "Windows Longhorn": "Windows",
    #     "Windows XP": "Windows",
    #     "Mac OS X": "Mac",
    #     "android": "other",
    #     "Linux": "other",
    #     "Unknown Platform": "other"
    # }
    # raw_data[os_col] = raw_data[os_col].map(os_mapping)
    #
    # # NOMINAL DUMMY-CODING
    # label_demographic_cols = [
    #     "free_lunch", "hispanic_ethnicity", "race", "gender"
    # ]
    # demo_dummies = raw_data[label_demographic_cols].copy()
    # demo_dummies = pd.get_dummies(demo_dummies, prefix=label_demographic_cols, dummy_na=True)
    # demo_dummy_cols = demo_dummies.columns.values.tolist()
    # raw_data[demo_dummy_cols] = demo_dummies[demo_dummy_cols]
    #
    # label_context_cols = [
    #     "browser", "os"
    # ]
    # context_dummies = raw_data[label_context_cols].copy()
    # context_dummies = pd.get_dummies(context_dummies, prefix=label_context_cols, dummy_na=True)
    # context_dummy_cols = context_dummies.columns.values.tolist()
    # raw_data[context_dummy_cols] = context_dummies[context_dummy_cols]
    #
    # numeric_demographic_cols = [
    #     "grade", "age"
    # ]
    #
    # numeric_context_cols = [
    #     "survey_hour", "school_size_by_surveyed"
    # ]

    reg_output_cols = [
        "survey_answer"
    ]

    # CLIP ACTION COLUMNS -- max 10 actions per 30 seconds
    raw_data[action_cols] = raw_data[action_cols].clip(lower=0, upper=10)
    raw_data[video_col] = raw_data[video_col].clip(lower=0, upper=100)

    numeric_cols = action_cols + [video_col] #+ numeric_demographic_cols

    # ZAK code below added from discussion with Joseph to handle lstm + FF
    # case where we no longer want to process the demo data with LSTM but
    # separately and then merge it
    proc_lstm_cols = []
    for proc_lstm_col in action_cols + [video_col]:
        proc_lstm_cols.append("proc_" + str(proc_lstm_col))

    # proc_ff_cols = []
    # for proc_ff_col in numeric_demographic_cols:
    #     proc_ff_cols.append("proc_" + str(proc_ff_col))

    # original input columns for record-keeping
    orig_input_cols = action_cols + [
        video_col] #+ numeric_demographic_cols + demo_dummy_cols
    # input columns for demographics model
    # demo_cols = demo_dummy_cols + context_dummy_cols + numeric_demographic_cols
    # demo_cols = demo_dummy_cols + proc_ff_cols

    models_input_output_list = []
    vll_reg_model = {
        "input": proc_lstm_cols,
        "output": reg_output_cols,
        "model_name": "vll_reg_lstm_action",
        "orig_input": orig_input_cols,
        "orig_output": reg_output_cols
    }

    # ZAK below modified for dual input model LSTM + FF
    # models_input_output_list = []
    # vll_reg_model = {
    #     "input": proc_lstm_cols,
    #     "output": reg_output_cols,
    #     "model_name": "vll_reg_lstm_all_features",
    #     "orig_input": orig_input_cols,
    #     "orig_output": reg_output_cols,
    #     "input2": demo_cols
    # }

    models_input_output_list.append(vll_reg_model)

    # write models and input / output features to pickle
    with open(pickle_models_list, 'wb') as pickle_file:
        pickle.dump(models_input_output_list, pickle_file)

    # keep_cols = ['couple', 'convo', 'convo_time'] + data_cols

    # raw_data = pd.read_csv('data/Engagement_LSTM_features_Sep-24-2017_to_Dec-15-2017.csv',
    #                        sep=',')
    # # raw_data = pd.read_csv('data/responded_reformat_mean_action_features_transpose_data_30s_with_tys_meta_features.csv', sep=',')
    # # raw_data = pd.HDFStore('data/coreg_reproc_1s.h5', mode='r')['coreg_reproc_1s']
    #
    # raw_data.index.rename('df_index', True)  # True makes it rename in place

    # remove_pids = []

    # print('removing pids: ' + str(remove_pids))

    # # drop participants with missing data
    # for pid in remove_pids:
    #     raw_data = raw_data.drop(raw_data[raw_data[id_col] == pid].index)
    #     # raw_data = raw_data.drop(raw_data[raw_data.participant_id == pid].index)

    # drop "non-conversation" data -- first 15 seconds and anything beyond the subsequent 6 minutes
    # raw_data = raw_data.drop(raw_data[raw_data[time_col] < data_start_time].index)
    # raw_data = raw_data.drop(raw_data[raw_data[time_col] > data_end_time].index)

    # create input data structures
    input_data_list = []

    np.random.seed(random_seed)

    for set_num, set_name in enumerate(subsets):
        #    for convo in [2, 3, 4]:
        # ZAK add set_list, part of major change for subsets and pooling
        set_list = []

        print('Starting ' + str(set_num) + str(set_name))
        process_report(start_time, current_process)

        pandas_df = raw_data[raw_data[subset_col] == set_name]
        # pandas_df = raw_data[raw_data.session_num == convo]

        # data = raw_data
        data = pandas_df.copy().reset_index(drop=True)  # Make a copy and reset index so it is easy to modify.

        for iter_num, fold_indices in enumerate(GroupKFold(cv_k_fold).split(X=data, groups=data[id_col])):
            train_indices = fold_indices[0]
            test_indices = fold_indices[1]

            # perform train / val split
            initial_train_data = data.iloc[train_indices]

            val_fold_indices = GroupKFold(train_denom_for_val).split(initial_train_data, initial_train_data,
                                                                     groups=initial_train_data[id_col])
            revised_train_indices, val_indices = next(val_fold_indices)

            # data processing -- normalize actions to [0, 1]
            action_train_data, action_val_data = nn_util.normalize_zero_to_value(
                train_pandas_df=initial_train_data.iloc[revised_train_indices],
                test_pandas_df=initial_train_data.iloc[val_indices],
                columns=action_cols,
                norm_value=10
            )
            action_redundant_train_data, action_test_data = nn_util.normalize_zero_to_value(
                train_pandas_df=initial_train_data.iloc[revised_train_indices],
                test_pandas_df=data.iloc[test_indices],
                columns=action_cols,
                norm_value=10
            )
            # data processing -- normalize video viewing to [0, 1]
            train_data, val_data = nn_util.normalize_zero_to_value(
                train_pandas_df=action_train_data,
                test_pandas_df=action_val_data,
                columns=[video_col],
                norm_value=100
            )
            redundant_train_data, test_data = nn_util.normalize_zero_to_value(
                train_pandas_df=action_redundant_train_data,
                test_pandas_df=action_test_data,
                columns=[video_col],
                norm_value=100
            )

            # # no data processing -- just do cross-validation partitions
            # train_data = initial_train_data.iloc[revised_train_indices]
            # val_data = initial_train_data.iloc[val_indices]
            # redundant_train_data = initial_train_data.iloc[revised_train_indices]
            # test_data = data.iloc[test_indices]

            # get unique participant IDs for processing and output
            train_pids = train_data[id_col].unique()
            red_train_pids = redundant_train_data[id_col].unique()
            val_pids = val_data[id_col].unique()
            test_pids = test_data[id_col].unique()
            num_train_pids = len(train_pids)
            num_red_train_pids = len(red_train_pids)
            num_val_pids = len(val_pids)
            num_test_pids = len(test_pids)

            # prepare unique pids for debugging output (list needed for appendData)
            train_pids_list = [train_pids.tolist()]
            red_train_pids_list = [red_train_pids.tolist()]
            val_pids_list = [val_pids.tolist()]
            test_pids_list = [test_pids.tolist()]

            # retrieve lengths of data sets
            len_train_data = len(train_data)
            len_red_train_data = len(redundant_train_data)
            len_val_data = len(val_data)
            len_test_data = len(test_data)

            # dump train/test pids for debugging
            append_data(train_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_train_pids) + "_rows=" + str(len_train_data) + "_train_pids.txt")
            append_data(red_train_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_red_train_pids) + "_rows=" + str(len_red_train_data) + "_red_train_pids.txt")
            append_data(val_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_val_pids) + "_rows=" + str(len_val_data) + "_val_pids.txt")
            append_data(test_pids_list, data_dump_dir + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_test_pids) + "_rows=" + str(len_test_data) + "_test_pids.txt")

            # dump z-score train/test data for input
            redundant_train_data.to_csv(
                input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                    num_red_train_pids) + "_rows=" + str(len_red_train_data) + "_seed=" + str(
                    random_seed) + "_red_train.txt")
            train_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_train_pids) + "_rows=" + str(len_train_data) + "_seed=" + str(random_seed) + "_train.txt"
            val_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_val_pids) + "_rows=" + str(len_val_data) + "_seed=" + str(random_seed) + "_val.txt"
            test_filename = input_dirs[set_num] + "c=" + str(set_name) + "_i=" + str(iter_num) + "_n=" + str(
                num_test_pids) + "_rows=" + str(len_test_data) + "_seed=" + str(random_seed) + "_test.txt"
            train_data.to_csv(train_filename)
            val_data.to_csv(val_filename)
            test_data.to_csv(test_filename)

            # create input dictionary for current iteration
            cur_input_dict = {'set_name': set_name,
                              'k': iter_num,
                              'train': train_filename,
                              'val': val_filename,
                              'test': test_filename,
                              "primary_col": id_col,
                              "secondary_col": secondary_col
                              }
            # ZAK mod to run on set_names, part of major change for subsets and pooling
            set_list.append(cur_input_dict)

        # ZAK mod to run on set_names, part of major change for subsets and pooling
        input_data_list.append(set_list)   # items = # of set_names

    # write the input list of dictionaries as a pickle
    with open(pickle_input_data, 'wb') as pickle_out:
        pickle.dump(input_data_list, pickle_out)


    print('Done processing input data')
    process_report(start_time, current_process)

    sys.stdout = stdout
    sys.stderr = stderr
    logging_output.close()
    logging_error.close()

    return pickle_dir


def run_ff():

    # ZAK add option to use previous pickle file
    use_previous_pickle = False

    # grab current time
    start_time = datetime.datetime.utcnow()

    current_process = psutil.Process(os.getpid())

    process_report(start_time, current_process)

    # CREATE CROSS-VALIDATION PARTITIONS, STANDARDIZE & NORMALIZE BY TRAINING FOLD
    # ZAK add option to use previous pickle file
    if use_previous_pickle:
        with open('most_recent_pickle_dir.pickle', 'rb') as pickle_file:
            pickle_dir = pickle.load(pickle_file)
    else:
        pickle_dir = prep_data_ff(start_time)

    # BEGIN TEN-FOLD CROSS-VALIDATION DATA
    pickle_input_data = pickle_dir + '/input_data_list.pickle'
    pickle_models_list = pickle_dir + '/models_list.pickle'

    with open(pickle_input_data, 'rb') as pickle_file:
        input_data_list = pickle.load(pickle_file)
    with open(pickle_models_list, 'rb') as pickle_file:
        models_list = pickle.load(pickle_file)

    # ZAK moved this up here, should be replaced by a dictionary
    for input_output in models_list:
        in_columns = input_output["input"]
        out_columns = input_output["output"]
        model_name = input_output["model_name"]
        orig_in = input_output["orig_input"]
        orig_out = input_output["orig_output"]
        # demo_cols = input_output['input2']

    # ZAK part of major change for subsets and pooling
    # input_data_list is now a list of lists of dictionaries
    # note iter_num is now passed as function argument to run_recurrent()
    for iter_num, set_list in enumerate(input_data_list):
        # print('length of set_list = ', len(set_list))
        data_list = []
        for file_entry in set_list:
            train_data = pd.read_csv(file_entry['train'], sep=',')
            train_data.index.rename('train_index',
                                True)  # True makes it rename in place
            val_data = pd.read_csv(file_entry['val'], sep=',')
            val_data.index.rename('val_index',
                                  True)  # True makes it rename in place
            test_data = pd.read_csv(file_entry['test'], sep=',')
            test_data.index.rename('test_index',
                                   True)  # True makes it rename in place
            data_dict = {'set_name': file_entry['set_name'],
                         'k': file_entry['k'], 'train': train_data,
                         'val': val_data,
                         'test': test_data,
                         'primary_col': file_entry['primary_col'],
                         'secondary_col': file_entry['secondary_col']}
            data_list.append(data_dict)

        process_report(start_time, current_process)

        pool = multiprocessing.Pool(1)

        # func_name = None
        func_dict = {'ff_in_cols': in_columns, 'num_neurons': 32,
                     'loss_func': 'mean_squared_error',
                     'out_columns': out_columns,
                     'out_activation': None,
                     'model_def': 'ff_l1_n32_leaky',
                     'metrics': 'mse'
                     }

        # def run_joint_ff(iter_num, model_type, in_columns, out_columns,
        #          orig_in_cols, orig_out_cols,
        #          num_layers, num_neurons, size_batches, num_epochs,
        #          stop_patience, dropout_perc, model_name, data_list,
        #          start_time, activation, out_activation, loss_func, func_dict)
        # ZAK added iter_num, part of major change for subsets and pooling
        # also added demo_cols for LSTM + FF and passing in func_name and
        # func_dict. Replaced some previous hard coded values with values from
        # func_dict
        # I think we should remove rec_structure and use other info instead
        pool.apply(run_joint_ff,
                   (iter_num, func_dict['model_def'], in_columns, out_columns,
                    orig_in, orig_out,
                    1, func_dict['num_neurons'], 32, 250,
                    -1, -1, model_name, data_list,
                    start_time, None, None, func_dict['loss_func'], func_dict))

        pool.close()
    # END JOINT MODELING

    process_report(start_time, current_process)

    print("Done processing")

    return

def run_lstm():

    # ZAK add option to use previous pickle file
    use_previous_pickle = False
    # grab current time
    start_time = datetime.datetime.utcnow()

    current_process = psutil.Process(os.getpid())

    process_report(start_time, current_process)

    # CREATE CROSS-VALIDATION PARTITIONS, STANDARDIZE & NORMALIZE BY TRAINING FOLD
    # ZAK add option to use previous pickle file
    if use_previous_pickle:
        with open('most_recent_pickle_dir.pickle', 'rb') as pickle_file:
            pickle_dir = pickle.load(pickle_file)
    else:
        pickle_dir = prep_data_lstm(start_time)

    # BEGIN LOAD TEN-FOLD CROSS-VALIDATION DATA
    pickle_input_data = pickle_dir + '/input_data_list.pickle'
    pickle_models_list = pickle_dir + '/models_list.pickle'

    with open(pickle_input_data, 'rb') as pickle_file:
        input_data_list = pickle.load(pickle_file)
    with open(pickle_models_list, 'rb') as pickle_file:
        models_list = pickle.load(pickle_file)

    # print('len of input_data_list = ', len(input_data_list))
    # print('pickle_input_data file = ',pickle_input_data)

    # ZAK moved this up here, should be replaced by a dictionary
    for input_output in models_list:
        in_columns = input_output["input"]
        out_columns = input_output["output"]
        model_name = input_output["model_name"]
        orig_in = input_output["orig_input"]
        orig_out = input_output["orig_output"]
        # demo_cols = input_output['input2']

    # for iter_num, file_entry in enumerate(input_data_list): # ZAK old way

    # ZAK part of major change for subsets and pooling
    # input_data_list is now a list of lists of dictionaries
    # note iter_num is now passed as function argument to run_recurrent()
    for iter_num, set_list in enumerate(input_data_list):
        # print('length of set_list = ', len(set_list))
        data_list = []
        for file_entry in set_list:
            train_data = pd.read_csv(file_entry['train'], sep=',')
            train_data.index.rename('train_index',
                                True)  # True makes it rename in place
            val_data = pd.read_csv(file_entry['val'], sep=',')
            val_data.index.rename('val_index',
                                  True)  # True makes it rename in place
            test_data = pd.read_csv(file_entry['test'], sep=',')
            test_data.index.rename('test_index',
                                   True)  # True makes it rename in place
            data_dict = {'set_name': file_entry['set_name'],
                         'k': file_entry['k'], 'train': train_data,
                         'val': val_data,
                         'test': test_data,
                         'primary_col': file_entry['primary_col'],
                         'secondary_col': file_entry['secondary_col']}
            data_list.append(data_dict)

        process_report(start_time, current_process)

        pool = multiprocessing.Pool(1)

        func_name = lstm_dense
        func_dict = {'lstm_in_cols': in_columns, 'lstm_num_neurons': 32,
                     'sequence_len': 10,
                     # 'dense_in_cols': demo_cols,
                     # 'dense_num_neurons': 8,
                     'loss_func': 'mean_squared_error',
                     'out_columns': out_columns,
                     'out_activation': None,
                     'model_def': 'lstm_l1_n32_leaky',
                     'metrics': 'mse'
                     }

        # def run_recurrent(iter_num, model_type, in_columns, out_columns, orig_in_cols, orig_out_cols, demo_cols,
        #                   num_layers, num_neurons, size_batches, num_epochs, stop_patience, dropout_perc,
        #                   model_name, data_list, start_time, sequence_len, rec_structure, activation,
        #                   out_activation, loss_func, func_name, func_dict):
        # ZAK added iter_num, part of major change for subsets and pooling
        # also added demo_cols for LSTM + FF and passing in func_name and
        # func_dict. Replaced some previous hard coded values with values from
        # func_dict
        # I think we should remove rec_structure and use other info instead
        pool.apply(run_recurrent,
                   (iter_num, func_dict['model_def'], in_columns, out_columns, orig_in, orig_out, None,
                    1, func_dict['lstm_num_neurons'], 32, 250, -1, -1,
                    model_name, data_list, start_time, func_dict['sequence_len'], "lstm", None,
                    None, func_dict['loss_func'], func_name, func_dict))

        pool.close()
    # END JOINT MODELING

    process_report(start_time, current_process)

    print("Done processing")

    return


def main():
    # run_ff()
    run_lstm()
    pass


if __name__ == "__main__":
    main()
