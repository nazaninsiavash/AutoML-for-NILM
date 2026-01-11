from __future__ import print_function, division
import warnings; warnings.filterwarnings("ignore")

from nilmtk import DataSet
import pandas as pd
import numpy as np
import argparse
import  json
import datetime
import time
import math
import glob

from sklearn.tree import DecisionTreeRegressor

# Bring packages onto the path
import sys, os
#sys.path.append('/content/drive/MyDrive/automl4nilm2/bayesian_optimization/')

sys.path.append(os.path.abspath('./bayesian_optimization/'))

from utils import metrics_np
from utils.metrics_np import Metrics



def decision_tree(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period, criterion, min_samples_split):

    # Start tracking time
    start = time.time()

    # Prepare dataset and options
    dataset_path = dataset_path
    train = DataSet(dataset_path)
    train.set_window(start=train_start, end=train_end)
    val = DataSet(dataset_path)
    val.set_window(start=val_start, end=val_end)
    test = DataSet(dataset_path)
    test.set_window(start=test_start, end=test_end)
    train_building = train_building
    val_building = val_building
    test_building = test_building
    meter_key = meter_key

    sample_period = sample_period


    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec
    X_train = train_elec.mains().power_series_all_data(sample_period=sample_period).fillna(0)
    y_train = next(train_elec[meter_key].power_series(sample_period=sample_period)).fillna(0)
    X_test = test_elec.mains().power_series_all_data(sample_period=sample_period).fillna(0)
    y_test = next(test_elec[meter_key].power_series(sample_period=sample_period)).fillna(0)
    X_val = val_elec.mains().power_series_all_data(sample_period=sample_period).fillna(0)
    y_val = next(val_elec[meter_key].power_series(sample_period=sample_period)).fillna(0)

        # Intersect between two dataframe - to make sure same training instances in X and y
        # Train set
    intersect_index = pd.Index(np.sort(list(set(X_train.index).intersection(set(y_train.index)))))
    X_train = X_train.loc[intersect_index]
    y_train = y_train.loc[intersect_index]

        # Val set
    intersect_index = pd.Index(np.sort(list(set(X_val.index).intersection(set(y_val.index)))))
    X_val = X_val.loc[intersect_index]
    y_val = y_val.loc[intersect_index]
        # Test set
    intersect_index = pd.Index(np.sort(list(set(X_test.index).intersection(set(y_test.index)))))
    X_test = X_test.loc[intersect_index]
    y_test = y_test.loc[intersect_index]

        # X_train = X_train.reshape(-1, 1)
        # y_train = y_train.reshape(-1, 1)
        # X_test = X_test.reshape(-1, 1)
        # y_test = y_test.reshape(-1, 1)

        # Get values from numpy array - Avoid server error
    X_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    X_val = X_val.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    # Model settings and hyperparameters
    min_samples_split = min_samples_split
    tree_clf = DecisionTreeRegressor(criterion = criterion, min_samples_split = min_samples_split)

    # print("========== TRAIN ============")
    tree_clf.fit(X_train, y_train)

    # print("========== DISAGGREGATE ============")
    y_val_predict = tree_clf.predict(X_val)
    y_test_predict = tree_clf.predict(X_test)

    # print("========== RESULTS ============")
    # me = Metrics(state_boundaries=[10])
    on_power_threshold = train_elec[meter_key].on_power_threshold()
    me = Metrics(state_boundaries=[on_power_threshold])
    val_metrics_results_dict = Metrics.compute_metrics(me, y_val_predict, y_val.flatten())
    test_metrics_results_dict = Metrics.compute_metrics(me, y_test_predict, y_test.flatten())

    # end tracking time
    end = time.time()

    time_taken = end-start # in seconds

    model_result_data = {
        'val_metrics':  val_metrics_results_dict,
        'test_metrics':  test_metrics_results_dict,
        'time_taken': format(time_taken, '.2f'),
        'epochs': None,
    }

    # Close Dataset files
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data


def main():
#
#     # Take in arguments from command line
    parser = argparse.ArgumentParser(description='decision tree')
    parser.add_argument('--datapath', '-d', type=str,  required=True,
                        help='hd5 filepath')

    parser.add_argument('--train_building', type=int, required=True)
    parser.add_argument('--train_start', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--train_end', type=str, required=True, help='YYYY-MM-DD')


    parser.add_argument('--val_building', type=int, required=True)
    parser.add_argument('--val_start', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--val_end', type=str, required=True, help='YYYY-MM-DD')

    parser.add_argument('--test_building', type=int, required=True)
    parser.add_argument('--test_start', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--test_end', type=str, default=None, help='YYYY-MM-DD')

    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--sampling_rate', type=int, required=True)
    parser.add_argument('--criterion', type=str, required=True)

#     # Model specific options and hyperparameters
    parser.add_argument('--min_sample_split', type=int, default=100)
    parser.add_argument('--n_estimators', type=int, default=100)
    args = parser.parse_args()

    hd5_filepath = args.datapath
    train_building = args.train_building
    train_start = pd.Timestamp(args.train_start) if args.train_start != None else None
    train_end = pd.Timestamp(args.train_end)
    val_building = args.val_building
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    test_building = args.test_building
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end) if args.test_end != None else None
    appliance = args.appliance
    criterion= args.criterion
    downsampling_period = args.sampling_rate
    min_samples_split = args.min_samples_split


    model_result_data = decision_tree(
         dataset_path=hd5_filepath,
         train_building=train_building, train_start=train_start, train_end=train_end,
        val_building=val_building, val_start=val_start, val_end=val_end,
         test_building=test_building, test_start=test_start, test_end=test_end,
         meter_key=appliance,
         sample_period=downsampling_period,

         criterion=criterion,
         min_samples_split=min_samples_split)


     # # Write options and results to file
    with open('dt_json.json', 'a+') as outfile:
        json.dump(model_result_data, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        print(model_result_data)
#
if __name__ == "__main__":
     main()
