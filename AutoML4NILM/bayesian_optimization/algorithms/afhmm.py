from __future__ import print_function, division
import warnings
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

warnings.filterwarnings("ignore")
import time
from nilmtk import DataSet, MeterGroup, HDFDataStore
from algorithms.AFHMM.afhmmdisaggregator import AFHMMDisaggregator

import pandas as pd
import argparse
import json
from utils import metrics


def afhmm(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building,
          test_start, test_end, meter_key, sample_period):
    start = time.time()

    # Load datasets
    train = DataSet(dataset_path)
    train.set_window(start=train_start, end=train_end)
    val = DataSet(dataset_path)
    val.set_window(start=val_start, end=val_end)
    test = DataSet(dataset_path)
    test.set_window(start=test_start, end=test_end)

    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec

    appliances = [meter_key]
    selected_meters = [train_elec[app] for app in appliances]
    selected_meters.append(train_elec.mains())
    selected = MeterGroup(selected_meters)

    afhmm = AFHMMDisaggregator()

    print("========== TRAIN ============")
    afhmm.train(selected, sample_period=sample_period)

    print("========== DISAGGREGATE ============")

    # Validation
    val_disag_filename = 'disag-out-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    output.save_metadata('/',train.metadata)  # ✅ Correct

    val_mains_series = val_elec.mains().power_series_all_data(sample_period=sample_period, resample=True)
    afhmm.disaggregate(val_mains_series, output_datastore=output)
    output.close()

    # Test
    test_disag_filename = 'disag-out-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    output.save_metadata('/',train.metadata)  # ✅ Correct

    test_mains_series = test_elec.mains().power_series_all_data()
    afhmm.disaggregate(test_mains_series, output_datastore=output)
    output.close()

    # Validation Metrics
    result_val = DataSet(val_disag_filename)
    res_elec_val = result_val.buildings[val_building].elec
    rpaf_val = metrics.recall_precision_accuracy_f1(res_elec_val[meter_key], val_elec[meter_key])

    val_metrics_results_dict = {
        'recall_score': rpaf_val[0],
        'precision_score': rpaf_val[1],
        'accuracy_score': rpaf_val[2],
        'f1_score': rpaf_val[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec_val[meter_key], val_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec_val[meter_key], val_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec_val[meter_key], val_elec[meter_key]),
        'nad': metrics.nad(res_elec_val[meter_key], val_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec_val[meter_key], val_elec[meter_key])
    }

    # Test Metrics
    result = DataSet(test_disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

    test_metrics_results_dict = {
        'recall_score': rpaf[0],
        'precision_score': rpaf[1],
        'accuracy_score': rpaf[2],
        'f1_score': rpaf[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec[meter_key], test_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key]),
        'nad': metrics.nad(res_elec[meter_key], test_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec[meter_key], test_elec[meter_key])
    }

    end = time.time()

    model_result_data = {
        'val_metrics': val_metrics_results_dict,
        'test_metrics': test_metrics_results_dict,
        'time_taken': format(end - start, '.2f'),
        'epochs': None,
    }

    result.store.close()
    result_val.store.close()
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data


def main():
    parser = argparse.ArgumentParser(description='AFHMM Disaggregator')
    parser.add_argument('--datapath', '-d', type=str, required=True)
    parser.add_argument('--train_building', type=int, required=True)
    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_building', type=int, required=True)
    parser.add_argument('--val_start', type=str)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_building', type=int, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--sampling_rate', type=int, required=True)

    args = parser.parse_args()

    model_result_data = afhmm(
        dataset_path=args.datapath,
        train_building=args.train_building,
        train_start=pd.Timestamp(args.train_start) if args.train_start else None,
        train_end=pd.Timestamp(args.train_end),
        val_building=args.val_building,
        val_start=pd.Timestamp(args.val_start),
        val_end=pd.Timestamp(args.val_end),
        test_building=args.test_building,
        test_start=pd.Timestamp(args.test_start),
        test_end=pd.Timestamp(args.test_end) if args.test_end else None,
        meter_key=args.appliance,
        sample_period=args.sampling_rate,
    )

    with open('afhmm_json.json', 'a+') as outfile:
        json.dump(model_result_data, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    print(model_result_data)


if __name__ == "__main__":
    main()
