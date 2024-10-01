import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
from collections import OrderedDict
import sys
import numpy as np
import pandas as pd

import mlflow
from sklearn.preprocessing import MinMaxScaler

import netCDF4

# Connect to Mlflow
# Remember to set to proper prot
mlflow.set_tracking_uri("http://localhost:5001")
mlflow_client = mlflow.MlflowClient()


def get_metric_names(mlflow_client, run_id):
    run_data = mlflow_client.get_run(run_id).data.to_dictionary()
    return list(run_data['metrics'].keys())


def get_param_names(mlflow_client, run_id):
    run_data = mlflow_client.get_run(run_id).data.to_dictionary()
    return list(run_data['params'].keys())


def load_data_cpl(label_set, day, sensor):
    """
    New improved with loading cpl

    label_set must be either 1 or 2
    water_vapor = 2
    temperature = 1
    """

    # must be T or q
    if label_set != 1 and label_set != 2:
        return

    # Test files
    base = "/data/nature_run/fulldays_reduced_evenmore"

    data = np.load(f"/data/nature_run/fulldays_reduced/all_cpl_{day}.npz")

    #TODO These scalars are wrong, you should just manually calc
    #TODO the resulting data ends up with an incorrect schema
    if sensor == "atms":
        rad = data["mh"]

        s_factors = np.load("minimac_scaling_factors_mh.npz")
        mins = s_factors['mins']
        maxs = s_factors['maxs']
        rad = np.nan_to_num(rad)
        rad = (rad - mins)/(maxs - mins)
    elif sensor == "hsel":
        rad = data["hsel"]

        s_factors = np.load("minimac_scaling_factors_cpl_hsel.npz")
        mins = s_factors['mins']
        maxs = s_factors['maxs']
        rad = np.nan_to_num(rad)
        rad = (rad - mins)/(maxs - mins)
    else:
        return

    cpl = data["cpl"]

    s_factors = np.load("minimac_scaling_factors_cpl_cpl.npz")
    mins = s_factors['mins']
    maxs = s_factors['maxs']
    rad = (cpl - mins)/(maxs - mins)

    nrun_sca = data["scalar"]
    nrun_tab = data["table"]

    scale_spress = np.load("spress_scalar.npz")
    mins = scale_spress['mins']
    maxs = scale_spress['maxs']
    spress_unscaled = nrun_sca[:, 3].reshape(-1, 1)
    spress = nrun_sca[:, 3].reshape(-1, 1).copy()
    spress = (spress - mins)/(maxs - mins)

    latlon = [nrun_sca[:, 0],  nrun_sca[:, 1]]

    # Attempted using OrderedDict for this, didn't work :[
    # Shouldn't make a difference python >3.6
    x = OrderedDict()
    x['rad'] = rad.astype(np.float32)
    x['spress'] = spress.astype(np.float32)
    x['cpl'] = cpl.astype(np.float32)
    
    y = nrun_tab[:, :, label_set]

    return x, y, latlon, spress_unscaled


def load_data(label_set, day, sensor):
    """
    Loader pre cpl, but using daves new file layout
    
    label_set must be either 1 or 2
    water_vapor = 2
    temperature = 1
    """

    # must be T or q
    if label_set != 1 and label_set != 2:
        return

    # Test files
    base = "/data/nature_run/fulldays_reduced_evenmore"
    data = np.load(f"{base}/{day}/mh.npy")


    if sensor == "atms":
        rad = data.copy()

        s_factors = np.load("minimac_scaling_factors_mh.npz")
        mins = s_factors['mins']
        maxs = s_factors['maxs']
        rad = np.nan_to_num(rad)
        rad = (rad - mins)/(maxs - mins)
    elif sensor == "hsel":
        rad = data.copy()

        s_factors = np.load("minimac_scaling_factors_hsel.npz")
        mins = s_factors['mins']
        maxs = s_factors['maxs']
        rad = np.nan_to_num(rad)
        rad = (rad - mins)/(maxs - mins)
    else:
        return

    nrun_sca = np.load(f"{base}/{day}/scalar.npy")
    nrun_tab = np.load(f"{base}/{day}/table.npy")

    scale_spress = np.load("spress_scalar.npz")
    mins = scale_spress['mins']
    maxs = scale_spress['maxs']
    spress_unscaled = nrun_sca[:, 3].reshape(-1, 1)
    spress = nrun_sca[:, 3].reshape(-1, 1).copy()
    spress = (spress - mins)/(maxs - mins)

    latlon = [nrun_sca[:, 0],  nrun_sca[:, 1]]

    x = {'rad': rad.astype(np.float32), 'spress': spress.astype(np.float32)}
    #x = {'rad': pd.DataFrame(rad), 'spress': pd.DataFrame(spress)}
    #x = [pd.DataFrame(rad), pd.DataFrame(spress)]

    y = nrun_tab[:, :, label_set]

    return x, y, latlon, spress_unscaled


def generate_netcdf(pred, truth, latlon, surface_pressure, cf, instr, tq, name_prefix):
    nc_path = "/data/nature_run/nc"

    fname = f"{name_prefix}.nc"
    fn = f"{nc_path}/{fname}"

    # mlflow.log_param(f"Path to {name_prefix} NetCDF", fn)

    with netCDF4.Dataset(fn, 'w', format='NETCDF4') as nc:

        num_rows = latlon[0].shape[0]

        nc.title = "Truth vs Predicted Profiles"
        nc.run_config = name_prefix # TODO


        # TODO Automate!
        nc.training_days="20060615,20061215,20060515,20060815,20060915,20060715,20061015,20060315,20061115"
        nc.test_days="20060803"
        nc.model_type=f"{instr} {tq}"

        nc.cf = str(cf)

        rows_dim = nc.createDimension('row', num_rows)

        # rows_dim.units = "Index"
        # rows_dim.long_name = "Sample Index in the data set"

        z_dim = nc.createDimension('z', pred.shape[1])
        # z_dim.units = "Sigma"
        # z_dim.long_name = "Pressure level in Sigma"

        z_true = nc.createVariable('profile_true', np.float64, ('row', 'z'))
        z_pred = nc.createVariable('profile_pred', np.float64, ('row', 'z'))
        spr = nc.createVariable('surface_pressure', np.float64, ('row'))
        lat = nc.createVariable('lat', np.float64, ('row'))
        lon = nc.createVariable('lon', np.float64, ('row'))

        lat[:] = latlon[0]
        lon[:] = latlon[1]
        spr[:] = surface_pressure[:]
        z_true[:, :] = truth
        z_pred[:, :] = pred


# Input args
# run id of best run, in mlflow
run_id = sys.argv[1]

# day in standard format like 20060803
day = sys.argv[2]

# T or q
runtype = int(sys.argv[3])

# TODO, make this auto
# literally a string "train" or "test"
traintest = sys.argv[4]

# TODO, make this auto
instr = sys.argv[5]

runtypes = {1: "T", 2: "q"}

print(f"Generating {run_id} for {runtypes[runtype]}")
# print(get_metric_names(mlflow_client, run_id))
# print(get_param_names(mlflow_client, run_id))


x, y, latlon, spress = load_data_cpl(runtype, day, instr)
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

pred = loaded_model.predict(x)
generate_netcdf(pred, y, latlon, spress, 0.1, instr, runtypes[runtype],
                f"{run_id}_{traintest}_{runtypes[runtype]}_{day}_{instr}")

########
########
# Beyond here is test code for using mlflow to query the 
# best model without needing to do it manually,
# could be useful
########
########

## Example run id
#run_id = "7c5bf86575cb4b4bac15e911d7016ce1"
#print(get_metric_names(mlflow_client, run_id))
#print(get_param_names(mlflow_client, run_id))
#
## search experiments
#experiment_list = mlflow_client.search_experiments()
#for i in experiment_list:
#    print(i.name)
#    # print(i.experiment_id)
#    # print(dir(i))
#
## experiment 0 = q
## experiment 1 = T
#
#current_experiment = 0
#current_data = 2
#
##current_experiment = 1
##current_data = 1
#
#for i in range(3):
#    print(i)
#    df = mlflow.search_runs([experiment_list[1].experiment_id],
#                            order_by=["metrics.test_loss DESC"])
#    best_run_id = df.loc[i, 'run_id']
#    #print(list(mlflow_client.get_run(best_run_id).data.to_dictionary()))
#    cf = mlflow_client.get_run(best_run_id).data.to_dictionary()["params"]["Cloud Fraction"]
#    cf = float(cf)
#    x_train, x_train_latlon, y_train, x_test, x_test_latlon, y_test = load_data(1, cf)
#
#    # logged_model = 'runs:/7c5bf86575cb4b4bac15e911d7016ce1/model'
#    # Load model as a PyFuncModel.
#    loaded_model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")
#    print(loaded_model)
#    print(x_train[0].shape, y_train.shape, x_test[0].shape, y_test.shape)
#
#    # Predict on a Pandas DataFrame.
#    train_predict = loaded_model.predict([pd.DataFrame(x_train[0]), pd.DataFrame(x_train[1])])
#    test_predict = loaded_model.predict([pd.DataFrame(x_test[0]), pd.DataFrame(x_test[1])])
#
#    #print(train_predict.shape, test_predict.shape)
#
#    generate_netcdf(train_predict, y_train, x_train_latlon, x_train[1], cf, f"{i}_train_T_20060315")
#    generate_netcdf(test_predict, y_test, x_test_latlon, x_test[1], cf, f"{i}_test_T_20060315")
#
#    #print(np.max(x_train_latlon[:, 0]), np.min(x_train_latlon[:, 0]), np.average(x_train_latlon[:, 0]))
#    #print(np.max(x_train_latlon[:, 1]), np.min(x_train_latlon[:, 1]), np.average(x_train_latlon[:, 1]))
#
#
#    df = mlflow.search_runs([experiment_list[0].experiment_id],
#                            order_by=["metrics.test_loss DESC"])
#    best_run_id = df.loc[i, 'run_id']
#    #print(list(mlflow_client.get_run(best_run_id).data.to_dictionary()))
#    cf = mlflow_client.get_run(best_run_id).data.to_dictionary()["params"]["Cloud Fraction"]
#    cf = float(cf)
#    x_train, x_train_latlon, y_train, x_test, x_test_latlon, y_test = load_data(2, cf)
#
#    # logged_model = 'runs:/7c5bf86575cb4b4bac15e911d7016ce1/model'
#    # Load model as a PyFuncModel.
#    loaded_model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")
#    print(loaded_model)
#    print(x_train[0].shape, y_train.shape, x_test[0].shape, y_test.shape)
#
#    # Predict on a Pandas DataFrame.
#    train_predict = loaded_model.predict([pd.DataFrame(x_train[0]), pd.DataFrame(x_train[1])])
#    test_predict = loaded_model.predict([pd.DataFrame(x_test[0]), pd.DataFrame(x_test[1])])
#
#    #print(train_predict.shape, test_predict.shape)
#
#    generate_netcdf(train_predict, y_train, x_train_latlon, x_train[1], cf, f"{i}_train_q_20060315")
#    generate_netcdf(test_predict, y_test, x_test_latlon, x_test[1], cf, f"{i}_test_q_20060315")
#
#    #print(np.max(x_train_latlon[:, 0]), np.min(x_train_latlon[:, 0]), np.average(x_train_latlon[:, 0]))
#    #print(np.max(x_train_latlon[:, 1]), np.min(x_train_latlon[:, 1]), np.average(x_train_latlon[:, 1]))
#
