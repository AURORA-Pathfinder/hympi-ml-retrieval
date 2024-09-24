import numpy as np
import mlflow
import numpy as np
import netCDF4

def generate_netcdf(name: str,
                    config_name: str,
                    output_path: str,
                    pred: np.ndarray,
                    truth: np.ndarray, 
                    latlon: np.ndarray, 
                    surface_pressure: np.ndarray):

    fname = f"{name}_{config_name}.nc"
    fn = f"{output_path}/{fname}"

    mlflow.log_param(f"Path to {name} NetCDF", fn)

    with netCDF4.Dataset(fn, 'w', format='NETCDF4') as nc:

        num_rows = latlon.shape[0]

        nc.title = "Truth vs Predicted Profiles"
        nc.run_config = config_name

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

        lat[:] = latlon[:, 0]
        lon[:] = latlon[:, 1]
        spr[:] = surface_pressure[:]
        z_true[:, :] = truth
        z_pred[:, :] = pred
