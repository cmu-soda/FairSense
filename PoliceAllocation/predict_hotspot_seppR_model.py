'''
We are using the SEPP model written in R language, which is provided by the original paper
We use rpy2 to run R function to call the SEPP model
'''

import os
import rpy2.situation
os.environ["R_HOME"] = rpy2.situation.get_r_home()

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import Converter
from rpy2.robjects.packages import importr

# from predict_policing_experiment import get_hotspot_from_crimes

class predict_hotspot_SEPP_R_model():
    # this class is the wrapper of the SEPP model written in R
    def __init__(self, model_param_path):
        self.model_param_path = model_param_path
        self.load_param()
        self.init_R_predict_func()
        self.cell_length = 1
        self.init_backgroundintensity()
        

    def load_param(self):
        # load the model parameters of the last optimization iteration
        model_param = pd.read_csv(self.model_param_path)
        n_iter = len(model_param)-1
        self.model_param = {}
        self.model_param['mu_bar'] = float(model_param.mu_bar[n_iter])
        self.model_param['theta'] = float(model_param.theta[n_iter])
        self.model_param['omega'] = float(model_param.omega[n_iter])
        self.model_param['sigma_x'] = float(model_param.sigma_x[n_iter])
        self.model_param['sigma_y'] = float(model_param.sigma_y[n_iter])


    def init_R_predict_func(self):
        ro.r['source']('utils/real_bogota_utils.R')
        self.get_aftershock_integrals_R = ro.globalenv['get_aftershock_integrals']

    def init_backgroundintensity(self):
        mask = get_bogota_mask(self.cell_length)
        bogota_mask_help = mask[mask['district'] != '0']

        get_background_integrals_R = ro.globalenv['get_background_integrals']
        center_R = ro.DataFrame({'x': 0, 'y': 0})
        background_sd = 15
        res_R = get_background_integrals_R(center_R, self.cell_length, 
                                                        self.model_param['mu_bar'], background_sd)
        with (ro.default_converter + pandas2ri.converter).context():
            res = ro.conversion.get_conversion().rpy2py(res_R)

        # merge the background intensity with the mask
        self.bogota_mask_help = pd.merge(bogota_mask_help, res, how='left')
        self.bogota_mask_help = self.bogota_mask_help.rename(columns={'superset_background_int': 'background_int'})

    def get_intensities(self, rel_data, ts):
        # call the R function to get the aftershock intensities
        # integrate background intensity and aftershock intensity to get the total intensity
        with (ro.default_converter + pandas2ri.converter).context():
            rel_data_R = ro.conversion.get_conversion().py2rpy(rel_data)
        sds_R = ro.FloatVector((self.model_param['sigma_x'], self.model_param['sigma_y']))

        res_R = self.get_aftershock_integrals_R(rel_data_R, ts, 
                                                sds_R, float(self.model_param['theta']), 
                                                float(self.model_param['omega']), self.cell_length)
        with (ro.default_converter + pandas2ri.converter).context():
            res = ro.conversion.get_conversion().rpy2py(res_R)
        res = res.rename(columns={'superset_aftershock_int': 'aftershock_int'})

        res_integral = pd.merge(self.bogota_mask_help, res, how='left')
        res_integral['intensity'] = res_integral.background_int + res_integral.aftershock_int

        return res_integral
    
    def predict_hot_spot(self, rel_data, ts, n_hot_spot=50):
        # predict the top 50 hot spot based on intensity
        # return the cell id of the hot spot
        res_integral = self.get_intensities(rel_data, ts)
        hot_spot = res_integral.sort_values(by='intensity', ascending=False)
        hot_spot = hot_spot.iloc[:n_hot_spot]
        return hot_spot, res_integral

    def get_action(self, observation, n_hot_spot=50, n_related_days=170):
        current_known_crimes = observation['current_known_crimes']
        ts = observation['current_ts']
        rel_data = current_known_crimes[(current_known_crimes['t'] <= ts) &
                                        (ts-current_known_crimes['t'] <= n_related_days)]
        hot_spot, res_integral = self.predict_hot_spot(rel_data, ts, n_hot_spot)
        return hot_spot, res_integral



def get_bogota_mask(cell_length=1):
    bogota_mask = pd.read_csv('metadata/bogota_mask_'+str(cell_length)+'.csv')
    # add one more column to indicate the cell id
    bogota_mask['cell_id'] = bogota_mask.index+1
    return bogota_mask

