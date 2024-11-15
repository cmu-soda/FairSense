'''
Define configuration and simulation class for policing case study
'''

import numpy as np
from predict_policing_agent import PredictivePolicingAgent, PredictivePolicingAgentParams
from predict_policing_env import PredictivePolicingEnv
import predict_policing_env
from predict_hotspot_seppR_model import predict_hotspot_SEPP_R_model, get_bogota_mask
import pandas as pd
import tqdm
import time
import multiprocessing as mp
import pickle

ALL_DISTRICTS = ['USME', 'CIUDAD BOLIVAR', 'SAN CRISTOBAL', 'RAFAEL URIBE', 'TUNJUELITO',
       'SANTA FE', 'ANTONIO NARINO', 'PUENTE ARANDA', 'LOS MARTIRES',
       'CANDELARIA', 'BOSA', 'KENNEDY', 'TEUSAQUILLO', 'CHAPINERO', 'FONTIBON',
       'ENGATIVA', 'BARRIOS UNIDOS', 'USAQUEN', 'SUBA']

'''
A class that defines the configuration of the predictive policing experiment.
Contains information such as hot_spot_discover_rate, non_hot_spot_discover_rate, 
n_hot_spots, hot_spot_effect_size, fairness metric requirement
'''
class PredictPolicingConfiguration():
    def __init__(self, hot_spot_discover_rate, non_hot_spot_discover_rate, 
                 n_hot_spots, hot_spot_effect_size, fairness_requirement):
        self.hot_spot_discover_rate = hot_spot_discover_rate
        self.non_hot_spot_discover_rate = non_hot_spot_discover_rate
        self.n_hot_spots = n_hot_spots
        self.hot_spot_effect_size = hot_spot_effect_size
        self.fairness_requirement = fairness_requirement

    def __repr__(self):
        return '{'+f"hot_spot_discover_rate: {self.hot_spot_discover_rate}\n" + \
                f"non_hot_spot_discover_rate: {self.non_hot_spot_discover_rate}\n" + \
                f"n_hot_spots: {self.n_hot_spots}\n" + \
                f"hot_spot_effect_size: {self.hot_spot_effect_size}\n" + \
                f"fairness_requirement: {self.fairness_requirement}" + "}\n"
    
class PredictPolicingHistory():
    def __init__(self):
        self.t_list = []
        # self.state_list = []
        # self.action_list = []
        self.relative_num_df = pd.DataFrame(columns=ALL_DISTRICTS)
        self.relative_num_single_df = pd.DataFrame(columns=ALL_DISTRICTS)
        self.n_pred_hot_spots_list = pd.DataFrame(columns=ALL_DISTRICTS)
        self.unfairness_list = []

        # for utility calculation
        self.pred_hot_spots_list = []
        self.incident_discover_num_district = pd.DataFrame(columns=ALL_DISTRICTS)
        self.incident_discover_num_city = []
        self.incident_discover_rate_district = pd.DataFrame(columns=ALL_DISTRICTS)
        self.incident_discover_rate_city = []
        self.correct_hotspot_num_list = []
        self.known_crimes = None

    def record_history(self, t, state, action, fairness_result):
        # record needed data
        self.t_list.append(t)
        relative_num_window, relative_num_single, n_pred_hot_spots, pred_hot_spots = fairness_result
        self.relative_num_df.loc[t] = relative_num_window.values
        self.relative_num_single_df.loc[t] = relative_num_single.values
        self.n_pred_hot_spots_list.loc[t] = n_pred_hot_spots.values
        self.pred_hot_spots_list.append(pred_hot_spots)

    def record_unfairness(self, t, unfairness_score):
        if t in self.t_list:
            self.unfairness_list.append(unfairness_score)
        else:
            raise ValueError(f"No history available for time {t}")
        
    def record_num_known_crimes(self, t, state):
        if t in self.t_list:
            for d in ALL_DISTRICTS:
                self.incident_discover_num_district.loc[t, d] = len(state.current_observed_crimes[state.current_observed_crimes['district'] == d])
                if len(state.current_crimes[state.current_crimes['district'] == d]) != 0:
                    self.incident_discover_rate_district.loc[t, d] = len(state.current_observed_crimes[state.current_observed_crimes['district'] == d]) /\
                                 len(state.current_crimes[state.current_crimes['district'] == d])
            self.incident_discover_num_city.append(len(state.current_observed_crimes))
            self.incident_discover_rate_city.append(len(state.current_observed_crimes) / len(state.current_crimes))
        else:
            raise ValueError(f"No history available for time {t}")
        
    def record_correct_hotspot_num(self, t, gt_real_hot_spots):
        if t in self.t_list:
            t_index = self.t_list.index(t)
            pred_hot_spots = self.pred_hot_spots_list[t_index]
            correct_hotspot_num = len(set(pred_hot_spots['cell_id'].tolist()).intersection(set(gt_real_hot_spots['cell_id'].tolist())))
            self.correct_hotspot_num_list.append(correct_hotspot_num)
        else:
            raise ValueError(f"No history available for time {t}")
        
    def record_all_known_crimes(self, known_crimes):
        self.known_crimes = known_crimes

    def get_history_at_time(self, t):
        if t in self.t_list:
            t_index = self.t_list.index(t)
            return {'t': self.t_list[t_index],
                    'relative_num': self.relative_num_df.loc[t],
                    'relative_num_single': self.relative_num_single_df.loc[t],
                    'n_pred_hot_spots': self.n_pred_hot_spots_list.loc[t]}
        else:
            raise ValueError(f"No history available for time {t}")
   

'''
A class that decide whether continue the monte carlo simulation or not
If the std of the traces' max unfairnes is below the threshold, stop.
'''
class MaxUnfairnessStopCriteria():
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.all_unfairness = []
    
    def add_new_trace(self, history):
        # get the max unfairness of the new trace
        # then append it to the list
        unfairness_scores = history.unfairness_list
        max_unfairness = max(unfairness_scores)
        self.all_unfairness.append(max_unfairness)

    def if_stop(self):
        # compute the variance of all_unfairness
        # check if it is below the threshold
        if len(self.all_unfairness) < 10:
            return False
        else:
            return 1.96 * np.std(self.all_unfairness)/np.sqrt(len(self.all_unfairness))/np.mean(self.all_unfairness) < self.threshold
     

'''
A class that run the Monte Carlo simulation until the stopping critieria is met.
'''
class PredictivePolicingMonteCarloSimulation():
    def __init__(self, configuration, init_params, stop_criteria):
        self.configuration = configuration
        self.init_params = init_params
        self.n_time_steps = init_params['n_time_steps']
        self.start_ts = init_params['start_ts']

        stop_criteria_class = stop_criteria[0]
        stop_criteria_params = stop_criteria[1]
        self.stop_criteria = stop_criteria_class(stop_criteria_params)

        gt_real_hot_spots_path = init_params['gt_real_hot_spots_path']
        with open(gt_real_hot_spots_path, 'rb') as f:
            self.gt_real_hot_spots = pickle.load(f)

    def simulation_scenario_builder(self):
        hot_spot_discover_rate = self.configuration.hot_spot_discover_rate
        non_hot_spot_discover_rate = self.configuration.non_hot_spot_discover_rate
        n_hot_spots = self.configuration.n_hot_spots
        hot_spot_effect_size = self.configuration.hot_spot_effect_size
        n_related_days = self.init_params['n_related_days']
        model_param_path = self.init_params['model_param_path']

        # load crime data and cells
        all_crime_data = pd.read_csv(self.init_params['crime_data_path'])
        cells = get_bogota_mask()
        # init R SEPP model
        model = predict_hotspot_SEPP_R_model(model_param_path)
        # init environment
        env_param = predict_policing_env.Params(all_crimes=all_crime_data,
                                                start_ts=self.start_ts,
                                                cells=cells,
                                                hot_spot_discover_rate=hot_spot_discover_rate,
                                                non_hot_spot_discover_rate=non_hot_spot_discover_rate,
                                                hot_spot_effect_size=hot_spot_effect_size)
        env = PredictivePolicingEnv(env_param)
        # init agent
        agent_param = PredictivePolicingAgentParams(n_hot_spot=n_hot_spots, 
                                                    n_related_days=n_related_days)
        agent = PredictivePolicingAgent(predict_model=model, params=agent_param)
        
        return agent, env
        

    def start_and_run_simulation(self, agent, env):
        '''
        Run once simulation process
        '''
        observation = env.reset()   # reset the environment
        history = PredictPolicingHistory()
        fairness_metric = self.configuration.fairness_requirement
        fail_flag = False
        gt_n_hot_spots = self.init_params['gt_n_hot_spots']

        for t in range(self.start_ts, self.start_ts+self.n_time_steps):
            action = agent.act(observation, False)
            fairness_result = fairness_metric.measure(env, agent, gt_n_hot_spots, history, t)
            history.record_history(t, env.state, action, fairness_result)
            satisfy_flag, unfairness_score = fairness_metric.is_satisfied(history, t)
            if not satisfy_flag:
                fail_flag = True
            history.record_unfairness(t, unfairness_score)
            observation, _, done, _ = env.step(action)
        return fail_flag, history

    def repeatedly_run_simulation(self, t_limit=180):
        '''
        Repeatly run the simulation until the time limit is reached.
        Count the number of times the fairness requirement is violated.
        '''
        agent, env = self.simulation_scenario_builder()

        n_fail = 0
        fail_index = []
        history_list = []
        init_time = time.time()
        stop_flag = False
        while time.time() - init_time < t_limit and not stop_flag:
            fail_flag, history = self.start_and_run_simulation(agent, env)
            if fail_flag:
                n_fail += 1
                fail_index.append(len(history_list))
            history_list.append(history)
            self.stop_criteria.add_new_trace(history)
            stop_flag = self.stop_criteria.if_stop()
        return stop_flag, len(history_list), history_list, n_fail, fail_index, time.time() - init_time
    
    def repeatedly_run_simulation_parallel(self, t_limit=180, n_processes=8):
        '''
        Parallel version of repeatedly_run_simulation
        '''
        n_fail = 0
        fail_index = []
        history_list = []
        init_time = time.time()
        stop_flag = False
        pool = mp.Pool(processes=n_processes)

        while time.time() - init_time < t_limit and not stop_flag:
            results = [pool.apply_async(self.start_and_run_simulation_single) for i in range(n_processes)]
            for res in results:
                fail_flag, history = res.get()
                if fail_flag:
                    n_fail += 1
                    fail_index.append(len(history_list))
                history_list.append(history)
                self.stop_criteria.add_new_trace(history)
                
            stop_flag = self.stop_criteria.if_stop()
        pool.close()
        pool.join()
        return stop_flag, len(history_list), history_list, n_fail, fail_index, time.time() - init_time
    
    def start_and_run_simulation_single(self):
        '''
        Run once simulation process
        Only for parallel running
        '''
        agent, env = self.simulation_scenario_builder()
        observation = env.reset()   # reset the environment
        history = PredictPolicingHistory()
        fairness_metric = self.configuration.fairness_requirement
        fail_flag = False
        gt_n_hot_spots = self.init_params['gt_n_hot_spots']

        for t in range(self.start_ts, self.start_ts+self.n_time_steps):
            action = agent.act(observation, False)
            fairness_result = fairness_metric.measure(env, agent, gt_n_hot_spots, history, t)
            history.record_history(t, env.state, action, fairness_result)
            satisfy_flag, unfairness_score = fairness_metric.is_satisfied(history, t)
            if not satisfy_flag:
                fail_flag = True
            history.record_unfairness(t, unfairness_score)
            observation, _, done, _ = env.step(action)
            history.record_num_known_crimes(t, env.state)
            history.record_correct_hotspot_num(t, self.gt_real_hot_spots[t-1900])
        history.record_all_known_crimes(env.state.current_known_crimes)
        return fail_flag, history
    