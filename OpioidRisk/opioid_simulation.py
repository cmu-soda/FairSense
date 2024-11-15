'''
Define configuration and simulation class for opioid case study
'''

import numpy as np
from opioid_agent import OpioidAgent, OpioidAgentParams
from opioid_model import opioid_XGBoost_model, opioid_mlp_model
from opioid_env import OpioidPrescribeEnv
import opioid_env
# import opioid_experiment
import pandas as pd
import tqdm
from sklearn.metrics import *
import time

'''
A class that defines the configuration of the opioid risk experiment.
Contains information such as model dir, threshold, transition function,
sampling ratio, fairness metric requirement
'''
class OpioidRiskConfiguration():
    def __init__(self, model_class, model_dir, model_threshold_dir,
                 model_precision_dir, threshold,
                 transition_func_type, sample_ratio,
                fairness_requirement, n_time_steps, 
                n_hosp_mode, n_prescription_mode):
        self.model_class = model_class
        self.model_dir = model_dir
        self.model_threshold_dir = model_threshold_dir
        self.model_precision_dir = model_precision_dir
        self.threshold = threshold
        self.transition_func_type = transition_func_type
        self.sample_ratio = sample_ratio
        self.fairness_requirement = fairness_requirement
        self.n_time_steps = n_time_steps
        self.n_hosp_mode = n_hosp_mode
        self.n_prescription_mode = n_prescription_mode

    def __str__(self):
        # print all the parameters
        return '{'+f"model_class: {self.model_class}\n" + \
                f"model_dir: {self.model_dir}\n" + \
                f"threshold: {self.threshold}\n" + \
                f"transition_func_type: {self.transition_func_type}\n" + \
                f"sample_ratio: {self.sample_ratio}\n" + \
                f"fairness_requirement: {self.fairness_requirement}\n" + \
                f"n_time_steps: {self.n_time_steps}\n"+ \
                f"n_hosp_mode: {self.n_hosp_mode}\n"+ \
                f"n_prescription_mode: {self.n_prescription_mode}" + "}\n"

    def __repr__(self):
        return str(self)

class OpioidHistory():
    def __init__(self):
        self.t_list = []
        self.state_list = []
        self.action_list = []
        self.avg_score_list = []
        self.tpr_list = []
        self.fpr_list = []
        self.precision_list = []
        self.f1_list = []
        self.accuracy_list = []
        self.unfairness_list = []


    def record_history(self, t, state, action, intermediate_data):
        # record needed data
        self.t_list.append(t)
        # self.state_list.append(state)
        # self.action_list.append(action)
        avg_scores, tprs, fprs, precisions, f1s, accuracies = intermediate_data
        self.avg_score_list.append(avg_scores)
        self.tpr_list.append(tprs)
        self.fpr_list.append(fprs)
        self.precision_list.append(precisions)
        self.f1_list.append(f1s)
        self.accuracy_list.append(accuracies)

    def record_unfairness(self, t, unfairness_score):
        if t in self.t_list:
            self.unfairness_list.append(unfairness_score)
        else:
            raise ValueError(f"No history available for time {t}")
        
    def get_history_at_time(self, t):
        if t in self.t_list:
            t_index = self.t_list.index(t)
            return {#'state': self.state_list[t_index],
                    # 'action': self.action_list[t_index],
                    'avg_score': self.avg_score_list[t_index]
                    }
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
class OpioidMonteCarloSimulation():
    def __init__(self, configuration, init_params, stop_criteria):
        self.configuration = configuration
        self.init_params = init_params
        self.n_time_steps = configuration.n_time_steps
        stop_criteria_class = stop_criteria[0]
        stop_criteria_params = stop_criteria[1]
        self.stop_criteria = stop_criteria_class(stop_criteria_params)

    def simulation_scenario_builder(self):
        simulation_data = self.init_params['simulation_data']
        n_patient = simulation_data.shape[0]
        patient_sample_ratio = self.configuration.sample_ratio

        env_params = opioid_env.Params(patient_pool=simulation_data,
                                       n_patient=n_patient,
                                       patient_sample_ratio_mean=patient_sample_ratio)
        env = OpioidPrescribeEnv(env_params)

        model = self.configuration.model_class(self.configuration.model_dir,
                                               self.configuration.model_threshold_dir,
                                               self.configuration.model_precision_dir,
                                               self.configuration.n_hosp_mode,
                                               self.configuration.n_prescription_mode)

        agent_params = OpioidAgentParams(threshold=self.configuration.threshold)
        agent = OpioidAgent(opioid_model=model, params=agent_params, threshold=self.configuration.threshold)

        return agent, env
    def start_and_run_simulation(self, agent, env):
        '''
        Run once simulation process
        '''
        observation = env.reset()   # reset the environment
        history = OpioidHistory()
        fairness_metric = self.configuration.fairness_requirement
        # utility_metric = self.configuration.utility_metric
        decision_threshold = self.configuration.threshold
        fail_flag = False
        for t in range(self.n_time_steps):
            action = agent.act(observation, False)
            intermediate_data = fairness_metric.measure(env, agent, decision_threshold)
            history.record_history(t, env.state, action, intermediate_data)
            satisfy_flag, unfairness_score = fairness_metric.is_satisfied(history, t)
            if not satisfy_flag:
                fail_flag = True
            history.record_unfairness(t, unfairness_score)
            # utilities = utility_metric.get_utility_score(history, t)
            observation, _, done, _ = env.step(action)
        return fail_flag, history


    def repeatedly_run_simulation(self, t_limit=60):
        '''
        Repeatly run the simulation until the time limit is reached.
        Count the number of times the fairness requirement is violated.
        '''
        # create agent and env instance
        # they can be reused in the repeated simulation with reset() in each iteration
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