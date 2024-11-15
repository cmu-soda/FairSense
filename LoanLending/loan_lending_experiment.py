'''
Code to run loan lending simulation.
'''

import numpy as np
from classes import Group, Environment, PredictiveModel, DecisionModel
import tqdm
import fico
import distribution_to_loans_outcomes as dlo
import pandas as pd
import time

'''
A class that defines the configuration of the loan lending experiment.
Contains information such as credict score updating amount, 
utility func parameter, sampling ratio, fairness metric requirement
'''
class LoanLendingConfiguration():
    def __init__(self, score_update_params, score_update_mode, util_repay_params, sample_ratio_mean, 
                sample_ratio_std, fairness_requirement, utility_metric, agent, n_time_steps=20):
        self.score_update_params = score_update_params
        self.score_update_mode = score_update_mode
        self.util_repay_params = util_repay_params
        self.sample_ratio_mean = sample_ratio_mean
        self.sample_ratio_std = sample_ratio_std
        self.fairness_requirement = fairness_requirement
        self.utility_metric = utility_metric
        self.agent = agent
        self.n_time_steps = n_time_steps

    def __str__(self):
        # print all the parameters
        return f"score_update_params: {self.score_update_params}\n" + \
                f"score_update_mode: {self.score_update_mode}\n" + \
                f"util_repay_params: {self.util_repay_params}\n" + \
                f"sample_ratio_mean: {self.sample_ratio_mean}\n" + \
                f"sample_ratio_std: {self.sample_ratio_std}\n" + \
                f"fairness_requirement: {self.fairness_requirement}\n" + \
                f"utility_metric: {self.utility_metric}\n" + \
                f"agent: {self.agent}\n" + \
                f"n_time_steps: {self.n_time_steps}\n"

'''
Define the history of one loan lending simulation process
'''
class LoanLendingHistory():
    def __init__(self, scores_list, repay_probs):
        self.thres = []
        self.avg_scores = []
        self.avg_repay_prob = []
        self.num_loans = []
        self.r_loans = []
        self.people_dis = []
        self.t_list = []
        self.unfairness_list = []

        self.n_repay = []
        self.n_default = []
        self.utility_list = []

        self.scores_list = scores_list
        self.repay_probs = repay_probs

    def record_history(self, t, thresholds, groups):
        # record needed data
        self.t_list.append(t)

        self.thres.append(thresholds)
        self.avg_scores.append([g.getMeanScore() for g in groups])
        self.avg_repay_prob.append([g.getMeanRepayProb(self.repay_probs) for g in groups])
        self.people_dis.append([g.getDistribution() for g in groups])
        
        self.avg_repay_prob.append([])
        self.num_loans.append([])
        self.r_loans.append([])
        for j in range(len(groups)):
            g = groups[j]
            self.avg_repay_prob[-1].append(g.getMeanRepayProb(self.repay_probs[j]))
            thres_index = np.where(self.scores_list >= thresholds[j])[0][0]
            n_g_get_loan = g.people[thres_index:].sum()
            r_g_get_loan = n_g_get_loan/g.total()
            self.num_loans[-1].append(n_g_get_loan)
            self.r_loans[-1].append(r_g_get_loan)

    def record_unfairness(self, t, unfairness_score):
        if t in self.t_list:
            self.unfairness_list.append(unfairness_score)
        else:
            raise ValueError(f"No history available for time {t}")
        
    def record_utility(self, t, utility_score, n_repay, n_default):
        if t in self.t_list:
            self.utility_list.append(utility_score)
            self.n_repay.append(n_repay)
            self.n_default.append(n_default)
        else:
            raise ValueError(f"No history available for time {t}")

    def get_history_at_time(self, t):
        if t in self.t_list:
            t_index = self.t_list.index(t)
            return {'thresholds': self.thres[t_index], 
                    'avg_scores': self.avg_scores[t_index], 
                    'avg_repay_prob': self.avg_repay_prob[t_index], 
                    'num_loans': self.num_loans[t_index], 
                    'r_loans': self.r_loans[t_index], 
                    'people_dis': self.people_dis[t_index]}
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
class LoanLendingMonteCarloSimulation():
    def __init__(self, configuration, init_params, stop_criteria):
        self.configuration = configuration
        self.init_params = init_params
        self.n_time_steps = configuration.n_time_steps
        stop_criteria_class = stop_criteria[0]
        stop_criteria_params = stop_criteria[1]
        self.stop_criteria = stop_criteria_class(stop_criteria_params)

    def setup_simulation(self):
        scores_list = self.init_params['scores_list']
        people_dist= self.init_params['people_dist']
        group_black = Group(people_dist[0], scores_list, r="Black")
        group_white = Group(people_dist[1], scores_list, r="White")
        env_model = Environment([group_black, group_white])

        loan_repaid_probs = self.init_params['repay_repaid_probs_func']
        util_repay = self.configuration.util_repay_params
        agent = self.configuration.agent
        pred_model = PredictiveModel(loan_repaid_probs, scores_list, util_repay, agent)
        
        score_change = self.configuration.score_update_params
        score_update_mode = self.configuration.score_update_mode
        deci_model = DecisionModel(loan_repaid_probs, scores_list, score_change, score_update_mode)

        return env_model, pred_model, deci_model
    
    def start_and_run_simulation(self):
        '''
        Run a single time 
        '''
        env_model, pred_model, deci_model = self.setup_simulation()
        sample_ratio_mean = self.configuration.sample_ratio_mean
        sample_ratio_std = self.configuration.sample_ratio_std
        people_num = [people.sum() for people in self.init_params['people_dist']]
        history = LoanLendingHistory(self.init_params['scores_list'], self.init_params['repay_probs'])
        fairness_metric = self.configuration.fairness_requirement
        utility_metric = self.configuration.utility_metric
        utility_metric.set_utility_default(self.configuration.util_repay_params[0])
        fail_flag = False
        for t in range(self.n_time_steps):
            thresholds = pred_model.computeThresholds(env_model)
            # record the current state's history
            history.record_history(t, thresholds, env_model.groups)
            satisfy_flag, unfairness_score = fairness_metric.is_satisfied(history, t)
            if not satisfy_flag:
                fail_flag = True
            history.record_unfairness(t, unfairness_score)
            # after recording, update the environment to the next state
            total_n_repay, total_n_default = env_model.sampleAndUpdate(sample_ratio_mean, sample_ratio_std,
                                      thresholds, deci_model)
            utility_score = utility_metric.computeUtilityScore(total_n_repay, total_n_default)
            history.record_utility(t, utility_score, total_n_repay, total_n_default)

        return fail_flag, history
    
    def repeatedly_run_simulation(self, t_limit=60):
        '''
        Repeatly run the simulation until the time limit is reached or the stop criteria is met.
        Count the number of times the fairness requirement is violated.
        '''
        n_fail = 0
        fail_index = []
        history_list = []
        init_time = time.time()
        stop_flag = False
        while time.time() - init_time < t_limit and not stop_flag:
            fail_flag, history = self.start_and_run_simulation()
            if fail_flag:
                n_fail += 1
                fail_index.append(len(history_list))
            history_list.append(history)
            self.stop_criteria.add_new_trace(history)
            stop_flag = self.stop_criteria.if_stop()
        return stop_flag, len(history_list), history_list, n_fail, fail_index, time.time() - init_time
    



DATA_DIR = './data/'
all_cdfs, performance, totals = fico.get_FICO_data(data_dir=DATA_DIR, do_convert_percentiles=False)
cdfs = all_cdfs[["White","Black"]]

cdf_B = cdfs['White'].values
cdf_A = cdfs['Black'].values

repay_B = performance['White']
repay_A = performance['Black']

scores = cdfs.index
scores_list = scores.tolist()
scores_repay = cdfs.index

# to populate group distributions
def get_pmf(cdf):
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size-1):
        pis[score+1] = cdf[score+1] - cdf[score]
    return pis

# to get loan repay probabilities for a given score
# loan_repaid_probs = [lambda i: repay_A[scores[scores.get_loc(i,method='nearest')]], 
#                      lambda i: repay_B[scores[scores.get_loc(i,method='nearest')]]]
def get_repay_A(i):
    return repay_A[scores[scores.get_loc(i, method='nearest')]]
def get_repay_B(i):
    return repay_B[scores[scores.get_loc(i, method='nearest')]]
loan_repaid_probs = [get_repay_A, get_repay_B]