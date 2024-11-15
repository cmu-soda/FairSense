"""
This file defines several fairness metrics for policing case study
"""

import numpy as np
import pandas as pd

START_DATE = 2000

ALL_DISTRICTS = ['USME', 'CIUDAD BOLIVAR', 'SAN CRISTOBAL', 'RAFAEL URIBE', 'TUNJUELITO',
       'SANTA FE', 'ANTONIO NARINO', 'PUENTE ARANDA', 'LOS MARTIRES',
       'CANDELARIA', 'BOSA', 'KENNEDY', 'TEUSAQUILLO', 'CHAPINERO', 'FONTIBON',
       'ENGATIVA', 'BARRIOS UNIDOS', 'USAQUEN', 'SUBA']

class ViolatedPair():
    def __init__(self, relative_num_threshold, n_violated_pair_threshold, window_size=30):
        self.relative_num_threshold = relative_num_threshold
        self.n_violated_pair_threshold = n_violated_pair_threshold
        self.window_size = window_size

        # get all pairs of districts
        self.all_pairs = []
        for i in range(len(ALL_DISTRICTS)):
            for j in range(i+1, len(ALL_DISTRICTS)):
                self.all_pairs.append((ALL_DISTRICTS[i], ALL_DISTRICTS[j]))
    
    def __str__(self):
        return "Violated Pair: " + str(self.relative_num_threshold) + " " + str(self.n_violated_pair_threshold) + " window size" + str(self.window_size)
    
    def measure(self, env, agent, gt_n_hot_spots, police_history, t):
        # compute the relative number of hot spots for each district
        pred_hot_spots = agent._last_action[0]
        relative_number_of_hotspots_window = pd.Series(index=ALL_DISTRICTS)
        relative_number_of_hotspots_single = pd.Series(index=ALL_DISTRICTS)
        n_predicted_hotspots_at_t = pd.Series(index=ALL_DISTRICTS)
        start_date = START_DATE
        history_relative_number_of_hotspots_single = police_history.relative_num_single_df
        for d in ALL_DISTRICTS:
            # get the cells in district d
            n_predicted_hotspots_at_t[d] = len(pred_hot_spots[pred_hot_spots['district'] == d])
            window_start = t-self.window_size+1
            # first compute relative number for each day, then average over the window
            pred = n_predicted_hotspots_at_t[d]
            gt = gt_n_hot_spots.loc[t, d]
            if pred==0 and gt==0:
                relative_number_of_hotspots_single[d] = 1
            elif pred!=0 and gt==0:
                relative_number_of_hotspots_single[d] = pred * 2
            else:
                relative_number_of_hotspots_single[d] = pred / gt
            
            if len(police_history.n_pred_hot_spots_list) == 0:
                relative_number_of_hotspots_window[d] = (
                        self.window_size - 1 +\
                        relative_number_of_hotspots_single[d]) / self.window_size
            elif window_start < start_date:
                relative_number_of_hotspots_window[d] = (
                        start_date - window_start +\
                        history_relative_number_of_hotspots_single.loc[start_date:t-1, d].sum() +\
                        relative_number_of_hotspots_single[d]) / self.window_size
            else:
                relative_number_of_hotspots_window[d] = (
                        history_relative_number_of_hotspots_single.loc[window_start:t-1, d].sum() +\
                        relative_number_of_hotspots_single[d]) / self.window_size

        return relative_number_of_hotspots_window, relative_number_of_hotspots_single, n_predicted_hotspots_at_t

    def is_satisfied(self, police_history, t):
        relative_num = police_history.get_history_at_time(t)['relative_num']
        n_violated_pair = 0
        for d1, d2 in self.all_pairs:
            rel_n1 = relative_num[d1]
            rel_n2 = relative_num[d2]
            if rel_n1 == 0 and rel_n2 == 0:
                n_violated_pair += 1
            elif abs(rel_n1 - rel_n2)/(rel_n1+rel_n2) >= self.relative_num_threshold:
                n_violated_pair += 1
        if n_violated_pair >= self.n_violated_pair_threshold:
            return False, n_violated_pair
        else:
            return True, n_violated_pair
        

class AverageRelativeNumberGap():
    def __init__(self, gap_threshold, window_size=30):
        self.gap_threshold = gap_threshold
        self.window_size = window_size

        # get all pairs of districts
        self.all_pairs = []
        for i in range(len(ALL_DISTRICTS)):
            for j in range(i+1, len(ALL_DISTRICTS)):
                self.all_pairs.append((ALL_DISTRICTS[i], ALL_DISTRICTS[j]))

    def measure(self, env, agent, gt_n_hot_spots, police_history, t):
        # compute the relative number of hot spots for each district
        pred_hot_spots = agent._last_action[0]
        relative_number_of_hotspots_window = pd.Series(index=ALL_DISTRICTS)
        relative_number_of_hotspots_single = pd.Series(index=ALL_DISTRICTS)
        n_predicted_hotspots_at_t = pd.Series(index=ALL_DISTRICTS)
        start_date = START_DATE
        history_relative_number_of_hotspots_single = police_history.relative_num_single_df
        for d in ALL_DISTRICTS:
            # get the cells in district d
            n_predicted_hotspots_at_t[d] = len(pred_hot_spots[pred_hot_spots['district'] == d])
            window_start = t-self.window_size+1
            pred = n_predicted_hotspots_at_t[d]
            gt = gt_n_hot_spots.loc[t, d]
            if pred==0 and gt==0:
                relative_number_of_hotspots_single[d] = 1
            elif pred!=0 and gt==0:
                relative_number_of_hotspots_single[d] = pred * 2
            else:
                relative_number_of_hotspots_single[d] = pred / gt
            
            if len(police_history.n_pred_hot_spots_list) == 0:
                relative_number_of_hotspots_window[d] = (
                        self.window_size - 1 +\
                        relative_number_of_hotspots_single[d]) / self.window_size
            elif window_start < start_date:
                relative_number_of_hotspots_window[d] = (
                        start_date - window_start +\
                        history_relative_number_of_hotspots_single.loc[start_date:t-1, d].sum() +\
                        relative_number_of_hotspots_single[d]) / self.window_size
            else:
                relative_number_of_hotspots_window[d] = (
                        history_relative_number_of_hotspots_single.loc[window_start:t-1, d].sum() +\
                        relative_number_of_hotspots_single[d]) / self.window_size

        return relative_number_of_hotspots_window, relative_number_of_hotspots_single, n_predicted_hotspots_at_t, \
                pred_hot_spots

    def is_satisfied(self, police_history, t):
        relative_num = police_history.get_history_at_time(t)['relative_num']
        n_violated_pair = 0
        average_relative_num_gap = 0
        for d1, d2 in self.all_pairs:
            rel_n1 = relative_num[d1]
            rel_n2 = relative_num[d2]
            if rel_n1 == 0 and rel_n2 == 0:
                average_relative_num_gap += 1
            else:
                average_relative_num_gap += abs(rel_n1 - rel_n2)/(rel_n1+rel_n2)
        average_relative_num_gap /= len(self.all_pairs)
        if average_relative_num_gap >= self.gap_threshold:
            return False, average_relative_num_gap
        else:
            return True, average_relative_num_gap
        
    def __str__(self):
        return "average_relative_num_gap: " + str(self.gap_threshold)