"""
This file defines several fairness metrics for loan lending case study
"""

import numpy as np

class DemographicParity():
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return "Demographic Parity: "+str(self.threshold)

    def is_satisfied(self, loan_lending_history, t):
        ratio_loans = loan_lending_history.get_history_at_time(t)['r_loans']
        max_unfairness_value = 0
        for i in range(len(ratio_loans)):
            for j in range(i+1, len(ratio_loans)):
                max_unfairness_value = max(max_unfairness_value, abs(ratio_loans[i] - ratio_loans[j]))
        if max_unfairness_value > self.threshold:
            return False, max_unfairness_value
        else:
            return True, max_unfairness_value

class BankAgentUtility():
    def __init__(self, utility_default = -2):
        self.utility_default = utility_default

    def __str__(self):
        return "Bank Agent Real Utility:" + str(self.utility_default)

    def set_utility_default(self, utility_default):
        self.utility_default = utility_default

    def computeUtilityScore(self, total_n_repay, total_n_default):
        return total_n_repay * 1 + total_n_default * self.utility_default