# This file is modified from github.com/lydiatliu/delayedimpact
# Please follow the license below.

# BSD 3-Clause License

# Copyright (c) 2018, lydiatliu
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Compute and Visualize outcome curves"""

import numpy as np
import solve_credit as sc


def get_thresholds(loan_repaid_probs, pis, group_size_ratio, utils, score_change_fns, scores):
    """get thresholds for all 3 fairness criteria"""
    # unpacking bank utilities
    utility_default, utility_repaid = utils
    break_even_prob = utility_default / (utility_default - utility_repaid)

    # getting loan policies
    data = sc.fairness_data(get_rhos(loan_repaid_probs, scores), pis,
                            group_size_ratio, break_even_prob)
    tau_maxprof = data.max_profit_loans()
    tau_eqopp = data.equal_opp_loans()

    thresh_maxprof = get_thresholds_from_taus(tau_maxprof, scores)
    thresh_eqopp = get_thresholds_from_taus(tau_eqopp, scores)

    return thresh_eqopp, thresh_maxprof


def get_shared_thresholds(loan_repaid_probs, pis, group_size_ratio, utils, score_change_fns, scores):
    """get shared thresholds for max-util fairness criteria"""
    # unpacking bank utilities
    utility_default, utility_repaid = utils
    break_even_prob = utility_default / (utility_default - utility_repaid)

    repay_A, repay_B = get_rhos(loan_repaid_probs, scores)
    ratio_A, ratio_B = group_size_ratio
    pi_A, pi_B = pis
    performance = repay_A*(pi_A*ratio_A/(pi_A*ratio_A+pi_B*ratio_B)) + repay_B*(pi_B*ratio_B/(pi_A*ratio_A+pi_B*ratio_B)) 

    # getting loan policies
    data = sc.fairness_data(performance, pis,
                            group_size_ratio, break_even_prob)
    tau_maxprof = [data.max_profit_loans()]

    thresh_maxprof = get_thresholds_from_taus(tau_maxprof, scores)

    return [thresh_maxprof[0], thresh_maxprof[0]]


def get_outcome_curve(loan_repay_fn, pi, scores, impacts):
    """compute the outcome curve"""
    delta_mu = np.zeros(scores.size)
    delta_mu[0] = exp_move(scores[-1], loan_repay_fn, bounds=[300, 850],
                           move_vec=impacts) * pi[-1]
    for i in range(1, scores.size):
        delta_mu[i] = exp_move(scores[-(i + 1)], loan_repay_fn, bounds=[300, 850],
                               move_vec=impacts) * pi[-(i + 1)] + delta_mu[i - 1]

    return delta_mu


def get_utility_curve(loan_repay_fns, pis, scores, utils):
    """compute the institution's utility curve"""
    util = np.zeros([2, scores.size])
    for j in range(2):
        util[j, 0] = bank_util(scores[-1], loan_repay_fns[j], utils=utils) * pis[j, -1]
        for i in range(1, scores.size):
            util[j, i] = bank_util(scores[-(i + 1)], loan_repay_fns[j], utils=utils) * \
                         pis[j, -(i + 1)] + util[j, i - 1]
    return util


def get_utility_curves_dempar(util, cdfs, group_size_ratio, scores):
    """compute the institution's utility curve under demographic parity constraint"""
    util_total = np.zeros([2, scores.size])
    cdfs[0] = list(reversed(1 - cdfs[0]))
    cdfs[1] = list(reversed(1 - cdfs[1]))

    for i_a in range(scores.size):
        prop_a = cdfs[0, i_a]
        i_b = find_nearest(cdfs[1], prop_a)
        util_total[0, i_a] = group_size_ratio[0] * util[0, i_a] + group_size_ratio[1] * util[1, i_b]

    for i_b in range(scores.size):
        prop_b = cdfs[1, i_b]
        i_a = find_nearest(cdfs[0], prop_b)
        util_total[1, i_b] = group_size_ratio[0] * util[0, i_a] + group_size_ratio[1] * util[1, i_b]

    return util_total


def get_utility_curves_eqopp(util, loan_repaid_probs, pis, group_size_ratio, scores):
    """compute the institution's utility curve under equal opportunity constraint"""
    rescaled_pis = np.zeros(pis.shape)
    n_groups, n_scores = pis.shape
    for group in range(n_groups):
        for x in range(n_scores):
            rescaled_pis[group, x] = pis[group, x] * loan_repaid_probs[group](scores[x])
        rescaled_pis[group] = rescaled_pis[group] / np.sum(rescaled_pis[group])

    cdfs = np.zeros(pis.shape)
    for group in range(n_groups):
        cdfs[group, 0] = rescaled_pis[group, 0]
        for x in range(1, n_scores):
            cdfs[group, x] = rescaled_pis[group, x] + cdfs[group, x - 1]
    return get_utility_curves_dempar(util, cdfs, group_size_ratio, scores)


def get_rhos(loan_repaid_probs, scores):
    """set up rhos[i,j] = probability of group i member repaying loan at state j"""
    n_scores = len(scores)
    n_groups = len(loan_repaid_probs)
    rhos = np.zeros((n_groups, n_scores))
    for j, s in enumerate(scores):
        for i in range(n_groups):
            rhos[i, j] = loan_repaid_probs[i](s)
    return rhos


def get_mean_movement_threshold(pi, scores, loan_repay_fn, score_change_fns, compare_pt=0.0):
    """randomized threshold below which group's mean score will decrease,
    above which it increases"""
    running_sum = 0
    for i, s in enumerate(reversed(scores)):
        x = len(scores) - 1 - i
        if ((running_sum + pi[x] * exp_move(s, loan_repay_fn, move_vec=score_change_fns))
                < compare_pt):
            randomized = (compare_pt - running_sum) / (exp_move(s, loan_repay_fn,
                                                                move_vec=score_change_fns) * pi[x])
            assert randomized >= 0
            assert randomized <= 1
            if i == 0:
                return s
            return s + randomized * abs(scores[x] - scores[x + 1])
        else:
            running_sum += pi[x] * exp_move(s, loan_repay_fn)
    return s


def exp_move(x, loan_repay_fn, bounds=[300, 850], move_vec=[-150, 75]):
    """compute expected move in score"""
    move_down = move_vec[0]
    move_up = move_vec[1]
    move = (1 - loan_repay_fn(x)) * move_down + loan_repay_fn(x) * move_up
    move_to_within_bounds = max(min(x + move, bounds[1]), bounds[0])
    move_within_bounds = move_to_within_bounds - x
    return move_within_bounds


def bank_util(x, loan_repay_fn, utils):
    """bank's expected utility"""
    util_repay = utils[1]
    util_def = utils[0]
    return (1 - loan_repay_fn(x)) * util_def + loan_repay_fn(x) * util_repay


def get_thresholds_from_taus(taus, scores):
    """Turn loaning policies into thresholds for visualization"""
    thresholds = []
    for tau in taus:
        x = np.amin(np.where(tau > 0))
        # to visualize the randomization
        if x<len(tau)-1:
            thresholds.append(
                scores[x] + (1 - tau[x]) * np.abs(scores[x + 1] - scores[x]))
        else:
            thresholds.append(np.array(scores[x]))
    return thresholds


def find_nearest(array, value):
    """find closest value"""
    idx = (np.abs(array - value)).argmin()
    return int(np.amin(idx))
