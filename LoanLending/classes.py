import numpy as np
import distribution_to_loans_outcomes as dlo

# profit and impact
utility_repaid_1 = 1
utility_default_1 = -10
utility_repaid_2 = 1
utility_default_2 = -10
score_change_repay = 13.5
score_change_default = -27

# considering several utility ratios to understand sensitivity of qualitative results
util_repay = [utility_default_1,utility_repaid_1]
impact = [score_change_default,score_change_repay]

'''
class Group represents the group information and contains the related functions
'''
class Group:
    def __init__(self, p, score, r="unknown"):
        assert len(p) == len(score)
        self.people = p.copy()
        self.scores_list = score
        self.n_score = len(self.scores_list)
        self.race = r

    def change(self, old, new):
        # old and new are also Group class instances
        # update old group of people to new group 
        assert old.total() == new.total()
        new_people = self.people - old.people + new.people
        assert (new_people >= 0).all()
        self.people = new_people.astype(int)

    def getDistribution(self):
        # turn number of people into percentage of people
        return self.people/self.total()

    def getIndividuals(self):
        # turn aggregative people number into individuals, which is used to sample
        ret = np.array([i for i in range(self.n_score) for j in range(self.people[i])])
        return ret

    def sample(self, size):
        # sample 'size' people from self.people
        # return a new group which only contains the people sampled out
        assert size >= 0 and size <= self.total()
        s_individual = np.random.choice(self.getIndividuals(), size, replace=False)
        s_statistic = np.zeros(self.n_score)
        for i in range(self.n_score):
            s_statistic[i] = (s_individual==i).sum()
        return Group(s_statistic, self.scores_list, r=self.race)
    
    def sample_gaussian(self, mean_sample_rate=0.05, std=0.01):
        # given a gaussian distribution of how many people to sample
        # return a new group which only contains the people sampled out
        sample_rate = np.random.normal(mean_sample_rate, std)
        sample_rate = max(0, min(sample_rate, 1))
        size = np.round(sample_rate*self.total()).astype(int)
        return self.sample(size)

    def total(self):
        return self.people.sum()

    def getMeanScore(self):
        return (self.people*self.scores_list).sum()/self.total()
    
    def getMeanRepayProb(self, repay_prob):
        mean_repay_prob = (self.people*repay_prob).sum()/self.total()
        return mean_repay_prob

'''
class Environment contains the groups in the environment
'''
class Environment:
    def __init__(self, groups = None):
        self.groups = groups
        if self.groups == None:
            self.groups = []

    def addGroups(self, groups):
        self.groups = self.groups + groups

    def sampleAndUpdate(self, sample_ratio_mean, sample_ratio_std, thresholds, dm):
        # sample people and then decide giving loan or not and update credit scores 
        sample_groups = []
        for i in range(len(self.groups)):
            sample_groups.append(self.groups[i].sample_gaussian(sample_ratio_mean, sample_ratio_std))
        updated_groups, total_n_repay, total_n_default = dm.decideAndUpdateIfRepayLoanGaussian(sample_groups, thresholds)
        for i in range(len(self.groups)):
            self.groups[i].change(sample_groups[i], updated_groups[i])

        return total_n_repay, total_n_default # return the total number of repay and default for real utility calculation

    def getGroupSizeRatio(self):
        ret = np.array([g.total() for g in self.groups])
        return ret/ret.sum()


'''
class PredictiveModel represents the agent's prediction for threshold
'''
class PredictiveModel:
    def __init__(self, loan_repaid_probs, score, util_repay, agent='eqopp'):
        self.loan_repaid_probs = loan_repaid_probs  # a func to correspond credit score to repay probablity
        self.scores_list = score
        self.util_repay = util_repay
        self.agent = agent
        assert self.agent in ['eqopp', 'maxprof']
        
    def computeThresholds(self, env):
        # given an Environment, compute the threshold for each group in it
        pis = np.vstack([env.groups[0].getDistribution(), env.groups[1].getDistribution()])

        # the return value of dlo.get_thresholds is (thresh_dempar, thresh_eqopp, thresh_maxprof, thresh_downwards)
        # currently only return the one we need
        thresholds = dlo.get_thresholds(self.loan_repaid_probs, pis, env.getGroupSizeRatio(),
                                                self.util_repay, impact, self.scores_list)
        if self.agent == 'eqopp':
            self.thresholds = thresholds[0]
        elif self.agent == 'maxprof':
            self.thresholds = thresholds[1]
        return self.thresholds

    def computeSharedThreshold(self, env):
        # given the env, compute a shared threshold using maxutil agent
        pis = np.vstack([env.groups[0].getDistribution(), env.groups[1].getDistribution()])

        # the return value of dlo.get_thresholds is (thresh_dempar, thresh_eqopp, thresh_maxprof, thresh_downwards)
        # currently only return the one we need
        self.thresholds = dlo.get_shared_thresholds(self.loan_repaid_probs, pis, env.getGroupSizeRatio(),
                                                self.util_repay, impact, self.scores_list)
        return self.thresholds


'''
class DecisionModel contains the functions that make decisions based
on thresholds and update credit score
'''
class DecisionModel:
    def __init__(self, loan_repaid_probs, score, score_change_params, update_mode='equal'):
        self.loan_repaid_probs = loan_repaid_probs  # a func to correspond credit score to repay probablity
        self.scores_list = np.array(score)
        self.score_change_params = score_change_params
        assert update_mode in ['equal', 'small_var', 'large_var']
        self.update_mode = update_mode
    
    def decideAndUpdateIfRepayLoanGaussian(self, groups, thresholds):
        # Decide whether give loan or not, and then update credit score.
        # Who repay the loan increase credit score, who default decrease
        # the amount of change is determined by score_change_params and update_mode
        # if update_mode == 'equal', then everyone change the same amount as score_change_params
        # if update_mode == 'small_var', then everyone change a random amount from normal distribution 
        #   with mean score_change_params and std 0.1*score_change_params
        # if update_mode == 'large_var', then everyone change a random amount from normal distribution
        #   with mean score_change_params and std 0.2*score_change_params
        # Otherwise, not change
        score_change_repay = self.score_change_params[0]
        score_change_default = self.score_change_params[1]

        ret = []
        total_n_repay, total_n_default = 0, 0  # record the repay and default number for real utility calculation
        for i in range(len(groups)):
            group = groups[i]
            new_people = np.zeros_like(group.people)
            thres_index = np.where(self.scores_list >= thresholds[i])[0][0]
            new_people[:thres_index] = group.people[:thres_index]   # before the index, everyone keep the same
            for j in range(thres_index, len(self.scores_list)):
                cur_score = self.scores_list[j]
                repay_prob = self.loan_repaid_probs[i](cur_score)
                n_repay = int((np.random.rand(int(group.people[j]))<=repay_prob).sum())
                n_default = int(group.people[j] - n_repay)
                total_n_repay += n_repay
                total_n_default += n_default
                if self.update_mode == 'equal':
                    new_score_repay_index = findNearScoreIndex(self.scores_list, cur_score + score_change_repay)
                    new_people[new_score_repay_index] += n_repay
                    new_score_defaulte_index = findNearScoreIndex(self.scores_list, cur_score + score_change_default)
                    new_people[new_score_defaulte_index] += n_default
                elif self.update_mode == 'small_var' or self.update_mode == 'large_var':
                    if self.update_mode == 'small_var':
                        coeff = 0.1
                    elif self.update_mode == 'large_var':
                        coeff = 0.2
                    delta_score_repay = np.random.normal(score_change_repay, abs(coeff*score_change_repay), size=n_repay)
                    delta_score_repay = np.clip(delta_score_repay, 0, 2*score_change_repay)
                    new_score_repay_indeces = findNearScoreIndex(self.scores_list, cur_score + delta_score_repay)
                    unique_indices, counts = np.unique(new_score_repay_indeces, return_counts=True)
                    new_people[unique_indices] += counts

                    delta_score_default = np.random.normal(score_change_default, abs(coeff*score_change_default), size=n_default)
                    delta_score_default = np.clip(delta_score_default, 2*score_change_default, 0)
                    new_score_default_indeces = findNearScoreIndex(self.scores_list, cur_score + delta_score_default)
                    unique_indices, counts = np.unique(new_score_default_indeces, return_counts=True)
                    new_people[unique_indices] += counts
                else:
                    raise ValueError("update_mode should be one of 'equal', 'small_var', 'large_var'")
            ret.append(Group(new_people.astype(int), group.scores_list, group.race))
            assert ret[i].total() == groups[i].total()

        return ret, total_n_repay, total_n_default
        

def findNearScoreIndex(scores, s):
    idx = np.abs(scores[:, np.newaxis] - s).argmin(axis=0)
    return idx
