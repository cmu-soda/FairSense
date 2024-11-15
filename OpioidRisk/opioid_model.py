'''
The opioid model class that loads in the pre-trained model and
implements the interaction functions for agent class.
'''

import numpy as np
import pickle
import pandas as pd
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class opioid_XGBoost_model():
    # opioid xgboost model interface class
    # self.model type: xgb.XGBClassifier
    def __init__(self, model_dir, threshold_dir, precision_dir, n_hosp_mode='expectation', n_prescription_mode='expectation'):
        self.model_dir = model_dir
        self.threshold_dir = threshold_dir
        self.precision_dir = precision_dir
        self.n_hosp_mode = n_hosp_mode
        self.n_prescription_mode = n_prescription_mode

        # Load in the pre-trained model and related information
        self.load_model(model_dir)
        self.load_threshold(threshold_dir)
        self.load_precision(precision_dir)
        assert self.thresholds.shape == self.precisions.shape

    def load_model(self, model_dir):
        # load model from model_dir using pickle
        with open(model_dir, 'rb') as f:
            self.model = pickle.load(f)

    def load_threshold(self, threshold_dir):
        # load threshold from threshold_dir using pickle
        with open(threshold_dir, 'rb') as f:
            self.thresholds = np.array(pickle.load(f))

    def load_precision(self, precision_dir):
        # load tpr from tpr_dir using pickle
        with open(precision_dir, 'rb') as f:
            self.precisions = np.array(pickle.load(f))

    def predict(self, observation):
        # observation is a pandas dataframe with the same columns as the training data
        # return a numpy array of predicted probabilities
        columns = ['age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'gender',
                   'n_hosp', 'anti_narcotic', 'narcotic', 'n_anti_narcotic', 'n_narcotic',
                   'oxymorphone', 'oxycodone', 'morphine', 'meperidine', 'hydromorphone',
                   'hydrocodone', 'fentanyl', 'codeine', 'buprenorphine', 'methadone',
                   'naloxone']
        X = observation[columns]
        # print("#### in predict():")
        # print(X.shape, type(X), X.dtypes)
        return self.model.predict_proba(X)[:,1]

    def get_action(self, observation, threshold):
        # Return an action which contains the patient features to be updated
        # It is a dataframe with columns {n_hosp, n_anti_narcotic, n_narcotic, all opioid names}
        n_patients = observation.shape[0]
        ret = pd.DataFrame(index=observation.index, columns=['n_hosp', 'n_anti_narcotic', 'n_narcotic',
                                                            'oxymorphone', 'oxycodone', 'morphine',
                                                            'meperidine', 'hydromorphone', 'hydrocodone',
                                                            'fentanyl', 'codeine', 'buprenorphine',
                                                            'methadone', 'naloxone'], dtype='int64')
        ret[:] = 0
        # first get the predicted probabilities
        prob = self.predict(observation)

        # simulate the number of hospital visits
        # ret['n_hosp'] = 4
        ret['n_hosp'][prob < threshold] = 1
        ret['n_hosp'][prob >= threshold] = get_expectation_gaussian(prob[prob >= threshold], self.thresholds, self.precisions, mode=self.n_hosp_mode)

        # sample the number of anti-narcotic and narcotic prescriptions
        n_prescription = get_n_prescription_gaussian(observation, mode=self.n_prescription_mode)
        # ret[n_prescription.columns][prob >= threshold] = n_prescription[n_prescription.columns][prob >= threshold]
        ret[n_prescription.columns] = n_prescription[n_prescription.columns]

        return ret

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.activations = nn.ModuleList()
        for i in range(hidden_layers):
            self.activations.append(nn.ReLU())
        self.activations.append(nn.Sigmoid())

    def forward(self, x):
        out = x
        for i in range(self.hidden_layers + 1):
            out = self.layers[i](out)
            # if i != self.hidden_layers:
            out = self.activations[i](out)
        return out
class opioid_mlp_model(opioid_XGBoost_model):
    def load_model(self, model_dir):
        # load model from model_dir
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = MLP(input_size=24, output_size=1,
                hidden_layers=8,
                hidden_size=64).to(self.device)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()

    def load_threshold(self, threshold_dir):
        # load threshold from threshold_dir using pickle
        with open(threshold_dir, 'rb') as f:
            self.thresholds = np.array(pickle.load(f))

    def load_precision(self, precision_dir):
        # load tpr from tpr_dir using pickle
        with open(precision_dir, 'rb') as f:
            self.precisions = np.array(pickle.load(f))

    def predict(self, observation):
        # observation is a pandas dataframe with the same columns as the training data
        # return a numpy array of predicted probabilities
        columns = ['age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'gender',
                   'n_hosp', 'anti_narcotic', 'narcotic', 'n_anti_narcotic', 'n_narcotic',
                   'oxymorphone', 'oxycodone', 'morphine', 'meperidine', 'hydromorphone',
                   'hydrocodone', 'fentanyl', 'codeine', 'buprenorphine', 'methadone',
                   'naloxone']
        X = observation[columns]
        index = observation.index
        with torch.no_grad():
            X_torch = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            predicted_Y = self.model(X_torch).cpu().numpy()
        ret_Y = pd.Series(predicted_Y.reshape(-1), index=index)
        return ret_Y



def get_expectation(scores, thresholds, precisions, mode='expectation'):
    assert mode in ['expectation', 'same', 'random_1', 'random_2']

    # find the corresponding tpr for each score using thresholds
    cols = np.argmax(scores[:, np.newaxis] <= thresholds[np.newaxis, :], axis=1)
    cols[cols > 0] -= 1
    probs = 1-(precisions[cols]+scores)/2
    # return the expected number of times to get 1
    epision = 1e-3
    num_flips = 1 / (probs+epision)

    # add randomness
    if mode == 'same':
        num_flips = 1
    elif mode == 'random_1':
        num_flips = np.random.randint(np.maximum(np.round(num_flips/2), 1), np.maximum(num_flips+2, np.round(num_flips*1.5)))
    elif mode == 'random_2':
        num_flips = np.random.randint(2, 10, size=num_flips.shape)

    # # turn num_flips into integer
    num_flips = np.round(num_flips)
    return num_flips

def get_expectation_gaussian(scores, thresholds, precisions, mode='expectation'):
    assert mode in ['expectation', 'same', 'random_1', 'random_2']

    # find the corresponding tpr for each score using thresholds
    cols = np.argmax(scores[:, np.newaxis] <= thresholds[np.newaxis, :], axis=1)
    cols[cols > 0] -= 1
    probs = 1-(precisions[cols]+scores)/2
    # return the expected number of times to get 1
    epision = 1e-3
    num_flips = 1 / (probs+epision)

    # add randomness
    if mode == 'same':
        num_flips = 1
    elif mode == 'random_1' or mode == 'random_2':
        coeff = 0.1 if mode == 'random_1' else 0.25
        num_flips = np.random.normal(num_flips, coeff*num_flips)
        num_flips = np.clip(num_flips, 1, 50)

    # # turn num_flips into integer
    num_flips = np.round(num_flips)

    return num_flips

def get_n_prescription(observation, mode='expectation'):
    assert mode in ['expectation', 'random_1', 'random_2']
    ret = pd.DataFrame(index=observation.index, columns=['n_anti_narcotic', 'n_narcotic',
                                                         'oxymorphone', 'oxycodone', 'morphine',
                                                         'meperidine', 'hydromorphone', 'hydrocodone',
                                                         'fentanyl', 'codeine', 'buprenorphine',
                                                         'methadone', 'naloxone'], dtype='int64')
    ret[:] = 0
    opioids = ['oxymorphone', 'oxycodone', 'morphine', 'meperidine', 'hydromorphone',
                'hydrocodone', 'fentanyl', 'codeine', 'buprenorphine', 'methadone', 'naloxone']

    props_narcotic = observation['n_narcotic'] / (observation['n_narcotic'] + observation['n_anti_narcotic'])
    avg_n_opioid = np.round((observation['n_narcotic'] + observation['n_anti_narcotic']) / observation['n_hosp'])

    if mode == 'random_1':
        avg_n_opioid = pd.Series(np.random.randint(np.maximum(np.round(avg_n_opioid/2), 1), np.maximum(avg_n_opioid+2, np.round(avg_n_opioid*1.5))),
                                    index=avg_n_opioid.index)
                                    
    elif mode == 'random_2':
        avg_n_opioid = pd.Series(np.random.randint(np.maximum(np.round(avg_n_opioid*0.8), 1), np.maximum(avg_n_opioid+2, np.round(avg_n_opioid*1.2))),
                                    index=avg_n_opioid.index)

    # decide which kind of prescription to sample
    if_narcotic = np.random.binomial(1, props_narcotic)

    # Divide patients into two parts based on if_narcotic
    narcotic_patients = observation[if_narcotic == 1]
    anti_narcotic_patients = observation[if_narcotic == 0]

    # Deal with narcotic patients
    # get the probabilities for each narcotic
    if narcotic_patients.shape[0] > 0:
        probs = narcotic_patients[opioids[:-1]].values / \
                narcotic_patients['n_narcotic'].values[:, np.newaxis]
        # Calculate the cumulative sum for each row
        cum_probs = np.cumsum(probs, axis=1)
        # Generate a random number between 0 and 1 for each row
        rand_nums = np.random.random(narcotic_patients.shape[0])
        # Find the first column index where the cumulative sum is greater than or equal to the random number
        col_indices = np.argmax(rand_nums[:, np.newaxis] <= cum_probs, axis=1)
        # Get the corresponding opioid name in pandas
        col_names = narcotic_patients[opioids[:-1]].columns[col_indices]
        # Go through each row and assign the number of prescriptions to the corresponding opioid
        for i in range(narcotic_patients.shape[0]):
            ret.loc[narcotic_patients.index[i], col_names[i]] = avg_n_opioid[narcotic_patients.index[i]]
        ret.loc[narcotic_patients.index, 'n_narcotic'] = avg_n_opioid[narcotic_patients.index]

    # do the similar thing for anti-narcotic
    if anti_narcotic_patients.shape[0] > 0:
        probs = anti_narcotic_patients[opioids[-2:]].values / \
                anti_narcotic_patients['n_anti_narcotic'].values[:, np.newaxis]
        cum_probs = np.cumsum(probs, axis=1)
        rand_nums = np.random.random(anti_narcotic_patients.shape[0])
        col_indices = np.argmax(rand_nums[:, np.newaxis] <= cum_probs, axis=1)
        col_names = anti_narcotic_patients[opioids[-2:]].columns[col_indices]
        for i in range(anti_narcotic_patients.shape[0]):
            ret.loc[anti_narcotic_patients.index[i], col_names[i]] = avg_n_opioid[anti_narcotic_patients.index[i]]
        ret.loc[anti_narcotic_patients.index, 'n_anti_narcotic'] = avg_n_opioid[anti_narcotic_patients.index]

    # completely randomly pick an opioid
    # col_indices = np.random.randint(0, len(opioids), size=ret.shape[0])
    # col_names = ret.columns[col_indices]
    # for i in range(ret.shape[0]):
    #     ret.loc[ret.index[i], col_names[i]] = avg_n_opioid[ret.index[i]]
    #     if col_names[i] in opioids[:-2]:
    #         ret.loc[ret.index[i], 'n_narcotic'] = avg_n_opioid[ret.index[i]]
    #     if col_names[i] in opioids[-2:]:
    #         ret.loc[ret.index[i], 'n_anti_narcotic'] = avg_n_opioid[ret.index[i]]

    return ret


def get_n_prescription_gaussian(observation, mode='expectation'):
    assert mode in ['expectation', 'random_1', 'random_2']
    ret = pd.DataFrame(index=observation.index, columns=['n_anti_narcotic', 'n_narcotic',
                                                         'oxymorphone', 'oxycodone', 'morphine',
                                                         'meperidine', 'hydromorphone', 'hydrocodone',
                                                         'fentanyl', 'codeine', 'buprenorphine',
                                                         'methadone', 'naloxone'], dtype='int64')
    ret[:] = 0
    opioids = ['oxymorphone', 'oxycodone', 'morphine', 'meperidine', 'hydromorphone',
                'hydrocodone', 'fentanyl', 'codeine', 'buprenorphine', 'methadone', 'naloxone']

    props_narcotic = observation['n_narcotic'] / (observation['n_narcotic'] + observation['n_anti_narcotic'])
    avg_n_opioid = (observation['n_narcotic'] + observation['n_anti_narcotic']) / observation['n_hosp']
    avg_index = avg_n_opioid.index

    if mode == 'random_1' or mode == 'random_2':
        coeff = 0.1 if mode == 'random_1' else 0.25
        avg_n_opioid = np.random.normal(avg_n_opioid, coeff*avg_n_opioid)
        avg_n_opioid = np.clip(avg_n_opioid, 1, 50)
        avg_n_opioid = pd.Series(avg_n_opioid, index=avg_index)

        
    avg_n_opioid = np.round(avg_n_opioid)

    # decide which kind of prescription to sample
    if_narcotic = np.random.binomial(1, props_narcotic)

    # Divide patients into two parts based on if_narcotic
    narcotic_patients = observation[if_narcotic == 1]
    anti_narcotic_patients = observation[if_narcotic == 0]

    # Deal with narcotic patients
    # get the probabilities for each narcotic
    if narcotic_patients.shape[0] > 0:
        probs = narcotic_patients[opioids[:-1]].values / \
                narcotic_patients['n_narcotic'].values[:, np.newaxis]
        # Calculate the cumulative sum for each row
        cum_probs = np.cumsum(probs, axis=1)
        # Generate a random number between 0 and 1 for each row
        rand_nums = np.random.random(narcotic_patients.shape[0])
        # Find the first column index where the cumulative sum is greater than or equal to the random number
        col_indices = np.argmax(rand_nums[:, np.newaxis] <= cum_probs, axis=1)
        # Get the corresponding opioid name in pandas
        col_names = narcotic_patients[opioids[:-1]].columns[col_indices]
        # Go through each row and assign the number of prescriptions to the corresponding opioid
        for i in range(narcotic_patients.shape[0]):
            ret.loc[narcotic_patients.index[i], col_names[i]] = avg_n_opioid[narcotic_patients.index[i]]
        ret.loc[narcotic_patients.index, 'n_narcotic'] = avg_n_opioid[narcotic_patients.index]

    # do the similar thing for anti-narcotic
    if anti_narcotic_patients.shape[0] > 0:
        probs = anti_narcotic_patients[opioids[-2:]].values / \
                anti_narcotic_patients['n_anti_narcotic'].values[:, np.newaxis]
        cum_probs = np.cumsum(probs, axis=1)
        rand_nums = np.random.random(anti_narcotic_patients.shape[0])
        col_indices = np.argmax(rand_nums[:, np.newaxis] <= cum_probs, axis=1)
        col_names = anti_narcotic_patients[opioids[-2:]].columns[col_indices]
        for i in range(anti_narcotic_patients.shape[0]):
            ret.loc[anti_narcotic_patients.index[i], col_names[i]] = avg_n_opioid[anti_narcotic_patients.index[i]]
        ret.loc[anti_narcotic_patients.index, 'n_anti_narcotic'] = avg_n_opioid[anti_narcotic_patients.index]

    return ret







