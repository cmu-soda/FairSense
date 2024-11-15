'''
Defines the environment for the opioid usage disorder case study.
Based on ML-fairness-gym's template.py
'''

import copy
# from typing import Optional
import attr
# import core
# from gym import spaces
import numpy as np
import pandas as pd

def empty_patient_pool():
    """Returns an empty patient information pd.dataframe. """
    columns = ['age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8',
             'gender', 'n_hosp', 'anti_narcotic', 'narcotic',
             'n_anti_narcotic', 'n_narcotic',
             'oxymorphone', 'oxycodone', 'morphine', 'meperidine',
             'hydromorphone', 'hydrocodone', 'fentanyl', 'codeine',
             'buprenorphine', 'methadone', 'naloxone',
             'adverse_flag']
    return pd.DataFrame(columns=columns)

@attr.s
class Params():
    """Params class for the opioid use disorder environment.

    These parameters are intended to represent all potential patients' properties
    in the environment.
    """
    n_patient = attr.ib(default=0)
    patient_pool = attr.ib(default=empty_patient_pool())  # type: pd.DataFrame
    patient_sample_ratio_mean = attr.ib(default=0.01)
    patient_sample_ratio_std = attr.ib(default=0.002)


# `cmp` must be set to False here to use core.State's equality methods.
@attr.s(cmp=False)
class State():
    """State object for Opioid Prescribing Environment."""

    # Random number generator for the simulation.
    rng = attr.ib()  # type: np.random.RandomState

    # State parameters that can evolve over time.
    params = attr.ib()  # type: Params

    # The current set of patients needed to be treated.
    current_patient_features = attr.ib(default=None)  # type: pd.DataFrame
    current_patient_group = attr.ib(default=None)  # type: pd.DataFrame
    current_patient_id = attr.ib(default=None)  # type: pd.DataFrame

class _PatientFeatureUpdater():
    """Updates the patient features in the environment.
    e.g. # of hospital visits, amount of opioid prescriptions, etc.
    """

    def update(self, state, action):
        """
        Args:
            action: Contains the patient features to be updated.
            a dataframe with columns {n_hosp, n_anti_narcotic, n_narcotic, all opioid names}
        """
        patient_id = action.index
        patient_feature_changed = action.columns
        state.params.patient_pool.loc[patient_id, patient_feature_changed] += action

class _PatientSampler():
    """Samples a new set of patients."""

    def update(self, state, action):
        del action   # Unused.
        params = state.params  # type: Params
        sample_rate = np.random.normal(params.patient_sample_ratio_mean, params.patient_sample_ratio_std)
        sample_rate = min(1, max(0, sample_rate))
        sample_size = int(sample_rate * params.n_patient)
        new_patients = params.patient_pool.sample(n=sample_size)

        state.current_patient_features = new_patients.drop(columns=['adverse_flag'])
        state.current_patient_group = new_patients['gender']
        state.current_patient_id = new_patients.index

class OpioidPrescribeEnv():
    """Opioid Prescribing Environment.

    In each step, simulate the process of a patient getting opioids.
    The doctor decides whether to prescribe a medication to a patient
    based on the risk score.
    If the patient is refused, he/she will go to another doctor, until
    he/she gets the medication.

    The detail of the medication-seeking process is implemented in agent
    """

    # metadata = {'render.modes': ['human']}  # idk what this does
    default_param_builder = Params
    group_membership_var = 'gender'
    _current_patient_updater = _PatientSampler()
    _parameter_updater = _PatientFeatureUpdater()

    def __init__(self, params = None):
        if params is None:
            params = self.default_param_builder()

        # TODO(): Fill this section in with action_space. Currently, no action space
        # Use a gym.Space to describe the action space. In the example below,
        # the action space has two discrete actions.
        self.action_space = None

        patient_features_space = None
        patient_group_space = None
        patient_id_space = None

        self.observable_state_vars = {
            'current_patient_features': patient_features_space,
            'current_patient_group': patient_group_space,
            'current_patient_id': patient_id_space
        }

        # This call sets up env.initial_params and history.
        self.history = []  # type: HistoryType
        self.initial_params = copy.deepcopy(params)
        # super(OpioidPrescribeEnv, self).__init__(params)
        self._state_init()

    def _state_init(self, rng=None):
        """Initialize the environment's state."""
        self.state = State(
            rng=rng or np.random.RandomState(),
            params=copy.deepcopy(self.initial_params)
        )
        self._current_patient_updater.update(self.state, None)

    def step(self, action):
        """Overrides the step function in gym.Env."""

        # self._update_history(self.state, action)
        self.state = self._step_impl(self.state, action)
        observation = self._get_observable_state()
        reward = 0
        return observation, reward, False, {}

    def _step_impl(self, state, action):
        """Run one timestep of the environment's dynamics.

        In a single step, simulate the process of a patient getting opioids.
        The action given by agent contains the # of hospitalizations, the opioid name,
        and the amount of opioids prescribed.

        Args:
            state: A `State` object containing the current state.
            action: An action in `action_space`.

        Returns:
            A `State` object containing the updated state.
        """

        self._current_patient_updater.update(self.state, action)
        self._parameter_updater.update(self.state, action)
        return state

    def reset(self):
        """Resets the environment."""
        self._state_init(self.state.rng)
        return self._get_observable_state()

    def _get_observable_state(self):
        """Overrides the _get_observable_state function in core.FairnessEnv."""
        return {
            var_name: getattr(self.state, var_name)
            for var_name in self.observable_state_vars
        }