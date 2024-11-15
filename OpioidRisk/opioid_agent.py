'''
Defines the agent that simulate doctors' behavior.
Based on ML-fairness-gym's core.Agent.
'''

# import collections
# import logging
# from typing import Any, Callable, List, Mapping, Optional, Text, Union

import attr
# import core
# import params
# from agents import threshold_policies
# import gym
import numpy as np
from sklearn import linear_model
from opioid_model import opioid_XGBoost_model

class dummy_reward_fn():
    def __call__(self, observation):
        """Reward function for the idle environment."""
        del observation
        return 0
@attr.s
class OpioidAgentParams():
    """Parameters for the OpioidAgent."""
    threshold = attr.ib(default=0.)

@attr.s
class OpioidAgent():
    """Agent that simulates doctors' behavior."""

    # TODO: params is useless?
    params = attr.ib()  # type: OpioidAgentParams
    rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState
    reward_fn = dummy_reward_fn()
    opioid_model = attr.ib(default=None)

    threshold = attr.ib(default=0.)
    # TODO: these two history are useless?
    threshold_history = attr.ib(default=None)
    target_recall_history = attr.ib(default=None)

    _step = attr.ib(default=0)
    _last_observation = attr.ib(default=None)
    _last_action = attr.ib(default=None)

    # def __init__(self, opioid_model, params):
    #     self.opioid_model = opioid_model # type: opioid_XGBoost_model
    #     self.params = params # type: OpioidAgentParams
    #     self.threshold = params.threshold

    def act(self, obervation, done):
        return self._act_impl(obervation, None, done)

    def _act_impl(self, observation, reward, done):
        """Returns an action based on the observation."""
        del reward, done # Unused.

        # get the action from the model's prediction and the threshold
        observation = observation['current_patient_features']
        action = self.opioid_model.get_action(observation, self.threshold)

        self._step += 1
        self._last_observation = observation
        self._last_action = action
        return action

