'''
Defines the agent that predict top 50 hot spots using known crimes history.
'''
import attr
import numpy as np

class dummy_reward_fn():
    def __call__(self, observation):
        """Reward function for the idle environment."""
        del observation
        return 0

@attr.s
class PredictivePolicingAgentParams():
    """Parameters for the PredictivePolicingAgent."""
    n_hot_spot = attr.ib(default=50)
    n_related_days = attr.ib(default=170)

@attr.s
class PredictivePolicingAgent():
    """Agent that predict top 50 hot spots using known crimes history"""

    params = attr.ib()  # type: PredictivePolicingAgentParams
    rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState
    reward_fn = dummy_reward_fn()

    predict_model = attr.ib(default=None)

    _last_observation = attr.ib(default=None)
    _last_action = attr.ib(default=None)

    def act(self, obersevation, done):
        """Returns an action based on the observation."""
        reward = None
        return self._act_impl(obersevation, reward, done)

    def _act_impl(self, observation, reward, done):
        """Returns an action based on the observation."""
        del reward, done # Unused.

        # get the action from the model's prediction
        action = self.predict_model.get_action(observation,
                                               self.params.n_hot_spot,
                                               self.params.n_related_days)

        self._last_observation = observation
        self._last_action = action
        return action
