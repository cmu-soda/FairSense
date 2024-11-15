'''
Defines the environment for the predictive policing case study.
Based on ML-fairness-gym's template.py
'''

import copy
# from typing import Optional
import attr
# import core
# from gym import spaces
import numpy as np
import pandas as pd

@attr.s
class Params():
    """Params class for Predictive Policing Environment.

    It is intended to contain the ground truth crime data.
    """
    all_crimes = attr.ib(default=None)  # type: pd.DataFrame
    # The first time step that apply the predictive policing system
    start_ts = attr.ib(default=0)  # type: int
    cells = attr.ib(default=None)  # type: pd.DataFrame
    hot_spot_discover_rate = attr.ib(default=0.8)  # type: float
    non_hot_spot_discover_rate = attr.ib(default=0.2)  # type: float
    hot_spot_effect_size = attr.ib(default=1)  # type: int

@attr.s(cmp=False)
class State():
    """State object for Predictive Policing Environment."""

    # Random number generator for the simulation.
    rng = attr.ib()  # type: np.random.RandomState

    # State parameters that can evolve over time.
    params = attr.ib()  # type: Params

    # Current time step
    current_ts = attr.ib(default=0)  # type: int
    # The set of all crimes that are committed within the current time step
    current_crimes = attr.ib(default=None)  # type: pd.DataFrame
    # The set of hot spots that are identified at the current time step
    current_hot_spots = attr.ib(default=None)  # type: dataframe of cell ids
    current_pred_intensity = attr.ib(default=None)  # type: dataframe of cell intensity
    # The set of observed crimes with the police allocation at the current time step
    current_observed_crimes = attr.ib(default=None)  # type: pd.DataFrame
    # The history of all observed/known crimes by agent
    current_known_crimes = attr.ib(default=None)  # type: pd.DataFrame

class _CrimeUpdater():
    """Updates the current crime data in the environment."""
    def update(self, state, action):
        del action
        params = state.params
        ts = state.current_ts
        state.current_crimes = params.all_crimes[params.all_crimes.t == ts]
        
class _ObservedCrimesUpdater():
    """
    Get the observed crimes based on the police allocation
    action here is the predicted top 50 hot spots, which represents the police allocation
    """
    def update(self, state, action):
        # sample the observed crimes based on discover rate of the cell
        # crimes in hot spot cell is more likely to be observed
        params = state.params
        hot_spot_discover_rate = params.hot_spot_discover_rate
        non_hot_spot_discover_rate = params.non_hot_spot_discover_rate
        current_crimes = state.current_crimes
        cells = params.cells
        # action_cells is the list of tuple for each pair of x, y in dataframe action
        hot_spot, cell_intensity = action
        action_cells = list(zip(hot_spot.x, hot_spot.y))
        effect_cells = action_cells.copy()
        effect_size = params.hot_spot_effect_size
        for x,y in action_cells:
            effect_cells += [(x+i, y+j) for i in range(1-effect_size, effect_size)
                             for j in range(1-effect_size, effect_size)]
        effect_cells = list(set(effect_cells))
        state.current_hot_spots = hot_spot.copy()
        state.current_pred_intensity = cell_intensity.copy()

        observed_crimes = pd.DataFrame(columns=current_crimes.columns)
        for _, cell in cells.iterrows():
            cell_x = cell.x
            cell_y = cell.y
            # get the crimes in the cell
            # crimes_in_cell = current_crimes[(cell_x + 1 > current_crimes['x'] >= cell_x) &
            #                                 (cell_y + 1 > current_crimes.y >= cell_y)]
            crimes_in_cell = current_crimes[(current_crimes['x'] <= cell_x) &
                                            (current_crimes['x'] > cell_x - 1) &
                                            (current_crimes['y'] <= cell_y) &
                                            (current_crimes['y'] > cell_y - 1)]

            # sample the observed crimes based on discover rate of the cell
            if (cell_x, cell_y) in effect_cells:
                crimes_in_cell['is_observed'] = np.random.binomial(1, hot_spot_discover_rate, len(crimes_in_cell))
            else:
                crimes_in_cell['is_observed'] = np.random.binomial(1, non_hot_spot_discover_rate, len(crimes_in_cell))
            # add the observed crimes to the observed crimes set
            observed_crimes = pd.concat([observed_crimes, crimes_in_cell[crimes_in_cell['is_observed'] == 1]])
        # observed_crimes = current_crimes
        state.current_observed_crimes = observed_crimes


class _KnownCrimesHistoryUpdater():
    """
    Updates the observed/known crimes
    """
    def update(self, state, action):
        del action
        params = state.params
        ts = state.current_ts
        if ts == -1:
            state.current_ts = params.start_ts
            state.current_known_crimes = params.all_crimes[params.all_crimes.t < state.current_ts]
            state.current_observed_crimes = state.current_known_crimes[state.current_known_crimes.t == state.current_ts-1]
        else:
            state.current_known_crimes = pd.concat([state.current_known_crimes,
                                                    state.current_observed_crimes])


class PredictivePolicingEnv():
    """
    Predictive Policing Environment.

    In each step, simulate the effect of hot spot police allocation.
    Based on the police allocation from the agent, get the observed crimes.
    """

    default_params = Params
    _gt_crime_updater = _CrimeUpdater()
    _observed_crimes_updater = _ObservedCrimesUpdater()
    _known_crimes_history_updater = _KnownCrimesHistoryUpdater()

    def __init__(self, params=None):
        if params is None:
            params = self.default_param_builder()

        self.action_space = None
        time_step_space = None
        observed_crimes_space = None
        known_crimes_space = None
        self.observable_state_vars = {
            'current_ts': time_step_space,
            'current_observed_crimes': observed_crimes_space,
            'current_known_crimes': known_crimes_space
        }

        self.history = []  
        self.initial_params = copy.deepcopy(params)
        self._state_init()

    def _state_init(self, rng=None):
        """Initialize the environment's state."""
        self.state = State(
            rng=rng or np.random.RandomState(),
            params=copy.deepcopy(self.initial_params)
        )
        self.state.current_ts = -1
        self._known_crimes_history_updater.update(self.state, None)

    def step(self, action):
        """Overrides the step function in gym.Env."""
        self._step_impl(action)
        observation = self._get_observable_state()
        reward = 0
        return observation, reward, False, {}

    def _step_impl(self, action):
        """Run one timestep of the environment's dynamics.

        In a single step, the police allocation would take effect.
        The new observed crimes would be got based on the police allocation.

        action here is the predicted top 50 hot spots, which represents the police allocation
        """
        self._gt_crime_updater.update(self.state, action)
        self._observed_crimes_updater.update(self.state, action)
        self._known_crimes_history_updater.update(self.state, action)
        self.state.current_ts += 1
        return self.state

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
