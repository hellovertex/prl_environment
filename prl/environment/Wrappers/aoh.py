import enum
from collections import defaultdict, deque

import numpy as np

from prl.environment.Wrappers.base import WrapperPokerRL, ActionSpace


class Positions6Max(enum.IntEnum):
    """Positions as in the literature, for a table with at most 6 Players.
    BTN for Button, SB for Small Blind, etc...
    """
    BTN = 0
    SB = 1
    BB = 2
    UTG = 3  # UnderTheGun
    MP = 4  # Middle Position
    CO = 5  # CutOff



class ActionHistory:
    # noinspection PyTypeChecker
    def __init__(self, max_players, max_actions_per_player_per_stage):
        self._max_players = max_players
        self._max_actions_per_player_per_stage = max_actions_per_player_per_stage
        self.deque = {}

        for pos in range(self._max_players):
            # create default dictionary for current player for each stage
            # default dictionary stores only the last two actions per stage per player
            self.deque[Positions6Max(pos).value] = defaultdict(
                lambda: deque(maxlen=max_actions_per_player_per_stage),
                keys=['preflop', 'flop', 'turn', 'river'])

    def buffer_actions_per_stage(self):
        return self._max_actions_per_player_per_stage

    def __str__(self):
        representation = ""
        for player_seat_id in range(self._max_players):
            representation += f'--- Player {player_seat_id}:---\n' + \
                              f'Actions preflop: {self.deque[player_seat_id]["preflop"]}\n' + \
                              f'Actions flop: {self.deque[player_seat_id]["flop"]}\n' + \
                              f'Actions turn: {self.deque[player_seat_id]["turn"]}\n' + \
                              f'Actions river: {self.deque[player_seat_id]["river"]}\n'
        return representation

    def __repr__(self):
        return self.__str__()


class ActionHistoryWrapper(WrapperPokerRL):
    """Intermediate wrapper that pushes back each action into a history buffer,
    before passing it to the AugmentObservationWrapper"""

    def __init__(self, env):
        """
        Args:
            env (PokerEnv subclass instance):   The environment instance to be wrapped
        """
        super().__init__(env=env)
        self._player_hands = []
        self._rounds = ['preflop', 'flop', 'turn', 'river']
        self._actions_per_stage = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)

        self._next_player_who_gets_observation = None
        # experimental
        self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
        self.stats_bet_size_buckets = {ActionSpace.RAISE_MIN_OR_3BB: {},
                                       ActionSpace.RAISE_HALF_POT: {},
                                       ActionSpace.RAISE_POT: {},
                                       ActionSpace.ALL_IN: {}}
    # _______________________________ Overridden ________________________________
    def _before_step(self, action):
        """
        """
        self.last_player_who_acted = self.env.current_player.seat_id
        # store action in history buffer
        self._pushback_action(action,
                              player_who_acted=self.env.current_player.seat_id,
                              in_which_stage=self.env.current_round)

    def _after_step(self, action):
        """Called before observation is computed by vectorizer"""
        if not self.done:
            self._next_player_who_gets_observation = self.env.current_player.seat_id

    def _before_reset(self, config=None):
        """Called before observation is computed by vectorizer"""
        self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
        self.stats_bet_size_buckets = {ActionSpace.RAISE_MIN_OR_3BB: {},
                                       ActionSpace.RAISE_HALF_POT: {},
                                       ActionSpace.RAISE_POT: {},
                                       ActionSpace.ALL_IN: {}}
        self.stats_bet_size_buckets = {ActionSpace.RAISE_MIN_OR_3BB: {},
                                       ActionSpace.RAISE_HALF_POT: {},
                                       ActionSpace.RAISE_POT: {},
                                       ActionSpace.ALL_IN: {}}
        self._next_player_who_gets_observation = None
        self.player_hands = []
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS)
        self.done = False # monkeypatch
        if config is not None and 'deck_state_dict' in config:
            if 'hand' in config['deck_state_dict']:
                # key 'hand' is set, when text files are parsed to vectorized observations
                # # we dont know the number of players at this point because it could have been created dynamically
                # n_players = 0
                # hands = config['deck_state_dict']['hand']
                # for h in hands:
                #     if h[0][0] != -127:
                #         n_players += 1
                #         continue
                # first = 0 if n_players < 4 else 3
                self._player_hands = config['deck_state_dict']['hand']

    def _after_reset(self):
        self._next_player_who_gets_observation = self.env.current_player.seat_id

    # _______________________________ Action History ________________________________

    def discretize(self, action_formatted):
        try:
            if isinstance(action_formatted, int) or isinstance(action_formatted, np.integer):
                return ActionSpace(action_formatted)
            if action_formatted[0] == 2:  # action is raise
                pot_size = self.env.get_all_winnable_money()
                raise_amt = action_formatted[1]
                if raise_amt < pot_size / 2:  # eval using pseudo harmonic mapping with A = 3BB, B = 1/2 Pot
                    return ActionSpace.RAISE_MIN_OR_3BB
                elif raise_amt < pot_size:  # eval using pseudo harmonic mapping with A = 1/2 pot, B = 1 Pot
                    return ActionSpace.RAISE_HALF_POT
                elif raise_amt < 2 * pot_size:  # eval using pseudo harmonic mapping with A = 1 pot, B = 2 Pot
                    return ActionSpace.RAISE_POT
                else:
                    return ActionSpace.ALL_IN  # eval using pseudo harmonic mapping with A = 2 pot, B = donk
            else:  # action is fold or check/call
                return ActionSpace(action_formatted[0])
        except Exception as e:
            print(f"ACTION TYPE = {type(action_formatted)}")
            raise e


    def _pushback_action(self, action_formatted, player_who_acted, in_which_stage):
        # part of observation
        self._actions_per_stage.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_formatted)

        action_discretized = self.discretize(action_formatted)
        # for the neural network labels
        self._actions_per_stage_discretized.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_discretized)

    # _______________________________ Override to Augment observation ________________________________
    def get_current_obs(self, env_obs, *args, **kwargs):
        """Implement this to encode Action History into observation"""
        raise NotImplementedError
