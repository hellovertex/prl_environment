import enum
from typing import Optional
import gym
import numpy as np
from gym import spaces
from collections import defaultdict, deque

from prl.environment.Wrappers.aoh import ActionHistoryWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import CanonicalVectorizer, AgentObservationType
from prl.environment.steinberger.PokerRL.game.Poker import Poker
from gym.spaces import Box


# noinspection DuplicatedCode
class AugmentObservationWrapper(ActionHistoryWrapper):
    """Runs our custom vectorizer after computing the observation from the steinberger env"""

    def __init__(self, env,
                 disable_info=False,
                 agent_observation_mode=AgentObservationType.CARD_KNOWLEDGE):
        super().__init__(env=env)
        self.disable_info = disable_info
        self.agent_observation_mode = agent_observation_mode
        # todo: (?) check how obs is normalized to avoid small floats
        self.normalization = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = env.N_SEATS
        self.max_players = 6
        self.num_board_cards = 5
        _, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        obs_space = Box(low=0.0, high=6.0, shape=(569,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            'obs': obs_space,
            # do not change key-name 'obs' it is internally used by rllib (!)
            'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
            # one-hot encoded [FOLD, CHECK_CALL, RAISE]
        })
        self.action_space = gym.spaces.Discrete(ActionSpace.__len__())
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS,
                                               mode=self.agent_observation_mode)
        a = 1

    #
    # def seed(self, seed: Optional[int] = None) -> None:
    #     np.random.seed(seed)

    def overwrite_args(self, args, agent_observation_mode=None, n_players=None):
        if n_players:
            self.num_players = n_players
        self.env.set_args(args)
        if not agent_observation_mode:
            agent_observation_mode = self.agent_observation_mode
        self.normalization = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = self.env.N_SEATS
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS,
                                               mode=agent_observation_mode)
        self._vectorizer.agent_observation_mode = agent_observation_mode

    def get_legal_moves_extended(self):
        legal_moves = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        # if done
        legal_moves[self.env.get_legal_actions()] += 1
        if legal_moves[2] == 1:
            legal_moves[[3, 4, 5, 6, 7]] = 1
        return legal_moves

    def agent_observation_mode(self):
        return self._vectorizer.agent_observation_mode

    def set_agent_observation_mode(self, mode: AgentObservationType):
        self._vectorizer.agent_observation_mode = mode

    # @override
    def get_current_obs(self, env_obs, backward_offset=0):
        """
        Args:
            env_obs: the observation returned by the base PokerEnv.
            The len(env_obs) is a function of the number of players.
        """
        observer_relative_index = (self._next_player_who_gets_observation -
                                          backward_offset) % self.num_players
        obs = self._vectorizer.vectorize(env_obs,
                                         observer_relative_index,
                                         action_history=self._actions_per_stage,
                                         player_hands=self._player_hands,
                                         # player_hands=[self.env.get_hole_cards_of_player(i)
                                         #               for i in range(self.num_players)],
                                         normalization=self.normalization,
                                         done=self.done)
        return obs

    def random_agent(self):
        legal_moves = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        legal_moves[self.env.get_legal_actions()] += 1
        if legal_moves[2] == 1:
            legal_moves[[3, 4, 5, 6, 7]] = 1
        # return self.env.get_legal_actions()
        return legal_moves

    def get_info(self):
        return self.env.get_info()

    def _construct_obs_space(self):
        """
        The maximum all chip-values can reach is n_seats, because we normalize by dividing by the average starting stack
        """
        obs_idx_dict = {}
        obs_parts_idxs_dict = {
            "board": [],
            "players": [[] for _ in range(self.max_players)],
            "table_state": [],
            "player_cards": [[] for _ in range(self.max_players)],
            "action_history": [[] for _ in range(len(self._rounds))]
        }
        next_idx = [0]  # list is a mutatable object. int not.

        def get_discrete(size, name, _curr_idx):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Discrete(size)

        def get_new_box(name, _curr_idx, high, low=0):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # __________________________  Public Information About Game State  _________________________
        _k = next_idx[0]
        _table_space = [
            # (blinds are in obs to give the agent a perspective on starting stack after normalization
            get_new_box("ante", next_idx, self.max_players),
            # .................................... self.ANTE
            get_new_box("small_blind", next_idx, self.max_players),
            # ............................. self.SMALL_BLIND
            get_new_box("big_blind", next_idx, self.max_players),
            # ............................... self.BIG_BLIND
            get_new_box("min_raise", next_idx, self.max_players),
            # ............................... min total raise
            get_new_box("pot_amt", next_idx, self.max_players),
            # ................................. main_pot amount
            get_new_box("total_to_call", next_idx, self.max_players),
            # ........................... total_to_call
            # get_new_box("last_action_how_much", next_idx, self.max_players),  # .................... self.last_action[1]
        ]
        # for i in range(3):  # .................................................................. self.last_action[0]
        #   _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))
        #
        # for i in range(self.max_players):  # ....................................................... self.last_action[2]
        #   _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

        for i in range(
                self.max_players):  # ....................................................... curr_player.seat_id
            _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

        for i in range(
                len(self._rounds)):  # ...................................... round onehot
            _table_space.append(
                get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

        for i in range(
                self.max_players):  # ....................................................... side pots
            _table_space.append(get_new_box("side_pot_" + str(i), next_idx, 1))

        # add to parts_dict for possible slicing for agents.
        obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

        # __________________________  Public Information About Each Player  ________________________
        _player_space = []
        for i in range(self.max_players):
            _k = next_idx[0]
            _player_space += [
                get_new_box("stack_p" + str(i), next_idx, self.max_players),
                # ..................... stack
                get_new_box("curr_bet_p" + str(i), next_idx, self.max_players),
                # .................. current_bet
                get_discrete(1, "has_folded_this_episode_p" + str(i), next_idx),
                # ............. folded_this_epis
                get_discrete(1, "is_allin_p" + str(i), next_idx),
                # ............................ is_allin
            ]
            for j in range(self.max_players):
                _player_space.append(
                    get_discrete(1, "side_pot_rank_p" + str(i) + "_is_" + str(j),
                                 next_idx))  # . side_pot_rank

            # add to parts_dict for possible slicing for agents
            obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

        # _______________________________  Public cards (i.e. board)  ______________________________
        _board_space = []
        _k = next_idx[0]
        for i in range(self.num_board_cards):

            x = []
            for j in range(
                    self.env.N_RANKS):  # .................................................... rank
                x.append(
                    get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

            for j in range(
                    self.env.N_SUITS):  # .................................................... suit
                x.append(
                    get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

            _board_space += x

        # add to parts_dict for possible slicing for agents
        obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

        # _______________________________  Private Cards (i.e. players hands)  ______________________________
        # add to parts_dict for possible slicing for agents
        _handcards_space = []

        for i in range(self.max_players):
            _k = next_idx[0]
            for k in range(self.env.N_HOLE_CARDS):
                x = []
                for j in range(
                        self.env.N_RANKS):  # .................................................... rank
                    x.append(
                        get_discrete(1, str(i) + f"th_player_card_{k}_rank_" + str(j),
                                     next_idx))

                for j in range(
                        self.env.N_SUITS):  # .................................................... suit
                    x.append(
                        get_discrete(1, str(i) + f"th_player_card_{k}_suit_" + str(j),
                                     next_idx))

                _handcards_space += x

            obs_parts_idxs_dict["player_cards"][i] += list(range(_k, next_idx[0]))
        # _______________________________  Action History (max 2 /stage/player)  ______________________________
        # add to parts_dict for possible slicing for agents
        # preflop_player_0_action_0_how_much
        # preflop_player_0_action_0_what_0
        # preflop_player_0_action_0_what_1
        # preflop_player_0_action_0_what_2
        # preflop_player_0_action_1_how_much
        # preflop_player_0_action_1_what_0
        # preflop_player_0_action_1_what_1
        # preflop_player_0_action_1_what_2
        _action_history_space = []

        for i in range(len(self._rounds)):
            _k = next_idx[0]
            for j in range(self.max_players):
                for a in [0, 1]:
                    _action_history_space.append(
                        get_new_box(f"{self._rounds[i]}_player_{j}_action_{a}_how_much",
                                    next_idx, self.max_players))
                    for k in range(3):
                        _action_history_space.append(
                            get_discrete(1,
                                         f"{self._rounds[i]}_player_{j}_action_{a}_what_{k}",
                                         next_idx)
                        )
            obs_parts_idxs_dict["action_history"][i] += list(range(_k, next_idx[0]))
        # preflop, flop, turn, river : [action0, action1], []

        # __________________________  Return Complete _Observation Space  __________________________
        # Tuple (lots of spaces.Discrete and spaces.Box)
        _observation_space = spaces.Tuple(_table_space +
                                          _player_space +
                                          _board_space +
                                          _handcards_space +
                                          _action_history_space +
                                          [spaces.Discrete(1)]
                                          # for btn_index which we added manually later
                                          )
        # obs_idx_dict["btn_idx"] = len(list(obs_idx_dict.keys()))
        obs_idx_dict["Btn_idx_is_0"] = 563
        obs_idx_dict["Btn_idx_is_1"] = 564
        obs_idx_dict["Btn_idx_is_2"] = 565
        obs_idx_dict["Btn_idx_is_3"] = 566
        obs_idx_dict["Btn_idx_is_4"] = 567
        obs_idx_dict["Btn_idx_is_5"] = 568
        try:
            _observation_space.shape = [len(_observation_space.spaces)]
        except AttributeError:
            # newer version of gym dont allow setting shape attr
            pass
        return _observation_space, obs_idx_dict, obs_parts_idxs_dict

    def print_augmented_obs(self, obs):
        """Can be used for debugging. Printed Observation example:
        ______________________________________ Printing _Observation _________________________________________
                              ante:   0.0
                       small_blind:   0.0032992411870509386
                         big_blind:   0.006598482374101877
                         min_raise:   0.013196964748203754
                           pot_amt:   0.0
                     total_to_call:   0.006598482374101877
                      p0_acts_next:   1.0
                      p1_acts_next:   0.0
                      p2_acts_next:   0.0
                      p3_acts_next:   1.0
                      p4_acts_next:   0.0
                      p5_acts_next:   0.0
                     round_preflop:   1.0
                        round_flop:   0.0
                        round_turn:   0.0
                       round_river:   0.0
                        side_pot_0:   0.0
                        side_pot_1:   0.0
                        side_pot_2:   0.0
                        side_pot_3:   0.0
                        side_pot_4:   0.0
                        side_pot_5:   0.0
                          stack_p0:   1.106103539466858
                       curr_bet_p0:   0.0
        has_folded_this_episode_p0:   0.0
                       is_allin_p0:   0.0
             side_pot_rank_p0_is_0:   0.0
             side_pot_rank_p0_is_1:   0.0
             side_pot_rank_p0_is_2:   0.0
             side_pot_rank_p0_is_3:   0.0
             side_pot_rank_p0_is_4:   0.0
             side_pot_rank_p0_is_5:   0.0
                          stack_p1:   0.5709006786346436
                       curr_bet_p1:   0.0032992411870509386
        has_folded_this_episode_p1:   0.0
                       is_allin_p1:   0.0
             side_pot_rank_p1_is_0:   0.0
             side_pot_rank_p1_is_1:   0.0
             side_pot_rank_p1_is_2:   0.0
             side_pot_rank_p1_is_3:   0.0
             side_pot_rank_p1_is_4:   0.0
             side_pot_rank_p1_is_5:   0.0
                          stack_p2:   1.3130979537963867
                       curr_bet_p2:   0.006598482374101877
        has_folded_this_episode_p2:   0.0
                       is_allin_p2:   0.0
             side_pot_rank_p2_is_0:   0.0
             side_pot_rank_p2_is_1:   0.0
             side_pot_rank_p2_is_2:   0.0
             side_pot_rank_p2_is_3:   0.0
             side_pot_rank_p2_is_4:   0.0
             side_pot_rank_p2_is_5:   0.0
                          stack_p3:   1.106103539466858
                       curr_bet_p3:   0.0
        has_folded_this_episode_p3:   0.0
                       is_allin_p3:   0.0
             side_pot_rank_p3_is_0:   0.0
             side_pot_rank_p3_is_1:   0.0
             side_pot_rank_p3_is_2:   0.0
             side_pot_rank_p3_is_3:   0.0
             side_pot_rank_p3_is_4:   0.0
             side_pot_rank_p3_is_5:   0.0
                          stack_p4:   0.5709006786346436
                       curr_bet_p4:   0.0032992411870509386
        has_folded_this_episode_p4:   0.0
                       is_allin_p4:   0.0
             side_pot_rank_p4_is_0:   0.0
             side_pot_rank_p4_is_1:   0.0
             side_pot_rank_p4_is_2:   0.0
             side_pot_rank_p4_is_3:   0.0
             side_pot_rank_p4_is_4:   0.0
             side_pot_rank_p4_is_5:   0.0
                          stack_p5:   1.3130979537963867
                       curr_bet_p5:   0.006598482374101877
        has_folded_this_episode_p5:   0.0
                       is_allin_p5:   0.0
             side_pot_rank_p5_is_0:   0.0
             side_pot_rank_p5_is_1:   0.0
             side_pot_rank_p5_is_2:   0.0
             side_pot_rank_p5_is_3:   0.0
             side_pot_rank_p5_is_4:   0.0
             side_pot_rank_p5_is_5:   0.0
             0th_board_card_rank_0:   0.0
             0th_board_card_rank_1:   0.0
             0th_board_card_rank_2:   0.0
             0th_board_card_rank_3:   0.0
             0th_board_card_rank_4:   0.0
             0th_board_card_rank_5:   0.0
             0th_board_card_rank_6:   0.0
             0th_board_card_rank_7:   0.0
             0th_board_card_rank_8:   0.0
             0th_board_card_rank_9:   0.0
            0th_board_card_rank_10:   0.0
            0th_board_card_rank_11:   0.0
            0th_board_card_rank_12:   0.0
             0th_board_card_suit_0:   0.0
             0th_board_card_suit_1:   0.0
             0th_board_card_suit_2:   0.0
             0th_board_card_suit_3:   0.0
             1th_board_card_rank_0:   0.0
             1th_board_card_rank_1:   0.0
             1th_board_card_rank_2:   0.0
             1th_board_card_rank_3:   0.0
             1th_board_card_rank_4:   0.0
             1th_board_card_rank_5:   0.0
             1th_board_card_rank_6:   0.0
             1th_board_card_rank_7:   0.0
             1th_board_card_rank_8:   0.0
             1th_board_card_rank_9:   0.0
            1th_board_card_rank_10:   0.0
            1th_board_card_rank_11:   0.0
            1th_board_card_rank_12:   0.0
             1th_board_card_suit_0:   0.0
             1th_board_card_suit_1:   0.0
             1th_board_card_suit_2:   0.0
             1th_board_card_suit_3:   0.0
             2th_board_card_rank_0:   0.0
             2th_board_card_rank_1:   0.0
             2th_board_card_rank_2:   0.0
             2th_board_card_rank_3:   0.0
             2th_board_card_rank_4:   0.0
             2th_board_card_rank_5:   0.0
             2th_board_card_rank_6:   0.0
             2th_board_card_rank_7:   0.0
             2th_board_card_rank_8:   0.0
             2th_board_card_rank_9:   0.0
            2th_board_card_rank_10:   0.0
            2th_board_card_rank_11:   0.0
            2th_board_card_rank_12:   0.0
             2th_board_card_suit_0:   0.0
             2th_board_card_suit_1:   0.0
             2th_board_card_suit_2:   0.0
             2th_board_card_suit_3:   0.0
             3th_board_card_rank_0:   0.0
             3th_board_card_rank_1:   0.0
             3th_board_card_rank_2:   0.0
             3th_board_card_rank_3:   0.0
             3th_board_card_rank_4:   0.0
             3th_board_card_rank_5:   0.0
             3th_board_card_rank_6:   0.0
             3th_board_card_rank_7:   0.0
             3th_board_card_rank_8:   0.0
             3th_board_card_rank_9:   0.0
            3th_board_card_rank_10:   0.0
            3th_board_card_rank_11:   0.0
            3th_board_card_rank_12:   0.0
             3th_board_card_suit_0:   0.0
             3th_board_card_suit_1:   0.0
             3th_board_card_suit_2:   0.0
             3th_board_card_suit_3:   0.0
             4th_board_card_rank_0:   0.0
             4th_board_card_rank_1:   0.0
             4th_board_card_rank_2:   0.0
             4th_board_card_rank_3:   0.0
             4th_board_card_rank_4:   0.0
             4th_board_card_rank_5:   0.0
             4th_board_card_rank_6:   0.0
             4th_board_card_rank_7:   0.0
             4th_board_card_rank_8:   0.0
             4th_board_card_rank_9:   0.0
            4th_board_card_rank_10:   0.0
            4th_board_card_rank_11:   0.0
            4th_board_card_rank_12:   0.0
             4th_board_card_suit_0:   0.0
             4th_board_card_suit_1:   0.0
             4th_board_card_suit_2:   0.0
             4th_board_card_suit_3:   0.0
          0th_player_card_0_rank_0:   0.0
          0th_player_card_0_rank_1:   0.0
          0th_player_card_0_rank_2:   0.0
          0th_player_card_0_rank_3:   0.0
          0th_player_card_0_rank_4:   0.0
          0th_player_card_0_rank_5:   1.0
          0th_player_card_0_rank_6:   0.0
          0th_player_card_0_rank_7:   0.0
          0th_player_card_0_rank_8:   0.0
          0th_player_card_0_rank_9:   0.0
         0th_player_card_0_rank_10:   0.0
         0th_player_card_0_rank_11:   0.0
         0th_player_card_0_rank_12:   0.0
           0th_board_card_0_suit_0:   0.0
           0th_board_card_0_suit_1:   0.0
           0th_board_card_0_suit_2:   0.0
           0th_board_card_0_suit_3:   1.0
          0th_player_card_1_rank_0:   0.0
          0th_player_card_1_rank_1:   0.0
          0th_player_card_1_rank_2:   0.0
          0th_player_card_1_rank_3:   0.0
          0th_player_card_1_rank_4:   0.0
          0th_player_card_1_rank_5:   1.0
          0th_player_card_1_rank_6:   0.0
          0th_player_card_1_rank_7:   0.0
          0th_player_card_1_rank_8:   0.0
          0th_player_card_1_rank_9:   0.0
         0th_player_card_1_rank_10:   0.0
         0th_player_card_1_rank_11:   0.0
         0th_player_card_1_rank_12:   0.0
           0th_board_card_1_suit_0:   1.0
           0th_board_card_1_suit_1:   0.0
           0th_board_card_1_suit_2:   0.0
           0th_board_card_1_suit_3:   0.0
          1th_player_card_0_rank_0:   0.0
          1th_player_card_0_rank_1:   0.0
          1th_player_card_0_rank_2:   0.0
          1th_player_card_0_rank_3:   0.0
          1th_player_card_0_rank_4:   0.0
          1th_player_card_0_rank_5:   0.0
          1th_player_card_0_rank_6:   0.0
          1th_player_card_0_rank_7:   0.0
          1th_player_card_0_rank_8:   0.0
          1th_player_card_0_rank_9:   0.0
         1th_player_card_0_rank_10:   0.0
         1th_player_card_0_rank_11:   0.0
         1th_player_card_0_rank_12:   1.0
           1th_board_card_0_suit_0:   1.0
           1th_board_card_0_suit_1:   0.0
           1th_board_card_0_suit_2:   0.0
           1th_board_card_0_suit_3:   0.0
          1th_player_card_1_rank_0:   0.0
          1th_player_card_1_rank_1:   0.0
          1th_player_card_1_rank_2:   0.0
          1th_player_card_1_rank_3:   0.0
          1th_player_card_1_rank_4:   0.0
          1th_player_card_1_rank_5:   0.0
          1th_player_card_1_rank_6:   0.0
          1th_player_card_1_rank_7:   0.0
          1th_player_card_1_rank_8:   0.0
          1th_player_card_1_rank_9:   1.0
         1th_player_card_1_rank_10:   0.0
         1th_player_card_1_rank_11:   0.0
         1th_player_card_1_rank_12:   0.0
           1th_board_card_1_suit_0:   0.0
           1th_board_card_1_suit_1:   1.0
           1th_board_card_1_suit_2:   0.0
           1th_board_card_1_suit_3:   0.0
          2th_player_card_0_rank_0:   1.0
          2th_player_card_0_rank_1:   0.0
          2th_player_card_0_rank_2:   0.0
          2th_player_card_0_rank_3:   0.0
          2th_player_card_0_rank_4:   0.0
          2th_player_card_0_rank_5:   0.0
          2th_player_card_0_rank_6:   0.0
          2th_player_card_0_rank_7:   0.0
          2th_player_card_0_rank_8:   0.0
          2th_player_card_0_rank_9:   0.0
         2th_player_card_0_rank_10:   0.0
         2th_player_card_0_rank_11:   0.0
         2th_player_card_0_rank_12:   0.0
           2th_board_card_0_suit_0:   1.0
           2th_board_card_0_suit_1:   0.0
           2th_board_card_0_suit_2:   0.0
           2th_board_card_0_suit_3:   0.0
          2th_player_card_1_rank_0:   1.0
          2th_player_card_1_rank_1:   0.0
          2th_player_card_1_rank_2:   0.0
          2th_player_card_1_rank_3:   0.0
          2th_player_card_1_rank_4:   0.0
          2th_player_card_1_rank_5:   0.0
          2th_player_card_1_rank_6:   0.0
          2th_player_card_1_rank_7:   0.0
          2th_player_card_1_rank_8:   0.0
          2th_player_card_1_rank_9:   0.0
         2th_player_card_1_rank_10:   0.0
         2th_player_card_1_rank_11:   0.0
         2th_player_card_1_rank_12:   0.0
           2th_board_card_1_suit_0:   1.0
           2th_board_card_1_suit_1:   0.0
           2th_board_card_1_suit_2:   0.0
           2th_board_card_1_suit_3:   0.0
          3th_player_card_0_rank_0:   0.0
          3th_player_card_0_rank_1:   0.0
          3th_player_card_0_rank_2:   0.0
          3th_player_card_0_rank_3:   0.0
          3th_player_card_0_rank_4:   0.0
          3th_player_card_0_rank_5:   1.0
          3th_player_card_0_rank_6:   0.0
          3th_player_card_0_rank_7:   0.0
          3th_player_card_0_rank_8:   0.0
          3th_player_card_0_rank_9:   0.0
         3th_player_card_0_rank_10:   0.0
         3th_player_card_0_rank_11:   0.0
         3th_player_card_0_rank_12:   0.0
           3th_board_card_0_suit_0:   0.0
           3th_board_card_0_suit_1:   0.0
           3th_board_card_0_suit_2:   0.0
           3th_board_card_0_suit_3:   1.0
          3th_player_card_1_rank_0:   0.0
          3th_player_card_1_rank_1:   0.0
          3th_player_card_1_rank_2:   0.0
          3th_player_card_1_rank_3:   0.0
          3th_player_card_1_rank_4:   0.0
          3th_player_card_1_rank_5:   1.0
          3th_player_card_1_rank_6:   0.0
          3th_player_card_1_rank_7:   0.0
          3th_player_card_1_rank_8:   0.0
          3th_player_card_1_rank_9:   0.0
         3th_player_card_1_rank_10:   0.0
         3th_player_card_1_rank_11:   0.0
         3th_player_card_1_rank_12:   0.0
           3th_board_card_1_suit_0:   1.0
           3th_board_card_1_suit_1:   0.0
           3th_board_card_1_suit_2:   0.0
           3th_board_card_1_suit_3:   0.0
          4th_player_card_0_rank_0:   0.0
          4th_player_card_0_rank_1:   0.0
          4th_player_card_0_rank_2:   0.0
          4th_player_card_0_rank_3:   0.0
          4th_player_card_0_rank_4:   0.0
          4th_player_card_0_rank_5:   0.0
          4th_player_card_0_rank_6:   0.0
          4th_player_card_0_rank_7:   0.0
          4th_player_card_0_rank_8:   0.0
          4th_player_card_0_rank_9:   0.0
         4th_player_card_0_rank_10:   0.0
         4th_player_card_0_rank_11:   0.0
         4th_player_card_0_rank_12:   1.0
           4th_board_card_0_suit_0:   1.0
           4th_board_card_0_suit_1:   0.0
           4th_board_card_0_suit_2:   0.0
           4th_board_card_0_suit_3:   0.0
          4th_player_card_1_rank_0:   0.0
          4th_player_card_1_rank_1:   0.0
          4th_player_card_1_rank_2:   0.0
          4th_player_card_1_rank_3:   0.0
          4th_player_card_1_rank_4:   0.0
          4th_player_card_1_rank_5:   0.0
          4th_player_card_1_rank_6:   0.0
          4th_player_card_1_rank_7:   0.0
          4th_player_card_1_rank_8:   0.0
          4th_player_card_1_rank_9:   1.0
         4th_player_card_1_rank_10:   0.0
         4th_player_card_1_rank_11:   0.0
         4th_player_card_1_rank_12:   0.0
           4th_board_card_1_suit_0:   0.0
           4th_board_card_1_suit_1:   1.0
           4th_board_card_1_suit_2:   0.0
           4th_board_card_1_suit_3:   0.0
          5th_player_card_0_rank_0:   1.0
          5th_player_card_0_rank_1:   0.0
          5th_player_card_0_rank_2:   0.0
          5th_player_card_0_rank_3:   0.0
          5th_player_card_0_rank_4:   0.0
          5th_player_card_0_rank_5:   0.0
          5th_player_card_0_rank_6:   0.0
          5th_player_card_0_rank_7:   0.0
          5th_player_card_0_rank_8:   0.0
          5th_player_card_0_rank_9:   0.0
         5th_player_card_0_rank_10:   0.0
         5th_player_card_0_rank_11:   0.0
         5th_player_card_0_rank_12:   0.0
           5th_board_card_0_suit_0:   1.0
           5th_board_card_0_suit_1:   0.0
           5th_board_card_0_suit_2:   0.0
           5th_board_card_0_suit_3:   0.0
          5th_player_card_1_rank_0:   1.0
          5th_player_card_1_rank_1:   0.0
          5th_player_card_1_rank_2:   0.0
          5th_player_card_1_rank_3:   0.0
          5th_player_card_1_rank_4:   0.0
          5th_player_card_1_rank_5:   0.0
          5th_player_card_1_rank_6:   0.0
          5th_player_card_1_rank_7:   0.0
          5th_player_card_1_rank_8:   0.0
          5th_player_card_1_rank_9:   0.0
         5th_player_card_1_rank_10:   0.0
         5th_player_card_1_rank_11:   0.0
         5th_player_card_1_rank_12:   0.0
           5th_board_card_1_suit_0:   1.0
           5th_board_card_1_suit_1:   0.0
           5th_board_card_1_suit_2:   0.0
           5th_board_card_1_suit_3:   0.0
preflop_player_0_action_0_how_much:   0.0
  preflop_player_0_action_0_what_0:   0.0
  preflop_player_0_action_0_what_1:   0.0
  preflop_player_0_action_0_what_2:   0.0
preflop_player_0_action_1_how_much:   0.0
  preflop_player_0_action_1_what_0:   0.0
  preflop_player_0_action_1_what_1:   0.0
  preflop_player_0_action_1_what_2:   0.0
preflop_player_1_action_0_how_much:   0.0
  preflop_player_1_action_0_what_0:   0.0
  preflop_player_1_action_0_what_1:   0.0
  preflop_player_1_action_0_what_2:   0.0
preflop_player_1_action_1_how_much:   0.0
  preflop_player_1_action_1_what_0:   0.0
  preflop_player_1_action_1_what_1:   0.0
  preflop_player_1_action_1_what_2:   0.0
preflop_player_2_action_0_how_much:   0.0
  preflop_player_2_action_0_what_0:   0.0
  preflop_player_2_action_0_what_1:   0.0
  preflop_player_2_action_0_what_2:   0.0
preflop_player_2_action_1_how_much:   0.0
  preflop_player_2_action_1_what_0:   0.0
  preflop_player_2_action_1_what_1:   0.0
  preflop_player_2_action_1_what_2:   0.0
preflop_player_3_action_0_how_much:   0.0
  preflop_player_3_action_0_what_0:   0.0
  preflop_player_3_action_0_what_1:   0.0
  preflop_player_3_action_0_what_2:   0.0
preflop_player_3_action_1_how_much:   0.0
  preflop_player_3_action_1_what_0:   0.0
  preflop_player_3_action_1_what_1:   0.0
  preflop_player_3_action_1_what_2:   0.0
preflop_player_4_action_0_how_much:   0.0
  preflop_player_4_action_0_what_0:   0.0
  preflop_player_4_action_0_what_1:   0.0
  preflop_player_4_action_0_what_2:   0.0
preflop_player_4_action_1_how_much:   0.0
  preflop_player_4_action_1_what_0:   0.0
  preflop_player_4_action_1_what_1:   0.0
  preflop_player_4_action_1_what_2:   0.0
preflop_player_5_action_0_how_much:   0.0
  preflop_player_5_action_0_what_0:   0.0
  preflop_player_5_action_0_what_1:   0.0
  preflop_player_5_action_0_what_2:   0.0
preflop_player_5_action_1_how_much:   0.0
  preflop_player_5_action_1_what_0:   0.0
  preflop_player_5_action_1_what_1:   0.0
  preflop_player_5_action_1_what_2:   0.0
   flop_player_0_action_0_how_much:   0.0
     flop_player_0_action_0_what_0:   0.0
     flop_player_0_action_0_what_1:   0.0
     flop_player_0_action_0_what_2:   0.0
   flop_player_0_action_1_how_much:   0.0
     flop_player_0_action_1_what_0:   0.0
     flop_player_0_action_1_what_1:   0.0
     flop_player_0_action_1_what_2:   0.0
   flop_player_1_action_0_how_much:   0.0
     flop_player_1_action_0_what_0:   0.0
     flop_player_1_action_0_what_1:   0.0
     flop_player_1_action_0_what_2:   0.0
   flop_player_1_action_1_how_much:   0.0
     flop_player_1_action_1_what_0:   0.0
     flop_player_1_action_1_what_1:   0.0
     flop_player_1_action_1_what_2:   0.0
   flop_player_2_action_0_how_much:   0.0
     flop_player_2_action_0_what_0:   0.0
     flop_player_2_action_0_what_1:   0.0
     flop_player_2_action_0_what_2:   0.0
   flop_player_2_action_1_how_much:   0.0
     flop_player_2_action_1_what_0:   0.0
     flop_player_2_action_1_what_1:   0.0
     flop_player_2_action_1_what_2:   0.0
   flop_player_3_action_0_how_much:   0.0
     flop_player_3_action_0_what_0:   0.0
     flop_player_3_action_0_what_1:   0.0
     flop_player_3_action_0_what_2:   0.0
   flop_player_3_action_1_how_much:   0.0
     flop_player_3_action_1_what_0:   0.0
     flop_player_3_action_1_what_1:   0.0
     flop_player_3_action_1_what_2:   0.0
   flop_player_4_action_0_how_much:   0.0
     flop_player_4_action_0_what_0:   0.0
     flop_player_4_action_0_what_1:   0.0
     flop_player_4_action_0_what_2:   0.0
   flop_player_4_action_1_how_much:   0.0
     flop_player_4_action_1_what_0:   0.0
     flop_player_4_action_1_what_1:   0.0
     flop_player_4_action_1_what_2:   0.0
   flop_player_5_action_0_how_much:   0.0
     flop_player_5_action_0_what_0:   0.0
     flop_player_5_action_0_what_1:   0.0
     flop_player_5_action_0_what_2:   0.0
   flop_player_5_action_1_how_much:   0.0
     flop_player_5_action_1_what_0:   0.0
     flop_player_5_action_1_what_1:   0.0
     flop_player_5_action_1_what_2:   0.0
   turn_player_0_action_0_how_much:   0.0
     turn_player_0_action_0_what_0:   0.0
     turn_player_0_action_0_what_1:   0.0
     turn_player_0_action_0_what_2:   0.0
   turn_player_0_action_1_how_much:   0.0
     turn_player_0_action_1_what_0:   0.0
     turn_player_0_action_1_what_1:   0.0
     turn_player_0_action_1_what_2:   0.0
   turn_player_1_action_0_how_much:   0.0
     turn_player_1_action_0_what_0:   0.0
     turn_player_1_action_0_what_1:   0.0
     turn_player_1_action_0_what_2:   0.0
   turn_player_1_action_1_how_much:   0.0
     turn_player_1_action_1_what_0:   0.0
     turn_player_1_action_1_what_1:   0.0
     turn_player_1_action_1_what_2:   0.0
   turn_player_2_action_0_how_much:   0.0
     turn_player_2_action_0_what_0:   0.0
     turn_player_2_action_0_what_1:   0.0
     turn_player_2_action_0_what_2:   0.0
   turn_player_2_action_1_how_much:   0.0
     turn_player_2_action_1_what_0:   0.0
     turn_player_2_action_1_what_1:   0.0
     turn_player_2_action_1_what_2:   0.0
   turn_player_3_action_0_how_much:   0.0
     turn_player_3_action_0_what_0:   0.0
     turn_player_3_action_0_what_1:   0.0
     turn_player_3_action_0_what_2:   0.0
   turn_player_3_action_1_how_much:   0.0
     turn_player_3_action_1_what_0:   0.0
     turn_player_3_action_1_what_1:   0.0
     turn_player_3_action_1_what_2:   0.0
   turn_player_4_action_0_how_much:   0.0
     turn_player_4_action_0_what_0:   0.0
     turn_player_4_action_0_what_1:   0.0
     turn_player_4_action_0_what_2:   0.0
   turn_player_4_action_1_how_much:   0.0
     turn_player_4_action_1_what_0:   0.0
     turn_player_4_action_1_what_1:   0.0
     turn_player_4_action_1_what_2:   0.0
   turn_player_5_action_0_how_much:   0.0
     turn_player_5_action_0_what_0:   0.0
     turn_player_5_action_0_what_1:   0.0
     turn_player_5_action_0_what_2:   0.0
   turn_player_5_action_1_how_much:   0.0
     turn_player_5_action_1_what_0:   0.0
     turn_player_5_action_1_what_1:   0.0
     turn_player_5_action_1_what_2:   0.0
  river_player_0_action_0_how_much:   0.0
    river_player_0_action_0_what_0:   0.0
    river_player_0_action_0_what_1:   0.0
    river_player_0_action_0_what_2:   0.0
  river_player_0_action_1_how_much:   0.0
    river_player_0_action_1_what_0:   0.0
    river_player_0_action_1_what_1:   0.0
    river_player_0_action_1_what_2:   0.0
  river_player_1_action_0_how_much:   0.0
    river_player_1_action_0_what_0:   0.0
    river_player_1_action_0_what_1:   0.0
    river_player_1_action_0_what_2:   0.0
  river_player_1_action_1_how_much:   0.0
    river_player_1_action_1_what_0:   0.0
    river_player_1_action_1_what_1:   0.0
    river_player_1_action_1_what_2:   0.0
  river_player_2_action_0_how_much:   0.0
    river_player_2_action_0_what_0:   0.0
    river_player_2_action_0_what_1:   0.0
    river_player_2_action_0_what_2:   0.0
  river_player_2_action_1_how_much:   0.0
    river_player_2_action_1_what_0:   0.0
    river_player_2_action_1_what_1:   0.0
    river_player_2_action_1_what_2:   0.0
  river_player_3_action_0_how_much:   0.0
    river_player_3_action_0_what_0:   0.0
    river_player_3_action_0_what_1:   0.0
    river_player_3_action_0_what_2:   0.0
  river_player_3_action_1_how_much:   0.0
    river_player_3_action_1_what_0:   0.0
    river_player_3_action_1_what_1:   0.0
    river_player_3_action_1_what_2:   0.0
  river_player_4_action_0_how_much:   0.0
    river_player_4_action_0_what_0:   0.0
    river_player_4_action_0_what_1:   0.0
    river_player_4_action_0_what_2:   0.0
  river_player_4_action_1_how_much:   0.0
    river_player_4_action_1_what_0:   0.0
    river_player_4_action_1_what_1:   0.0
    river_player_4_action_1_what_2:   0.0
  river_player_5_action_0_how_much:   0.0
    river_player_5_action_0_what_0:   0.0
    river_player_5_action_0_what_1:   0.0
    river_player_5_action_0_what_2:   0.0
  river_player_5_action_1_how_much:   0.0
    river_player_5_action_1_what_0:   0.0
    river_player_5_action_1_what_1:   0.0
    river_player_5_action_1_what_2:   0.0"""
        print(
            "______________________________________ Printing _Observation _________________________________________")
        names = [e + ":  " for e in list(self.obs_idx_dict.keys())]
        str_len = max([len(e) for e in names])
        for name, key in zip(names, list(self.obs_idx_dict.keys())):
            name = name.rjust(str_len)
            print(name, obs[self.obs_idx_dict[key]])

    @property
    def current_player(self):
        return self.env.current_player


class AugmentedObservationFeatureColumns(enum.IntEnum):
    Ante = 0
    Small_blind = 1
    Big_blind = 2
    Min_raise = 3
    Pot_amt = 4
    Total_to_call = 5
    P0_acts_next = 6
    P1_acts_next = 7
    P2_acts_next = 8
    P3_acts_next = 9
    P4_acts_next = 10
    P5_acts_next = 11
    Round_preflop = 12
    Round_flop = 13
    Round_turn = 14
    Round_river = 15
    Side_pot_0 = 16
    Side_pot_1 = 17
    Side_pot_2 = 18
    Side_pot_3 = 19
    Side_pot_4 = 20
    Side_pot_5 = 21
    Stack_p0 = 22
    Curr_bet_p0 = 23
    Has_folded_this_episode_p0 = 24
    Is_allin_p0 = 25
    Side_pot_rank_p0_is_0 = 26
    Side_pot_rank_p0_is_1 = 27
    Side_pot_rank_p0_is_2 = 28
    Side_pot_rank_p0_is_3 = 29
    Side_pot_rank_p0_is_4 = 30
    Side_pot_rank_p0_is_5 = 31
    Stack_p1 = 32
    Curr_bet_p1 = 33
    Has_folded_this_episode_p1 = 34
    Is_allin_p1 = 35
    Side_pot_rank_p1_is_0 = 36
    Side_pot_rank_p1_is_1 = 37
    Side_pot_rank_p1_is_2 = 38
    Side_pot_rank_p1_is_3 = 39
    Side_pot_rank_p1_is_4 = 40
    Side_pot_rank_p1_is_5 = 41
    Stack_p2 = 42
    Curr_bet_p2 = 43
    Has_folded_this_episode_p2 = 44
    Is_allin_p2 = 45
    Side_pot_rank_p2_is_0 = 46
    Side_pot_rank_p2_is_1 = 47
    Side_pot_rank_p2_is_2 = 48
    Side_pot_rank_p2_is_3 = 49
    Side_pot_rank_p2_is_4 = 50
    Side_pot_rank_p2_is_5 = 51
    Stack_p3 = 52
    Curr_bet_p3 = 53
    Has_folded_this_episode_p3 = 54
    Is_allin_p3 = 55
    Side_pot_rank_p3_is_0 = 56
    Side_pot_rank_p3_is_1 = 57
    Side_pot_rank_p3_is_2 = 58
    Side_pot_rank_p3_is_3 = 59
    Side_pot_rank_p3_is_4 = 60
    Side_pot_rank_p3_is_5 = 61
    Stack_p4 = 62
    Curr_bet_p4 = 63
    Has_folded_this_episode_p4 = 64
    Is_allin_p4 = 65
    Side_pot_rank_p4_is_0 = 66
    Side_pot_rank_p4_is_1 = 67
    Side_pot_rank_p4_is_2 = 68
    Side_pot_rank_p4_is_3 = 69
    Side_pot_rank_p4_is_4 = 70
    Side_pot_rank_p4_is_5 = 71
    Stack_p5 = 72
    Curr_bet_p5 = 73
    Has_folded_this_episode_p5 = 74
    Is_allin_p5 = 75
    Side_pot_rank_p5_is_0 = 76
    Side_pot_rank_p5_is_1 = 77
    Side_pot_rank_p5_is_2 = 78
    Side_pot_rank_p5_is_3 = 79
    Side_pot_rank_p5_is_4 = 80
    Side_pot_rank_p5_is_5 = 81
    First_board_card_rank_0 = 82
    First_board_card_rank_1 = 83
    First_board_card_rank_2 = 84
    First_board_card_rank_3 = 85
    First_board_card_rank_4 = 86
    First_board_card_rank_5 = 87
    First_board_card_rank_6 = 88
    First_board_card_rank_7 = 89
    First_board_card_rank_8 = 90
    First_board_card_rank_9 = 91
    First_board_card_rank_10 = 92
    First_board_card_rank_11 = 93
    First_board_card_rank_12 = 94
    First_board_card_suit_0 = 95
    First_board_card_suit_1 = 96
    First_board_card_suit_2 = 97
    First_board_card_suit_3 = 98
    Second_board_card_rank_0 = 99
    Second_board_card_rank_1 = 100
    Second_board_card_rank_2 = 101
    Second_board_card_rank_3 = 102
    Second_board_card_rank_4 = 103
    Second_board_card_rank_5 = 104
    Second_board_card_rank_6 = 105
    Second_board_card_rank_7 = 106
    Second_board_card_rank_8 = 107
    Second_board_card_rank_9 = 108
    Second_board_card_rank_10 = 109
    Second_board_card_rank_11 = 110
    Second_board_card_rank_12 = 111
    Second_board_card_suit_0 = 112
    Second_board_card_suit_1 = 113
    Second_board_card_suit_2 = 114
    Second_board_card_suit_3 = 115
    Third_board_card_rank_0 = 116
    Third_board_card_rank_1 = 117
    Third_board_card_rank_2 = 118
    Third_board_card_rank_3 = 119
    Third_board_card_rank_4 = 120
    Third_board_card_rank_5 = 121
    Third_board_card_rank_6 = 122
    Third_board_card_rank_7 = 123
    Third_board_card_rank_8 = 124
    Third_board_card_rank_9 = 125
    Third_board_card_rank_10 = 126
    Third_board_card_rank_11 = 127
    Third_board_card_rank_12 = 128
    Third_board_card_suit_0 = 129
    Third_board_card_suit_1 = 130
    Third_board_card_suit_2 = 131
    Third_board_card_suit_3 = 132
    Fourth_board_card_rank_0 = 133
    Fourth_board_card_rank_1 = 134
    Fourth_board_card_rank_2 = 135
    Fourth_board_card_rank_3 = 136
    Fourth_board_card_rank_4 = 137
    Fourth_board_card_rank_5 = 138
    Fourth_board_card_rank_6 = 139
    Fourth_board_card_rank_7 = 140
    Fourth_board_card_rank_8 = 141
    Fourth_board_card_rank_9 = 142
    Fourth_board_card_rank_10 = 143
    Fourth_board_card_rank_11 = 144
    Fourth_board_card_rank_12 = 145
    Fourth_board_card_suit_0 = 146
    Fourth_board_card_suit_1 = 147
    Fourth_board_card_suit_2 = 148
    Fourth_board_card_suit_3 = 149
    Fifth_board_card_rank_0 = 150
    Fifth_board_card_rank_1 = 151
    Fifth_board_card_rank_2 = 152
    Fifth_board_card_rank_3 = 153
    Fifth_board_card_rank_4 = 154
    Fifth_board_card_rank_5 = 155
    Fifth_board_card_rank_6 = 156
    Fifth_board_card_rank_7 = 157
    Fifth_board_card_rank_8 = 158
    Fifth_board_card_rank_9 = 159
    Fifth_board_card_rank_10 = 160
    Fifth_board_card_rank_11 = 161
    Fifth_board_card_rank_12 = 162
    Fifth_board_card_suit_0 = 163
    Fifth_board_card_suit_1 = 164
    Fifth_board_card_suit_2 = 165
    Fifth_board_card_suit_3 = 166
    First_player_card_0_rank_0 = 167
    First_player_card_0_rank_1 = 168
    First_player_card_0_rank_2 = 169
    First_player_card_0_rank_3 = 170
    First_player_card_0_rank_4 = 171
    First_player_card_0_rank_5 = 172
    First_player_card_0_rank_6 = 173
    First_player_card_0_rank_7 = 174
    First_player_card_0_rank_8 = 175
    First_player_card_0_rank_9 = 176
    First_player_card_0_rank_10 = 177
    First_player_card_0_rank_11 = 178
    First_player_card_0_rank_12 = 179
    First_player_card_0_suit_0 = 180
    First_player_card_0_suit_1 = 181
    First_player_card_0_suit_2 = 182
    First_player_card_0_suit_3 = 183
    First_player_card_1_rank_0 = 184
    First_player_card_1_rank_1 = 185
    First_player_card_1_rank_2 = 186
    First_player_card_1_rank_3 = 187
    First_player_card_1_rank_4 = 188
    First_player_card_1_rank_5 = 189
    First_player_card_1_rank_6 = 190
    First_player_card_1_rank_7 = 191
    First_player_card_1_rank_8 = 192
    First_player_card_1_rank_9 = 193
    First_player_card_1_rank_10 = 194
    First_player_card_1_rank_11 = 195
    First_player_card_1_rank_12 = 196
    First_player_card_1_suit_0 = 197
    First_player_card_1_suit_1 = 198
    First_player_card_1_suit_2 = 199
    First_player_card_1_suit_3 = 200
    Second_player_card_0_rank_0 = 201
    Second_player_card_0_rank_1 = 202
    Second_player_card_0_rank_2 = 203
    Second_player_card_0_rank_3 = 204
    Second_player_card_0_rank_4 = 205
    Second_player_card_0_rank_5 = 206
    Second_player_card_0_rank_6 = 207
    Second_player_card_0_rank_7 = 208
    Second_player_card_0_rank_8 = 209
    Second_player_card_0_rank_9 = 210
    Second_player_card_0_rank_10 = 211
    Second_player_card_0_rank_11 = 212
    Second_player_card_0_rank_12 = 213
    Second_player_card_0_suit_0 = 214
    Second_player_card_0_suit_1 = 215
    Second_player_card_0_suit_2 = 216
    Second_player_card_0_suit_3 = 217
    Second_player_card_1_rank_0 = 218
    Second_player_card_1_rank_1 = 219
    Second_player_card_1_rank_2 = 220
    Second_player_card_1_rank_3 = 221
    Second_player_card_1_rank_4 = 222
    Second_player_card_1_rank_5 = 223
    Second_player_card_1_rank_6 = 224
    Second_player_card_1_rank_7 = 225
    Second_player_card_1_rank_8 = 226
    Second_player_card_1_rank_9 = 227
    Second_player_card_1_rank_10 = 228
    Second_player_card_1_rank_11 = 229
    Second_player_card_1_rank_12 = 230
    Second_player_card_1_suit_0 = 231
    Second_player_card_1_suit_1 = 232
    Second_player_card_1_suit_2 = 233
    Second_player_card_1_suit_3 = 234
    Third_player_card_0_rank_0 = 235
    Third_player_card_0_rank_1 = 236
    Third_player_card_0_rank_2 = 237
    Third_player_card_0_rank_3 = 238
    Third_player_card_0_rank_4 = 239
    Third_player_card_0_rank_5 = 240
    Third_player_card_0_rank_6 = 241
    Third_player_card_0_rank_7 = 242
    Third_player_card_0_rank_8 = 243
    Third_player_card_0_rank_9 = 244
    Third_player_card_0_rank_10 = 245
    Third_player_card_0_rank_11 = 246
    Third_player_card_0_rank_12 = 247
    Third_player_card_0_suit_0 = 248
    Third_player_card_0_suit_1 = 249
    Third_player_card_0_suit_2 = 250
    Third_player_card_0_suit_3 = 251
    Third_player_card_1_rank_0 = 252
    Third_player_card_1_rank_1 = 253
    Third_player_card_1_rank_2 = 254
    Third_player_card_1_rank_3 = 255
    Third_player_card_1_rank_4 = 256
    Third_player_card_1_rank_5 = 257
    Third_player_card_1_rank_6 = 258
    Third_player_card_1_rank_7 = 259
    Third_player_card_1_rank_8 = 260
    Third_player_card_1_rank_9 = 261
    Third_player_card_1_rank_10 = 262
    Third_player_card_1_rank_11 = 263
    Third_player_card_1_rank_12 = 264
    Third_player_card_1_suit_0 = 265
    Third_player_card_1_suit_1 = 266
    Third_player_card_1_suit_2 = 267
    Third_player_card_1_suit_3 = 268
    Fourth_player_card_0_rank_0 = 269
    Fourth_player_card_0_rank_1 = 270
    Fourth_player_card_0_rank_2 = 271
    Fourth_player_card_0_rank_3 = 272
    Fourth_player_card_0_rank_4 = 273
    Fourth_player_card_0_rank_5 = 274
    Fourth_player_card_0_rank_6 = 275
    Fourth_player_card_0_rank_7 = 276
    Fourth_player_card_0_rank_8 = 277
    Fourth_player_card_0_rank_9 = 278
    Fourth_player_card_0_rank_10 = 279
    Fourth_player_card_0_rank_11 = 280
    Fourth_player_card_0_rank_12 = 281
    Fourth_player_card_0_suit_0 = 282
    Fourth_player_card_0_suit_1 = 283
    Fourth_player_card_0_suit_2 = 284
    Fourth_player_card_0_suit_3 = 285
    Fourth_player_card_1_rank_0 = 286
    Fourth_player_card_1_rank_1 = 287
    Fourth_player_card_1_rank_2 = 288
    Fourth_player_card_1_rank_3 = 289
    Fourth_player_card_1_rank_4 = 290
    Fourth_player_card_1_rank_5 = 291
    Fourth_player_card_1_rank_6 = 292
    Fourth_player_card_1_rank_7 = 293
    Fourth_player_card_1_rank_8 = 294
    Fourth_player_card_1_rank_9 = 295
    Fourth_player_card_1_rank_10 = 296
    Fourth_player_card_1_rank_11 = 297
    Fourth_player_card_1_rank_12 = 298
    Fourth_player_card_1_suit_0 = 299
    Fourth_player_card_1_suit_1 = 300
    Fourth_player_card_1_suit_2 = 301
    Fourth_player_card_1_suit_3 = 302
    Fifth_player_card_0_rank_0 = 303
    Fifth_player_card_0_rank_1 = 304
    Fifth_player_card_0_rank_2 = 305
    Fifth_player_card_0_rank_3 = 306
    Fifth_player_card_0_rank_4 = 307
    Fifth_player_card_0_rank_5 = 308
    Fifth_player_card_0_rank_6 = 309
    Fifth_player_card_0_rank_7 = 310
    Fifth_player_card_0_rank_8 = 311
    Fifth_player_card_0_rank_9 = 312
    Fifth_player_card_0_rank_10 = 313
    Fifth_player_card_0_rank_11 = 314
    Fifth_player_card_0_rank_12 = 315
    Fifth_player_card_0_suit_0 = 316
    Fifth_player_card_0_suit_1 = 317
    Fifth_player_card_0_suit_2 = 318
    Fifth_player_card_0_suit_3 = 319
    Fifth_player_card_1_rank_0 = 320
    Fifth_player_card_1_rank_1 = 321
    Fifth_player_card_1_rank_2 = 322
    Fifth_player_card_1_rank_3 = 323
    Fifth_player_card_1_rank_4 = 324
    Fifth_player_card_1_rank_5 = 325
    Fifth_player_card_1_rank_6 = 326
    Fifth_player_card_1_rank_7 = 327
    Fifth_player_card_1_rank_8 = 328
    Fifth_player_card_1_rank_9 = 329
    Fifth_player_card_1_rank_10 = 330
    Fifth_player_card_1_rank_11 = 331
    Fifth_player_card_1_rank_12 = 332
    Fifth_player_card_1_suit_0 = 333
    Fifth_player_card_1_suit_1 = 334
    Fifth_player_card_1_suit_2 = 335
    Fifth_player_card_1_suit_3 = 336
    Sixth_player_card_0_rank_0 = 337
    Sixth_player_card_0_rank_1 = 338
    Sixth_player_card_0_rank_2 = 339
    Sixth_player_card_0_rank_3 = 340
    Sixth_player_card_0_rank_4 = 341
    Sixth_player_card_0_rank_5 = 342
    Sixth_player_card_0_rank_6 = 343
    Sixth_player_card_0_rank_7 = 344
    Sixth_player_card_0_rank_8 = 345
    Sixth_player_card_0_rank_9 = 346
    Sixth_player_card_0_rank_10 = 347
    Sixth_player_card_0_rank_11 = 348
    Sixth_player_card_0_rank_12 = 349
    Sixth_player_card_0_suit_0 = 350
    Sixth_player_card_0_suit_1 = 351
    Sixth_player_card_0_suit_2 = 352
    Sixth_player_card_0_suit_3 = 353
    Sixth_player_card_1_rank_0 = 354
    Sixth_player_card_1_rank_1 = 355
    Sixth_player_card_1_rank_2 = 356
    Sixth_player_card_1_rank_3 = 357
    Sixth_player_card_1_rank_4 = 358
    Sixth_player_card_1_rank_5 = 359
    Sixth_player_card_1_rank_6 = 360
    Sixth_player_card_1_rank_7 = 361
    Sixth_player_card_1_rank_8 = 362
    Sixth_player_card_1_rank_9 = 363
    Sixth_player_card_1_rank_10 = 364
    Sixth_player_card_1_rank_11 = 365
    Sixth_player_card_1_rank_12 = 366
    Sixth_player_card_1_suit_0 = 367
    Sixth_player_card_1_suit_1 = 368
    Sixth_player_card_1_suit_2 = 369
    Sixth_player_card_1_suit_3 = 370
    Preflop_player_0_action_0_how_much = 371
    Preflop_player_0_action_0_what_0 = 372
    Preflop_player_0_action_0_what_1 = 373
    Preflop_player_0_action_0_what_2 = 374
    Preflop_player_0_action_1_how_much = 375
    Preflop_player_0_action_1_what_0 = 376
    Preflop_player_0_action_1_what_1 = 377
    Preflop_player_0_action_1_what_2 = 378
    Preflop_player_1_action_0_how_much = 379
    Preflop_player_1_action_0_what_0 = 380
    Preflop_player_1_action_0_what_1 = 381
    Preflop_player_1_action_0_what_2 = 382
    Preflop_player_1_action_1_how_much = 383
    Preflop_player_1_action_1_what_0 = 384
    Preflop_player_1_action_1_what_1 = 385
    Preflop_player_1_action_1_what_2 = 386
    Preflop_player_2_action_0_how_much = 387
    Preflop_player_2_action_0_what_0 = 388
    Preflop_player_2_action_0_what_1 = 389
    Preflop_player_2_action_0_what_2 = 390
    Preflop_player_2_action_1_how_much = 391
    Preflop_player_2_action_1_what_0 = 392
    Preflop_player_2_action_1_what_1 = 393
    Preflop_player_2_action_1_what_2 = 394
    Preflop_player_3_action_0_how_much = 395
    Preflop_player_3_action_0_what_0 = 396
    Preflop_player_3_action_0_what_1 = 397
    Preflop_player_3_action_0_what_2 = 398
    Preflop_player_3_action_1_how_much = 399
    Preflop_player_3_action_1_what_0 = 400
    Preflop_player_3_action_1_what_1 = 401
    Preflop_player_3_action_1_what_2 = 402
    Preflop_player_4_action_0_how_much = 403
    Preflop_player_4_action_0_what_0 = 404
    Preflop_player_4_action_0_what_1 = 405
    Preflop_player_4_action_0_what_2 = 406
    Preflop_player_4_action_1_how_much = 407
    Preflop_player_4_action_1_what_0 = 408
    Preflop_player_4_action_1_what_1 = 409
    Preflop_player_4_action_1_what_2 = 410
    Preflop_player_5_action_0_how_much = 411
    Preflop_player_5_action_0_what_0 = 412
    Preflop_player_5_action_0_what_1 = 413
    Preflop_player_5_action_0_what_2 = 414
    Preflop_player_5_action_1_how_much = 415
    Preflop_player_5_action_1_what_0 = 416
    Preflop_player_5_action_1_what_1 = 417
    Preflop_player_5_action_1_what_2 = 418
    Flop_player_0_action_0_how_much = 419
    Flop_player_0_action_0_what_0 = 420
    Flop_player_0_action_0_what_1 = 421
    Flop_player_0_action_0_what_2 = 422
    Flop_player_0_action_1_how_much = 423
    Flop_player_0_action_1_what_0 = 424
    Flop_player_0_action_1_what_1 = 425
    Flop_player_0_action_1_what_2 = 426
    Flop_player_1_action_0_how_much = 427
    Flop_player_1_action_0_what_0 = 428
    Flop_player_1_action_0_what_1 = 429
    Flop_player_1_action_0_what_2 = 430
    Flop_player_1_action_1_how_much = 431
    Flop_player_1_action_1_what_0 = 432
    Flop_player_1_action_1_what_1 = 433
    Flop_player_1_action_1_what_2 = 434
    Flop_player_2_action_0_how_much = 435
    Flop_player_2_action_0_what_0 = 436
    Flop_player_2_action_0_what_1 = 437
    Flop_player_2_action_0_what_2 = 438
    Flop_player_2_action_1_how_much = 439
    Flop_player_2_action_1_what_0 = 440
    Flop_player_2_action_1_what_1 = 441
    Flop_player_2_action_1_what_2 = 442
    Flop_player_3_action_0_how_much = 443
    Flop_player_3_action_0_what_0 = 444
    Flop_player_3_action_0_what_1 = 445
    Flop_player_3_action_0_what_2 = 446
    Flop_player_3_action_1_how_much = 447
    Flop_player_3_action_1_what_0 = 448
    Flop_player_3_action_1_what_1 = 449
    Flop_player_3_action_1_what_2 = 450
    Flop_player_4_action_0_how_much = 451
    Flop_player_4_action_0_what_0 = 452
    Flop_player_4_action_0_what_1 = 453
    Flop_player_4_action_0_what_2 = 454
    Flop_player_4_action_1_how_much = 455
    Flop_player_4_action_1_what_0 = 456
    Flop_player_4_action_1_what_1 = 457
    Flop_player_4_action_1_what_2 = 458
    Flop_player_5_action_0_how_much = 459
    Flop_player_5_action_0_what_0 = 460
    Flop_player_5_action_0_what_1 = 461
    Flop_player_5_action_0_what_2 = 462
    Flop_player_5_action_1_how_much = 463
    Flop_player_5_action_1_what_0 = 464
    Flop_player_5_action_1_what_1 = 465
    Flop_player_5_action_1_what_2 = 466
    Turn_player_0_action_0_how_much = 467
    Turn_player_0_action_0_what_0 = 468
    Turn_player_0_action_0_what_1 = 469
    Turn_player_0_action_0_what_2 = 470
    Turn_player_0_action_1_how_much = 471
    Turn_player_0_action_1_what_0 = 472
    Turn_player_0_action_1_what_1 = 473
    Turn_player_0_action_1_what_2 = 474
    Turn_player_1_action_0_how_much = 475
    Turn_player_1_action_0_what_0 = 476
    Turn_player_1_action_0_what_1 = 477
    Turn_player_1_action_0_what_2 = 478
    Turn_player_1_action_1_how_much = 479
    Turn_player_1_action_1_what_0 = 480
    Turn_player_1_action_1_what_1 = 481
    Turn_player_1_action_1_what_2 = 482
    Turn_player_2_action_0_how_much = 483
    Turn_player_2_action_0_what_0 = 484
    Turn_player_2_action_0_what_1 = 485
    Turn_player_2_action_0_what_2 = 486
    Turn_player_2_action_1_how_much = 487
    Turn_player_2_action_1_what_0 = 488
    Turn_player_2_action_1_what_1 = 489
    Turn_player_2_action_1_what_2 = 490
    Turn_player_3_action_0_how_much = 491
    Turn_player_3_action_0_what_0 = 492
    Turn_player_3_action_0_what_1 = 493
    Turn_player_3_action_0_what_2 = 494
    Turn_player_3_action_1_how_much = 495
    Turn_player_3_action_1_what_0 = 496
    Turn_player_3_action_1_what_1 = 497
    Turn_player_3_action_1_what_2 = 498
    Turn_player_4_action_0_how_much = 499
    Turn_player_4_action_0_what_0 = 500
    Turn_player_4_action_0_what_1 = 501
    Turn_player_4_action_0_what_2 = 502
    Turn_player_4_action_1_how_much = 503
    Turn_player_4_action_1_what_0 = 504
    Turn_player_4_action_1_what_1 = 505
    Turn_player_4_action_1_what_2 = 506
    Turn_player_5_action_0_how_much = 507
    Turn_player_5_action_0_what_0 = 508
    Turn_player_5_action_0_what_1 = 509
    Turn_player_5_action_0_what_2 = 510
    Turn_player_5_action_1_how_much = 511
    Turn_player_5_action_1_what_0 = 512
    Turn_player_5_action_1_what_1 = 513
    Turn_player_5_action_1_what_2 = 514
    River_player_0_action_0_how_much = 515
    River_player_0_action_0_what_0 = 516
    River_player_0_action_0_what_1 = 517
    River_player_0_action_0_what_2 = 518
    River_player_0_action_1_how_much = 519
    River_player_0_action_1_what_0 = 520
    River_player_0_action_1_what_1 = 521
    River_player_0_action_1_what_2 = 522
    River_player_1_action_0_how_much = 523
    River_player_1_action_0_what_0 = 524
    River_player_1_action_0_what_1 = 525
    River_player_1_action_0_what_2 = 526
    River_player_1_action_1_how_much = 527
    River_player_1_action_1_what_0 = 528
    River_player_1_action_1_what_1 = 529
    River_player_1_action_1_what_2 = 530
    River_player_2_action_0_how_much = 531
    River_player_2_action_0_what_0 = 532
    River_player_2_action_0_what_1 = 533
    River_player_2_action_0_what_2 = 534
    River_player_2_action_1_how_much = 535
    River_player_2_action_1_what_0 = 536
    River_player_2_action_1_what_1 = 537
    River_player_2_action_1_what_2 = 538
    River_player_3_action_0_how_much = 539
    River_player_3_action_0_what_0 = 540
    River_player_3_action_0_what_1 = 541
    River_player_3_action_0_what_2 = 542
    River_player_3_action_1_how_much = 543
    River_player_3_action_1_what_0 = 544
    River_player_3_action_1_what_1 = 545
    River_player_3_action_1_what_2 = 546
    River_player_4_action_0_how_much = 547
    River_player_4_action_0_what_0 = 548
    River_player_4_action_0_what_1 = 549
    River_player_4_action_0_what_2 = 550
    River_player_4_action_1_how_much = 551
    River_player_4_action_1_what_0 = 552
    River_player_4_action_1_what_1 = 553
    River_player_4_action_1_what_2 = 554
    River_player_5_action_0_how_much = 555
    River_player_5_action_0_what_0 = 556
    River_player_5_action_0_what_1 = 557
    River_player_5_action_0_what_2 = 558
    River_player_5_action_1_how_much = 559
    River_player_5_action_1_what_0 = 560
    River_player_5_action_1_what_1 = 561
    River_player_5_action_1_what_2 = 562
    Position_is_btn = 563
    Position_is_sb = 564
    Position_is_bb = 565
    Position_is_utg = 566
    Position_is_mp = 567
    Position_is_co = 568


features_with_hud_stats = [c.name for c in AugmentedObservationFeatureColumns]
features_with_hud_stats += ['Win_probability',
                            'Player_0_is_tight',
                            'Player_0_is_aggressive',
                            'Player_0_is_balanced_or_unknown',
                            'Player_1_is_tight',
                            'Player_1_is_aggressive',
                            'Player_1_is_balanced_or_unknown',
                            'Player_2_is_tight',
                            'Player_2_is_aggressive',
                            'Player_2_is_balanced_or_unknown',
                            'Player_3_is_tight',
                            'Player_3_is_aggressive',
                            'Player_3_is_balanced_or_unknown',
                            'Player_4_is_tight',
                            'Player_4_is_aggressive',
                            'Player_4_is_balanced_or_unknown',
                            'Player_5_is_tight',
                            'Player_5_is_aggressive',
                            'Player_5_is_balanced_or_unknown', ]


# FeaturesWithHudStats = enum.IntEnum('FeaturesWithHudStats', features_with_hud_stats)
class FeaturesWithHudStats(enum.IntEnum):
    Ante = 0
    Small_blind = 1
    Big_blind = 2
    Min_raise = 3
    Pot_amt = 4
    Total_to_call = 5
    P0_acts_next = 6
    P1_acts_next = 7
    P2_acts_next = 8
    P3_acts_next = 9
    P4_acts_next = 10
    P5_acts_next = 11
    Round_preflop = 12
    Round_flop = 13
    Round_turn = 14
    Round_river = 15
    Side_pot_0 = 16
    Side_pot_1 = 17
    Side_pot_2 = 18
    Side_pot_3 = 19
    Side_pot_4 = 20
    Side_pot_5 = 21
    Stack_p0 = 22
    Curr_bet_p0 = 23
    Has_folded_this_episode_p0 = 24
    Is_allin_p0 = 25
    Side_pot_rank_p0_is_0 = 26
    Side_pot_rank_p0_is_1 = 27
    Side_pot_rank_p0_is_2 = 28
    Side_pot_rank_p0_is_3 = 29
    Side_pot_rank_p0_is_4 = 30
    Side_pot_rank_p0_is_5 = 31
    Stack_p1 = 32
    Curr_bet_p1 = 33
    Has_folded_this_episode_p1 = 34
    Is_allin_p1 = 35
    Side_pot_rank_p1_is_0 = 36
    Side_pot_rank_p1_is_1 = 37
    Side_pot_rank_p1_is_2 = 38
    Side_pot_rank_p1_is_3 = 39
    Side_pot_rank_p1_is_4 = 40
    Side_pot_rank_p1_is_5 = 41
    Stack_p2 = 42
    Curr_bet_p2 = 43
    Has_folded_this_episode_p2 = 44
    Is_allin_p2 = 45
    Side_pot_rank_p2_is_0 = 46
    Side_pot_rank_p2_is_1 = 47
    Side_pot_rank_p2_is_2 = 48
    Side_pot_rank_p2_is_3 = 49
    Side_pot_rank_p2_is_4 = 50
    Side_pot_rank_p2_is_5 = 51
    Stack_p3 = 52
    Curr_bet_p3 = 53
    Has_folded_this_episode_p3 = 54
    Is_allin_p3 = 55
    Side_pot_rank_p3_is_0 = 56
    Side_pot_rank_p3_is_1 = 57
    Side_pot_rank_p3_is_2 = 58
    Side_pot_rank_p3_is_3 = 59
    Side_pot_rank_p3_is_4 = 60
    Side_pot_rank_p3_is_5 = 61
    Stack_p4 = 62
    Curr_bet_p4 = 63
    Has_folded_this_episode_p4 = 64
    Is_allin_p4 = 65
    Side_pot_rank_p4_is_0 = 66
    Side_pot_rank_p4_is_1 = 67
    Side_pot_rank_p4_is_2 = 68
    Side_pot_rank_p4_is_3 = 69
    Side_pot_rank_p4_is_4 = 70
    Side_pot_rank_p4_is_5 = 71
    Stack_p5 = 72
    Curr_bet_p5 = 73
    Has_folded_this_episode_p5 = 74
    Is_allin_p5 = 75
    Side_pot_rank_p5_is_0 = 76
    Side_pot_rank_p5_is_1 = 77
    Side_pot_rank_p5_is_2 = 78
    Side_pot_rank_p5_is_3 = 79
    Side_pot_rank_p5_is_4 = 80
    Side_pot_rank_p5_is_5 = 81
    First_board_card_rank_0 = 82
    First_board_card_rank_1 = 83
    First_board_card_rank_2 = 84
    First_board_card_rank_3 = 85
    First_board_card_rank_4 = 86
    First_board_card_rank_5 = 87
    First_board_card_rank_6 = 88
    First_board_card_rank_7 = 89
    First_board_card_rank_8 = 90
    First_board_card_rank_9 = 91
    First_board_card_rank_10 = 92
    First_board_card_rank_11 = 93
    First_board_card_rank_12 = 94
    First_board_card_suit_0 = 95
    First_board_card_suit_1 = 96
    First_board_card_suit_2 = 97
    First_board_card_suit_3 = 98
    Second_board_card_rank_0 = 99
    Second_board_card_rank_1 = 100
    Second_board_card_rank_2 = 101
    Second_board_card_rank_3 = 102
    Second_board_card_rank_4 = 103
    Second_board_card_rank_5 = 104
    Second_board_card_rank_6 = 105
    Second_board_card_rank_7 = 106
    Second_board_card_rank_8 = 107
    Second_board_card_rank_9 = 108
    Second_board_card_rank_10 = 109
    Second_board_card_rank_11 = 110
    Second_board_card_rank_12 = 111
    Second_board_card_suit_0 = 112
    Second_board_card_suit_1 = 113
    Second_board_card_suit_2 = 114
    Second_board_card_suit_3 = 115
    Third_board_card_rank_0 = 116
    Third_board_card_rank_1 = 117
    Third_board_card_rank_2 = 118
    Third_board_card_rank_3 = 119
    Third_board_card_rank_4 = 120
    Third_board_card_rank_5 = 121
    Third_board_card_rank_6 = 122
    Third_board_card_rank_7 = 123
    Third_board_card_rank_8 = 124
    Third_board_card_rank_9 = 125
    Third_board_card_rank_10 = 126
    Third_board_card_rank_11 = 127
    Third_board_card_rank_12 = 128
    Third_board_card_suit_0 = 129
    Third_board_card_suit_1 = 130
    Third_board_card_suit_2 = 131
    Third_board_card_suit_3 = 132
    Fourth_board_card_rank_0 = 133
    Fourth_board_card_rank_1 = 134
    Fourth_board_card_rank_2 = 135
    Fourth_board_card_rank_3 = 136
    Fourth_board_card_rank_4 = 137
    Fourth_board_card_rank_5 = 138
    Fourth_board_card_rank_6 = 139
    Fourth_board_card_rank_7 = 140
    Fourth_board_card_rank_8 = 141
    Fourth_board_card_rank_9 = 142
    Fourth_board_card_rank_10 = 143
    Fourth_board_card_rank_11 = 144
    Fourth_board_card_rank_12 = 145
    Fourth_board_card_suit_0 = 146
    Fourth_board_card_suit_1 = 147
    Fourth_board_card_suit_2 = 148
    Fourth_board_card_suit_3 = 149
    Fifth_board_card_rank_0 = 150
    Fifth_board_card_rank_1 = 151
    Fifth_board_card_rank_2 = 152
    Fifth_board_card_rank_3 = 153
    Fifth_board_card_rank_4 = 154
    Fifth_board_card_rank_5 = 155
    Fifth_board_card_rank_6 = 156
    Fifth_board_card_rank_7 = 157
    Fifth_board_card_rank_8 = 158
    Fifth_board_card_rank_9 = 159
    Fifth_board_card_rank_10 = 160
    Fifth_board_card_rank_11 = 161
    Fifth_board_card_rank_12 = 162
    Fifth_board_card_suit_0 = 163
    Fifth_board_card_suit_1 = 164
    Fifth_board_card_suit_2 = 165
    Fifth_board_card_suit_3 = 166
    First_player_card_0_rank_0 = 167
    First_player_card_0_rank_1 = 168
    First_player_card_0_rank_2 = 169
    First_player_card_0_rank_3 = 170
    First_player_card_0_rank_4 = 171
    First_player_card_0_rank_5 = 172
    First_player_card_0_rank_6 = 173
    First_player_card_0_rank_7 = 174
    First_player_card_0_rank_8 = 175
    First_player_card_0_rank_9 = 176
    First_player_card_0_rank_10 = 177
    First_player_card_0_rank_11 = 178
    First_player_card_0_rank_12 = 179
    First_player_card_0_suit_0 = 180
    First_player_card_0_suit_1 = 181
    First_player_card_0_suit_2 = 182
    First_player_card_0_suit_3 = 183
    First_player_card_1_rank_0 = 184
    First_player_card_1_rank_1 = 185
    First_player_card_1_rank_2 = 186
    First_player_card_1_rank_3 = 187
    First_player_card_1_rank_4 = 188
    First_player_card_1_rank_5 = 189
    First_player_card_1_rank_6 = 190
    First_player_card_1_rank_7 = 191
    First_player_card_1_rank_8 = 192
    First_player_card_1_rank_9 = 193
    First_player_card_1_rank_10 = 194
    First_player_card_1_rank_11 = 195
    First_player_card_1_rank_12 = 196
    First_player_card_1_suit_0 = 197
    First_player_card_1_suit_1 = 198
    First_player_card_1_suit_2 = 199
    First_player_card_1_suit_3 = 200
    Second_player_card_0_rank_0 = 201
    Second_player_card_0_rank_1 = 202
    Second_player_card_0_rank_2 = 203
    Second_player_card_0_rank_3 = 204
    Second_player_card_0_rank_4 = 205
    Second_player_card_0_rank_5 = 206
    Second_player_card_0_rank_6 = 207
    Second_player_card_0_rank_7 = 208
    Second_player_card_0_rank_8 = 209
    Second_player_card_0_rank_9 = 210
    Second_player_card_0_rank_10 = 211
    Second_player_card_0_rank_11 = 212
    Second_player_card_0_rank_12 = 213
    Second_player_card_0_suit_0 = 214
    Second_player_card_0_suit_1 = 215
    Second_player_card_0_suit_2 = 216
    Second_player_card_0_suit_3 = 217
    Second_player_card_1_rank_0 = 218
    Second_player_card_1_rank_1 = 219
    Second_player_card_1_rank_2 = 220
    Second_player_card_1_rank_3 = 221
    Second_player_card_1_rank_4 = 222
    Second_player_card_1_rank_5 = 223
    Second_player_card_1_rank_6 = 224
    Second_player_card_1_rank_7 = 225
    Second_player_card_1_rank_8 = 226
    Second_player_card_1_rank_9 = 227
    Second_player_card_1_rank_10 = 228
    Second_player_card_1_rank_11 = 229
    Second_player_card_1_rank_12 = 230
    Second_player_card_1_suit_0 = 231
    Second_player_card_1_suit_1 = 232
    Second_player_card_1_suit_2 = 233
    Second_player_card_1_suit_3 = 234
    Third_player_card_0_rank_0 = 235
    Third_player_card_0_rank_1 = 236
    Third_player_card_0_rank_2 = 237
    Third_player_card_0_rank_3 = 238
    Third_player_card_0_rank_4 = 239
    Third_player_card_0_rank_5 = 240
    Third_player_card_0_rank_6 = 241
    Third_player_card_0_rank_7 = 242
    Third_player_card_0_rank_8 = 243
    Third_player_card_0_rank_9 = 244
    Third_player_card_0_rank_10 = 245
    Third_player_card_0_rank_11 = 246
    Third_player_card_0_rank_12 = 247
    Third_player_card_0_suit_0 = 248
    Third_player_card_0_suit_1 = 249
    Third_player_card_0_suit_2 = 250
    Third_player_card_0_suit_3 = 251
    Third_player_card_1_rank_0 = 252
    Third_player_card_1_rank_1 = 253
    Third_player_card_1_rank_2 = 254
    Third_player_card_1_rank_3 = 255
    Third_player_card_1_rank_4 = 256
    Third_player_card_1_rank_5 = 257
    Third_player_card_1_rank_6 = 258
    Third_player_card_1_rank_7 = 259
    Third_player_card_1_rank_8 = 260
    Third_player_card_1_rank_9 = 261
    Third_player_card_1_rank_10 = 262
    Third_player_card_1_rank_11 = 263
    Third_player_card_1_rank_12 = 264
    Third_player_card_1_suit_0 = 265
    Third_player_card_1_suit_1 = 266
    Third_player_card_1_suit_2 = 267
    Third_player_card_1_suit_3 = 268
    Fourth_player_card_0_rank_0 = 269
    Fourth_player_card_0_rank_1 = 270
    Fourth_player_card_0_rank_2 = 271
    Fourth_player_card_0_rank_3 = 272
    Fourth_player_card_0_rank_4 = 273
    Fourth_player_card_0_rank_5 = 274
    Fourth_player_card_0_rank_6 = 275
    Fourth_player_card_0_rank_7 = 276
    Fourth_player_card_0_rank_8 = 277
    Fourth_player_card_0_rank_9 = 278
    Fourth_player_card_0_rank_10 = 279
    Fourth_player_card_0_rank_11 = 280
    Fourth_player_card_0_rank_12 = 281
    Fourth_player_card_0_suit_0 = 282
    Fourth_player_card_0_suit_1 = 283
    Fourth_player_card_0_suit_2 = 284
    Fourth_player_card_0_suit_3 = 285
    Fourth_player_card_1_rank_0 = 286
    Fourth_player_card_1_rank_1 = 287
    Fourth_player_card_1_rank_2 = 288
    Fourth_player_card_1_rank_3 = 289
    Fourth_player_card_1_rank_4 = 290
    Fourth_player_card_1_rank_5 = 291
    Fourth_player_card_1_rank_6 = 292
    Fourth_player_card_1_rank_7 = 293
    Fourth_player_card_1_rank_8 = 294
    Fourth_player_card_1_rank_9 = 295
    Fourth_player_card_1_rank_10 = 296
    Fourth_player_card_1_rank_11 = 297
    Fourth_player_card_1_rank_12 = 298
    Fourth_player_card_1_suit_0 = 299
    Fourth_player_card_1_suit_1 = 300
    Fourth_player_card_1_suit_2 = 301
    Fourth_player_card_1_suit_3 = 302
    Fifth_player_card_0_rank_0 = 303
    Fifth_player_card_0_rank_1 = 304
    Fifth_player_card_0_rank_2 = 305
    Fifth_player_card_0_rank_3 = 306
    Fifth_player_card_0_rank_4 = 307
    Fifth_player_card_0_rank_5 = 308
    Fifth_player_card_0_rank_6 = 309
    Fifth_player_card_0_rank_7 = 310
    Fifth_player_card_0_rank_8 = 311
    Fifth_player_card_0_rank_9 = 312
    Fifth_player_card_0_rank_10 = 313
    Fifth_player_card_0_rank_11 = 314
    Fifth_player_card_0_rank_12 = 315
    Fifth_player_card_0_suit_0 = 316
    Fifth_player_card_0_suit_1 = 317
    Fifth_player_card_0_suit_2 = 318
    Fifth_player_card_0_suit_3 = 319
    Fifth_player_card_1_rank_0 = 320
    Fifth_player_card_1_rank_1 = 321
    Fifth_player_card_1_rank_2 = 322
    Fifth_player_card_1_rank_3 = 323
    Fifth_player_card_1_rank_4 = 324
    Fifth_player_card_1_rank_5 = 325
    Fifth_player_card_1_rank_6 = 326
    Fifth_player_card_1_rank_7 = 327
    Fifth_player_card_1_rank_8 = 328
    Fifth_player_card_1_rank_9 = 329
    Fifth_player_card_1_rank_10 = 330
    Fifth_player_card_1_rank_11 = 331
    Fifth_player_card_1_rank_12 = 332
    Fifth_player_card_1_suit_0 = 333
    Fifth_player_card_1_suit_1 = 334
    Fifth_player_card_1_suit_2 = 335
    Fifth_player_card_1_suit_3 = 336
    Sixth_player_card_0_rank_0 = 337
    Sixth_player_card_0_rank_1 = 338
    Sixth_player_card_0_rank_2 = 339
    Sixth_player_card_0_rank_3 = 340
    Sixth_player_card_0_rank_4 = 341
    Sixth_player_card_0_rank_5 = 342
    Sixth_player_card_0_rank_6 = 343
    Sixth_player_card_0_rank_7 = 344
    Sixth_player_card_0_rank_8 = 345
    Sixth_player_card_0_rank_9 = 346
    Sixth_player_card_0_rank_10 = 347
    Sixth_player_card_0_rank_11 = 348
    Sixth_player_card_0_rank_12 = 349
    Sixth_player_card_0_suit_0 = 350
    Sixth_player_card_0_suit_1 = 351
    Sixth_player_card_0_suit_2 = 352
    Sixth_player_card_0_suit_3 = 353
    Sixth_player_card_1_rank_0 = 354
    Sixth_player_card_1_rank_1 = 355
    Sixth_player_card_1_rank_2 = 356
    Sixth_player_card_1_rank_3 = 357
    Sixth_player_card_1_rank_4 = 358
    Sixth_player_card_1_rank_5 = 359
    Sixth_player_card_1_rank_6 = 360
    Sixth_player_card_1_rank_7 = 361
    Sixth_player_card_1_rank_8 = 362
    Sixth_player_card_1_rank_9 = 363
    Sixth_player_card_1_rank_10 = 364
    Sixth_player_card_1_rank_11 = 365
    Sixth_player_card_1_rank_12 = 366
    Sixth_player_card_1_suit_0 = 367
    Sixth_player_card_1_suit_1 = 368
    Sixth_player_card_1_suit_2 = 369
    Sixth_player_card_1_suit_3 = 370
    Preflop_player_0_action_0_how_much = 371
    Preflop_player_0_action_0_what_0 = 372
    Preflop_player_0_action_0_what_1 = 373
    Preflop_player_0_action_0_what_2 = 374
    Preflop_player_0_action_1_how_much = 375
    Preflop_player_0_action_1_what_0 = 376
    Preflop_player_0_action_1_what_1 = 377
    Preflop_player_0_action_1_what_2 = 378
    Preflop_player_1_action_0_how_much = 379
    Preflop_player_1_action_0_what_0 = 380
    Preflop_player_1_action_0_what_1 = 381
    Preflop_player_1_action_0_what_2 = 382
    Preflop_player_1_action_1_how_much = 383
    Preflop_player_1_action_1_what_0 = 384
    Preflop_player_1_action_1_what_1 = 385
    Preflop_player_1_action_1_what_2 = 386
    Preflop_player_2_action_0_how_much = 387
    Preflop_player_2_action_0_what_0 = 388
    Preflop_player_2_action_0_what_1 = 389
    Preflop_player_2_action_0_what_2 = 390
    Preflop_player_2_action_1_how_much = 391
    Preflop_player_2_action_1_what_0 = 392
    Preflop_player_2_action_1_what_1 = 393
    Preflop_player_2_action_1_what_2 = 394
    Preflop_player_3_action_0_how_much = 395
    Preflop_player_3_action_0_what_0 = 396
    Preflop_player_3_action_0_what_1 = 397
    Preflop_player_3_action_0_what_2 = 398
    Preflop_player_3_action_1_how_much = 399
    Preflop_player_3_action_1_what_0 = 400
    Preflop_player_3_action_1_what_1 = 401
    Preflop_player_3_action_1_what_2 = 402
    Preflop_player_4_action_0_how_much = 403
    Preflop_player_4_action_0_what_0 = 404
    Preflop_player_4_action_0_what_1 = 405
    Preflop_player_4_action_0_what_2 = 406
    Preflop_player_4_action_1_how_much = 407
    Preflop_player_4_action_1_what_0 = 408
    Preflop_player_4_action_1_what_1 = 409
    Preflop_player_4_action_1_what_2 = 410
    Preflop_player_5_action_0_how_much = 411
    Preflop_player_5_action_0_what_0 = 412
    Preflop_player_5_action_0_what_1 = 413
    Preflop_player_5_action_0_what_2 = 414
    Preflop_player_5_action_1_how_much = 415
    Preflop_player_5_action_1_what_0 = 416
    Preflop_player_5_action_1_what_1 = 417
    Preflop_player_5_action_1_what_2 = 418
    Flop_player_0_action_0_how_much = 419
    Flop_player_0_action_0_what_0 = 420
    Flop_player_0_action_0_what_1 = 421
    Flop_player_0_action_0_what_2 = 422
    Flop_player_0_action_1_how_much = 423
    Flop_player_0_action_1_what_0 = 424
    Flop_player_0_action_1_what_1 = 425
    Flop_player_0_action_1_what_2 = 426
    Flop_player_1_action_0_how_much = 427
    Flop_player_1_action_0_what_0 = 428
    Flop_player_1_action_0_what_1 = 429
    Flop_player_1_action_0_what_2 = 430
    Flop_player_1_action_1_how_much = 431
    Flop_player_1_action_1_what_0 = 432
    Flop_player_1_action_1_what_1 = 433
    Flop_player_1_action_1_what_2 = 434
    Flop_player_2_action_0_how_much = 435
    Flop_player_2_action_0_what_0 = 436
    Flop_player_2_action_0_what_1 = 437
    Flop_player_2_action_0_what_2 = 438
    Flop_player_2_action_1_how_much = 439
    Flop_player_2_action_1_what_0 = 440
    Flop_player_2_action_1_what_1 = 441
    Flop_player_2_action_1_what_2 = 442
    Flop_player_3_action_0_how_much = 443
    Flop_player_3_action_0_what_0 = 444
    Flop_player_3_action_0_what_1 = 445
    Flop_player_3_action_0_what_2 = 446
    Flop_player_3_action_1_how_much = 447
    Flop_player_3_action_1_what_0 = 448
    Flop_player_3_action_1_what_1 = 449
    Flop_player_3_action_1_what_2 = 450
    Flop_player_4_action_0_how_much = 451
    Flop_player_4_action_0_what_0 = 452
    Flop_player_4_action_0_what_1 = 453
    Flop_player_4_action_0_what_2 = 454
    Flop_player_4_action_1_how_much = 455
    Flop_player_4_action_1_what_0 = 456
    Flop_player_4_action_1_what_1 = 457
    Flop_player_4_action_1_what_2 = 458
    Flop_player_5_action_0_how_much = 459
    Flop_player_5_action_0_what_0 = 460
    Flop_player_5_action_0_what_1 = 461
    Flop_player_5_action_0_what_2 = 462
    Flop_player_5_action_1_how_much = 463
    Flop_player_5_action_1_what_0 = 464
    Flop_player_5_action_1_what_1 = 465
    Flop_player_5_action_1_what_2 = 466
    Turn_player_0_action_0_how_much = 467
    Turn_player_0_action_0_what_0 = 468
    Turn_player_0_action_0_what_1 = 469
    Turn_player_0_action_0_what_2 = 470
    Turn_player_0_action_1_how_much = 471
    Turn_player_0_action_1_what_0 = 472
    Turn_player_0_action_1_what_1 = 473
    Turn_player_0_action_1_what_2 = 474
    Turn_player_1_action_0_how_much = 475
    Turn_player_1_action_0_what_0 = 476
    Turn_player_1_action_0_what_1 = 477
    Turn_player_1_action_0_what_2 = 478
    Turn_player_1_action_1_how_much = 479
    Turn_player_1_action_1_what_0 = 480
    Turn_player_1_action_1_what_1 = 481
    Turn_player_1_action_1_what_2 = 482
    Turn_player_2_action_0_how_much = 483
    Turn_player_2_action_0_what_0 = 484
    Turn_player_2_action_0_what_1 = 485
    Turn_player_2_action_0_what_2 = 486
    Turn_player_2_action_1_how_much = 487
    Turn_player_2_action_1_what_0 = 488
    Turn_player_2_action_1_what_1 = 489
    Turn_player_2_action_1_what_2 = 490
    Turn_player_3_action_0_how_much = 491
    Turn_player_3_action_0_what_0 = 492
    Turn_player_3_action_0_what_1 = 493
    Turn_player_3_action_0_what_2 = 494
    Turn_player_3_action_1_how_much = 495
    Turn_player_3_action_1_what_0 = 496
    Turn_player_3_action_1_what_1 = 497
    Turn_player_3_action_1_what_2 = 498
    Turn_player_4_action_0_how_much = 499
    Turn_player_4_action_0_what_0 = 500
    Turn_player_4_action_0_what_1 = 501
    Turn_player_4_action_0_what_2 = 502
    Turn_player_4_action_1_how_much = 503
    Turn_player_4_action_1_what_0 = 504
    Turn_player_4_action_1_what_1 = 505
    Turn_player_4_action_1_what_2 = 506
    Turn_player_5_action_0_how_much = 507
    Turn_player_5_action_0_what_0 = 508
    Turn_player_5_action_0_what_1 = 509
    Turn_player_5_action_0_what_2 = 510
    Turn_player_5_action_1_how_much = 511
    Turn_player_5_action_1_what_0 = 512
    Turn_player_5_action_1_what_1 = 513
    Turn_player_5_action_1_what_2 = 514
    River_player_0_action_0_how_much = 515
    River_player_0_action_0_what_0 = 516
    River_player_0_action_0_what_1 = 517
    River_player_0_action_0_what_2 = 518
    River_player_0_action_1_how_much = 519
    River_player_0_action_1_what_0 = 520
    River_player_0_action_1_what_1 = 521
    River_player_0_action_1_what_2 = 522
    River_player_1_action_0_how_much = 523
    River_player_1_action_0_what_0 = 524
    River_player_1_action_0_what_1 = 525
    River_player_1_action_0_what_2 = 526
    River_player_1_action_1_how_much = 527
    River_player_1_action_1_what_0 = 528
    River_player_1_action_1_what_1 = 529
    River_player_1_action_1_what_2 = 530
    River_player_2_action_0_how_much = 531
    River_player_2_action_0_what_0 = 532
    River_player_2_action_0_what_1 = 533
    River_player_2_action_0_what_2 = 534
    River_player_2_action_1_how_much = 535
    River_player_2_action_1_what_0 = 536
    River_player_2_action_1_what_1 = 537
    River_player_2_action_1_what_2 = 538
    River_player_3_action_0_how_much = 539
    River_player_3_action_0_what_0 = 540
    River_player_3_action_0_what_1 = 541
    River_player_3_action_0_what_2 = 542
    River_player_3_action_1_how_much = 543
    River_player_3_action_1_what_0 = 544
    River_player_3_action_1_what_1 = 545
    River_player_3_action_1_what_2 = 546
    River_player_4_action_0_how_much = 547
    River_player_4_action_0_what_0 = 548
    River_player_4_action_0_what_1 = 549
    River_player_4_action_0_what_2 = 550
    River_player_4_action_1_how_much = 551
    River_player_4_action_1_what_0 = 552
    River_player_4_action_1_what_1 = 553
    River_player_4_action_1_what_2 = 554
    River_player_5_action_0_how_much = 555
    River_player_5_action_0_what_0 = 556
    River_player_5_action_0_what_1 = 557
    River_player_5_action_0_what_2 = 558
    River_player_5_action_1_how_much = 559
    River_player_5_action_1_what_0 = 560
    River_player_5_action_1_what_1 = 561
    River_player_5_action_1_what_2 = 562
    Position_is_btn = 563
    Position_is_sb = 564
    Position_is_bb = 565
    Position_is_utg = 566
    Position_is_mp = 567
    Position_is_co = 568
    Win_probability = 569
    Player_0_is_tight = 570
    Player_0_is_aggressive = 571
    Player_0_is_balanced_or_unknown = 572
    Player_1_is_tight = 573
    Player_1_is_aggressive = 574
    Player_1_is_balanced_or_unknown = 575
    Player_2_is_tight = 576
    Player_2_is_aggressive = 577
    Player_2_is_balanced_or_unknown = 578
    Player_3_is_tight = 579
    Player_3_is_aggressive = 580
    Player_3_is_balanced_or_unknown = 581
    Player_4_is_tight = 582
    Player_4_is_aggressive = 583
    Player_4_is_balanced_or_unknown = 584
    Player_5_is_tight = 585
    Player_5_is_aggressive = 586
    Player_5_is_balanced_or_unknown = 587


def make_enum__AugmentedObservationFeatureColumns():
    env = init_wrapped_env(AugmentObservationWrapper,
                           [100, 125, 150, 175, 200, 250])
    for k, v in env.obs_idx_dict.items():
        kr = k.capitalize().replace("0th", "First").replace("1th", "Second").replace(
            "2th", "Third").replace("3th", "Fourth").replace("4th", "Fifth").replace(
            "5th", "Sixth")
        print(f'{kr} = {v}')


if __name__ == "__main__":
    make_enum__AugmentedObservationFeatureColumns()
