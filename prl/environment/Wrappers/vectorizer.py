import enum
from typing import List, Optional

import numpy as np

from prl.environment.steinberger.PokerRL.game.Poker import Poker

FOLD = 0
CHECK_CALL = 1
RAISE = 2


class AgentObservationType(enum.IntEnum):
    CARD_KNOWLEDGE = 1  # default where agent only sees his own cards and the board
    SEER = 2  # agent sees all player cards


class Vectorizer:
    """ Abstract Vectorizer Interface. All vectorizers should be derived from this base class
    and implement the method "vectorize"."""

    def vectorize(self, obs, *args, **kwargs):
        """todo"""
        raise NotImplementedError


class CanonicalVectorizer(Vectorizer):
    """Docstring"""

    # todo write vectorizer, such that it can be associated with exactly one env class
    def __init__(self,
                 num_players,
                 obs_idx_dict,
                 btn_pos,  # self.env.BTN_POS, always equal to 0
                 max_players=6,
                 n_ranks=13,
                 n_suits=4,
                 n_board_cards=5,
                 n_hand_cards=2,
                 use_zero_padding=True,
                 mode=AgentObservationType.CARD_KNOWLEDGE):
        # --- Utils --- #
        # todo [optional] switch to use num_players instead of max_players
        self._agent_observation_type = mode
        self._use_zero_padding = use_zero_padding
        self._next_player_who_gets_observation = None
        self.num_players = num_players
        self.obs_idx_dict = obs_idx_dict
        self._max_players = max_players
        self.n_ranks = n_ranks
        self.n_suits = n_suits
        self._n_board_cards = n_board_cards
        self._n_hand_cards = n_hand_cards
        self._n_stages = len(['preflop', 'flop', 'turn', 'river'])
        card = List[int]  # 2 cards
        hand = List[card]
        self._player_hands: Optional[List[hand]] = None
        self._action_history = None
        # btn_idx is equal to current player offset, since button is at index 0 inside environment
        # but we encode observation such that player is at index 0
        self._btn_idx = btn_pos
        max_actions_per_stage_per_player = 2
        max_actions = max_actions_per_stage_per_player * self._n_stages * self._max_players
        self._bits_per_action = len(['fold', 'check/call', 'bet/raise']) \
                                + len(['last_action_how_much'])
        self._bits_stats_per_player_original = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                               + len(['side_pot_rank_p0_is_']) * self.num_players
        self._bits_stats_per_player = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                      + len(['side_pot_rank_p0_is_']) * self._max_players
        self._bits_per_card = n_ranks + n_suits  # 13 ranks + 4 suits
        # --- Observation Bits --- #
        self._bits_table = len(['ante',
                                'small_blind',
                                'big_blind',
                                'min_raise',
                                'pot_amt',
                                'total_to_call'])
        self._bits_next_player = self._max_players
        self._bits_stage = self._n_stages
        self._bits_side_pots = self._max_players
        self._bits_player_stats = self._bits_stats_per_player * self._max_players
        self._bits_player_stats_original = self._bits_stats_per_player_original * self.num_players
        self._bits_board = self._bits_per_card * n_board_cards  # 3 cards flop, 1 card turn, 1 card river
        self._bits_player_hands = self._max_players * n_hand_cards * self._bits_per_card
        self._bits_action_history_one_player = max_actions_per_stage_per_player * self._n_stages * self._bits_per_action
        self._bits_action_history = self._bits_action_history_one_player * self._max_players

        # --- Offsets --- #
        self._start_table = 0
        self._start_next_player = self._bits_table
        self._start_stage = self._start_next_player + self._bits_next_player
        self._start_side_pots = self._start_stage + self._bits_stage
        self._start_player_stats = self._start_side_pots + self._bits_side_pots
        self._start_board = self._start_player_stats + self._bits_player_stats
        self._start_player_hands = self._start_board + self._bits_board
        self._start_action_history = self._start_player_hands + self._bits_player_hands
        self.offset = None

        # --- Number of features --- #
        self._obs_len = self._bits_table \
                        + self._bits_next_player \
                        + self._bits_stage \
                        + self._bits_side_pots \
                        + self._bits_player_stats \
                        + self._bits_board \
                        + self._bits_player_hands \
                        + self._bits_action_history
        self._obs = np.zeros(self._obs_len)

    @property
    def agent_observation_mode(self):
        return self._agent_observation_type

    @agent_observation_mode.setter
    def agent_observation_mode(self, mode: AgentObservationType):
        self._agent_observation_type = mode

    def vectorized_observation_shape(self):
        return self._obs.shape

    def encode_table(self, obs):
        """Example:
                    ante:   0.0
             small_blind:   0.05
               big_blind:   0.1
               min_raise:   0.2
                 pot_amt:   0.0
           total_to_call:   0.1
        """

        self.offset = 0 + self._bits_table
        assert self.offset == self._start_next_player
        # copy unchanged
        self._obs[0:self.offset] = obs[0:self.offset]

    def encode_next_player(self, obs):
        """Example:
            p0_acts_next:   1.0
            p1_acts_next:   0.0
            p2_acts_next:   0.0
          Since this is relative to observer, p0_acts_next will always be 1.
          We could remove this but keep it for consistency with the Steinberger Env.
        """
        self.offset += self._bits_next_player
        assert self.offset == self._start_stage
        # original obs indices
        start_orig = self.obs_idx_dict['p0_acts_next']
        end_orig = start_orig + self.num_players
        # extract from original observation
        bits = obs[start_orig:end_orig]
        bits = np.roll(bits, -self._next_player_who_gets_observation)
        # zero padding
        bits = np.pad(bits, (0, self._max_players - self.num_players), 'constant')
        # copy from original observation with zero padding
        self._obs[self._start_next_player:self.offset] = bits

    def encode_stage(self, obs):
        """Example:
           round_preflop:   1.0
              round_flop:   0.0
              round_turn:   0.0
             round_river:   0.0
        """
        self.offset += self._bits_stage
        assert self.offset == self._start_side_pots
        # original obs indices
        start_orig = self.obs_idx_dict['round_preflop']
        end_orig = start_orig + self._n_stages
        # extract from original observation
        bits = obs[start_orig:end_orig]
        # zero padding is not necessary
        # copy from original observation without zero padding
        self._obs[self._start_stage:self.offset] = bits

    def encode_side_pots(self, obs):
        """Example:
            side_pot_0:   0.0
            side_pot_1:   0.0
            side_pot_2:   0.0
        """
        self.offset += self._bits_side_pots
        assert self.offset == self._start_player_stats
        # original obs indices
        try:
            # if side pots are available in original observation
            start_orig = self.obs_idx_dict['side_pot_0']
            end_orig = start_orig + self.num_players
            bits = obs[start_orig:end_orig]
        except Exception as e:
            bits = np.zeros(self.num_players)

        # move self to index 0
        bits = np.roll(bits, -self._next_player_who_gets_observation)

        # zero padding
        bits = np.pad(bits, (0, self._max_players - self.num_players), 'constant')

        # copy from original observation with zero padding
        self._obs[self._start_side_pots:self.offset] = bits

    def encode_player_stats(self, obs):
        """Example:
        stack_p0:   0.9
                 curr_bet_p0:   0.1
    has_folded_this_episode_p0:   0.0
                 is_allin_p0:   0.0
       side_pot_rank_p0_is_0:   0.0
       side_pot_rank_p0_is_...:   0.0
       side_pot_rank_p0_is_n:   0.0

                    stack_p1:   0.95
                 curr_bet_p1:   0.05
    has_folded_this_episode_p1:   0.0
                 is_allin_p1:   0.0
       side_pot_rank_p1_is_0:   0.0
       side_pot_rank_p1_is_...:   0.0
       side_pot_rank_p1_is_n:   0.0
       """
        self.offset += self._bits_player_stats
        assert self.offset == self._start_board
        # original obs indices
        start_orig = self.obs_idx_dict['stack_p0']
        end_orig = start_orig + self._bits_player_stats_original
        # extract from original observation
        bits = np.array(obs[start_orig:end_orig])
        # zero padding in between players for additional side pot info
        bits_per_player = np.split(bits, self.num_players)
        bits_to_pad_in_between = np.zeros(self._max_players - self.num_players)
        padded_in_between = np.array([np.append(s, bits_to_pad_in_between) for s in bits_per_player])
        # roll
        roll_by = -self._next_player_who_gets_observation
        padded_in_between = np.roll(padded_in_between, roll_by, axis=0).flatten()
        # zero padding until the end
        padded_in_between = np.pad(padded_in_between, (0, self._bits_player_stats - len(padded_in_between)), 'constant')

        # copy from original observation with zero padding
        self._obs[self._start_player_stats:self.offset] = padded_in_between

    def encode_board(self, obs):
        """Example:
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
       """
        self.offset += self._bits_board
        assert self.offset == self._start_player_hands
        # original obs indices
        start_orig = self.obs_idx_dict['0th_board_card_rank_0']
        end_orig = start_orig + self._bits_board
        # extract from original observation
        bits = obs[start_orig:end_orig]
        # zero padding is not necessary
        # copy from original observation without zero padding
        self._obs[self._start_board:self.offset] = bits

    # IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
    # IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
    # IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
    # IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
    # IDX_BOARD_START = 82  #
    # IDX_BOARD_END = 167  #
    # CARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
    # BOARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
    #                               'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
    #                               'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
    #                               '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
    #                               '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
    #                               'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
    #                               'Q', 'K', 'A', 'h', 'd', 's', 'c'])
    # RANK = 0
    # SUITE = 1
    # def card_bit_mask_to_int(self, c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[List[int], List[int]]:
    #     c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
    #     c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
    #     board = BOARD_BITS_TO_STR[board_mask.astype(bool)]
    #     # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
    #     board_cards = []
    #     for i in range(0, sum(board_mask) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
    #         board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])
    #
    #     return [c0_1d, c1_1d], board_cards
    #
    # def look_at_cards(self, obs: np.array) -> Tuple[List[int], List[int]]:
    #     c0_bits = obs[IDX_C0_0:IDX_C0_1].astype(bool)
    #     c1_bits = obs[IDX_C1_0:IDX_C1_1].astype(bool)
    #     board_bits = obs[IDX_BOARD_START:IDX_BOARD_END].astype(int)  # bit representation
    #     return self.card_bit_mask_to_int(c0_bits, c1_bits, board_bits)

    def encode_player_hands(self, obs):
        """Example:"""
        # todo this returns zero if player hands have not been set by ditionary,
        #  instead get them from observation
        self.offset += self._bits_player_hands
        assert self.offset == self._start_action_history
        # move own cards to index 0
        roll_by = -self._next_player_who_gets_observation
        rolled_cards = np.roll(self._player_hands, roll_by, axis=0).reshape(-1, self._n_hand_cards)
        # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ -127  -127], [ -127  -127]]
        # replace NAN with 0
        # todo: if not SEER mode, set rolled_cards[2:] indices to zero
        # the following line introduces a bug:  -- keep for reference, do NOT uncomment
        # rolled_cards[np.where(rolled_cards == Poker.CARD_NOT_DEALT_TOKEN_1D)] = 0
        # if not self._agent_observation_type == AgentObservationType.SEER:
        if not self.done:
            # ignore all other players cards -> the agent should not see these
            rolled_cards = rolled_cards[:2]
        # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ 0  0], [ 0  0]]

        # initialize hand_bits to 0
        card_bits = self.n_ranks + self.n_suits
        # hand_bits = [0] * self._n_hand_cards * self.num_players * card_bits
        hand_bits = [0] * self._bits_player_hands
        # overwrite one_hot card_bits
        # print(f'NUM_PLAYERS INSIDE VEC = {self.num_players}')
        for n_card, card in enumerate(rolled_cards):
            offset = card_bits * n_card
            # set rank for visible cards:
            if card[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                hand_bits[card[0] + offset] = 1
                # set suit
                hand_bits[card[1] + offset + self.n_ranks] = 1
        # zero padding
        # hand_bits = np.pad(hand_bits, (0,  - len(hand_bits)), 'constant')
        # hand_bits = np.resize(hand_bits, self._bits_player_hands)
        self._obs[self._start_player_hands:self.offset] = hand_bits

    @staticmethod
    def _vectorize_deque(deq, normalization):
        bits_per_action = 4  # ==len([IS_FOLD, IS_CHECK_CALL, IS_RAISE, ACTION_HOW_MUCH])
        bits = [0 for _ in range(deq.maxlen * bits_per_action)]
        for i, action in enumerate(deq):
            how_much = 0 if not action[0] == RAISE else action[1] / normalization
            idx_action = 1 + action[0]
            bits[i * bits_per_action] = how_much
            bits[i * bits_per_action + idx_action] = 1
        return bits

    def encode_action_history(self, normalization):
        """Example:"""
        self.offset += self._bits_action_history
        assert self.offset == self._obs_len
        d = self._action_history.deque
        _bits_action_history = 192
        pids = [i for i in range(self.num_players)]
        pids = np.roll(pids, - self._next_player_who_gets_observation)
        bits = []
        # iterate all players, get preflop actions
        # iterate all players, get flop actions
        # iterate all players, get turn actions
        # iterate all players, get river actions
        for stage in ['preflop', 'flop', 'turn', 'river']:
            for pid in pids:
                bits.append(self._vectorize_deque(d[pid][stage], normalization))
            # append 0-bits for missing players
            for _ in range(self._max_players - self.num_players):
                bits.append([0, 0, 0, 0] * self._action_history.buffer_actions_per_stage())

        # pad flattened numpy array with zeros
        bits = np.array([bit for byte in bits for bit in byte])
        # bits = np.pad(bits, (0, self._bits_action_history - len(bits)), 'constant')
        self._obs[self._start_action_history:self.offset] = bits

    def vectorize(self, obs, _next_player_who_gets_observation=None, action_history=None, player_hands=None,
                  normalization=None, done=None):
        self.done = done
        # reset
        self._obs = np.zeros(self._obs_len)
        self.offset = None
        self._player_hands = player_hands
        self._action_history = action_history
        self._next_player_who_gets_observation = _next_player_who_gets_observation
        # encode
        self.encode_table(obs)
        self.encode_next_player(obs)
        self.encode_stage(obs)
        self.encode_side_pots(obs)
        self.encode_player_stats(obs)
        self.encode_board(obs)
        self.encode_player_hands(obs)
        self.encode_action_history(normalization)
        # append button relative to observer
        # [4,5,0,1,2,3] -> btn_idx = 2
        # [1,2,3,4,5,0] -> btn_idx = 5
        assert self.offset == self._obs_len
        # append offset to button at the end as a hotfix
        btn_one_hot_encoded = [0,0,0,0,0,0]
        btn_one_hot_encoded[self._next_player_who_gets_observation] = 1
        # return np.concatenate([self._obs, [self._next_player_who_gets_observation]])
        return np.concatenate([self._obs, btn_one_hot_encoded])
