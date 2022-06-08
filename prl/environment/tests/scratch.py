# 2. roll stack sizes
import numpy as np

from prl.environment.Wrappers.prl_wrappers import ActionHistory
from prl.environment.tests.utils import make_wrapped_env

DEFAULT_STARTING_STACK_SIZE = 2000


# def vectorize_deque(deq, normalization):
#     bits_per_action = 4
#     bits = [0 for _ in range(deq.maxlen * bits_per_action)]
#     for i, action in enumerate(deq):
#         bits[i * bits_per_action + action[0]] = 1
#         bits[i * bits_per_action + 3] = action[1] / normalization
#     return bits
#
#
# def vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation):
#     normalization = 1
#     d = aoh.deque
#     _max_players = 6
#     _bits_action_history = 192
#     pids = [i for i in range(_max_players)]
#     pids = np.roll(pids, - _next_player_who_gets_observation)
#     bits = []
#
#     # iterate all players, get preflop actions
#     # iterate all players, get flop actions
#     # iterate all players, get turn actions
#     # iterate all players, get river actions
#     for stage in ['preflop', 'flop', 'turn', 'river']:
#         for pid in pids:
#             bits.append(vectorize_deque(d[pid][stage], normalization))
#     return bits
#
#
# def test_vectorizer_action_history_vectorization():
#     aoh = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
#     seat = 0
#     stage = 'preflop'
#     aoh.deque[seat][stage].append((2, 100))
#     vec = vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation=1)
#     print(vec)
#     aoh.deque[seat + 1][stage].append((0, 100))
#     vec = vectorize_aoh_relative_to_observer(aoh, _next_player_who_gets_observation=0)
#     print(vec)
#     # print([v for vc in vec for v in vc])


# create env with 2,3,6 players and step until river and assert action history for all of them
def test_two_player_action_history():
    n_players = 2
    starting_stack_sizes = [DEFAULT_STARTING_STACK_SIZE for _ in range(n_players)]
    env = make_wrapped_env(n_players, starting_stack_sizes)
    obs, _, _, _ = env.reset()
    # scenario 1: raise-fold
    # p1 raises
    obs, _, _, _ = env.step((2, 200))

    obs_idx_dict = env.obs_idx_dict
    obs_keys = [k for k in obs_idx_dict.keys()]
    start = obs_keys.index('preflop_player_0_action_0_how_much')
    end = obs_keys.index('river_player_5_action_1_what_2') + 1
    bits = obs[start:end]
    assert sum([bit for bit in bits if bit == 1]) == 1
    obs, _, _, _ = env.step((0, 100))
    bits = obs[start:end]
    assert sum([bit for bit in bits if bit == 1]) == 2
    bits_flop_and_later = obs[obs_keys.index('flop_player_0_action_0_how_much'):end]
    assert sum([bit for bit in bits_flop_and_later if bit == 1]) == 0
    # env.print_augmented_obs(obs)
    # todo:
    # scenario 2: call/check until river
    # scenario 3: raise call until river
    # scenario 4: pre flop all ins -> rundown till showdown


def test_three_player_action_history():
    # todo:
    # scenario 2: call/check until river
    # scenario 3: raise call until river
    # scenario 4: pre flop all ins -> rundown till showdown
    pass


def test_six_player_action_history():
    # todo:
    # scenario 2: call/check until river
    # scenario 3: raise call until river
    # scenario 4: pre flop all ins -> rundown till showdown
    pass


if __name__ == '__main__':
    # test_two_player_action_history()
    test_two_player_action_history()
