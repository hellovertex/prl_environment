import numpy as np
#
# from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper
#
# from prl.environment.steinberger.PokerRL import NoLimitHoldem
#
#
# def make_wrapped_env(n_players, starting_stack_sizes):
#     # set env_args such that new starting stacks are used
#     args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
#                                   starting_stack_sizes_list=starting_stack_sizes,
#                                   use_simplified_headsup_obs=False)
#     env = NoLimitHoldem(is_evaluating=True,
#                         env_args=args,
#                         lut_holder=NoLimitHoldem.get_lut_holder())
#     env = AugmentObservationWrapper(env)
#     env.overwrite_args(args)
#     return env
#
#
# env = make_wrapped_env(3, [2000, 2000, 2000])
# obs, _, done, _ = env.reset()
# print(f'done = {done}')
# print(f'sum obs = {sum(obs[:166])}')
# print(env.env.board)
# obs, _, done, _ = env.step((2,200))
# print(f'done = {done}')
# print(f'sum obs = {sum(obs[:166])}')
# print(env.env.board)
# obs, _, done, _ = env.step((1,-1))
# print(f'done = {done}')
# print(f'sum obs = {sum(obs[:166])}')
# print(env.env.board)
# obs, _, done, _ = env.step((1,-1    ))
# print(env.env.board)
# print(f'done = {done}')
# print(f'sum obs = {sum(obs[:166])}')
# print(env.obs_idx_dict['4th_board_card_suit_3'])
# print(env.env.board)
from pydantic import BaseModel

from prl.environment.steinberger.PokerRL.game import Poker

RANK_DICT = {
    Poker.CARD_NOT_DEALT_TOKEN_1D: "",
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9",
    8: "T",
    9: "J",
    10: "Q",
    11: "K",
    12: "A"
}
SUIT_DICT = {
    Poker.CARD_NOT_DEALT_TOKEN_1D: "",
    0: "h",
    1: "d",
    2: "s",
    3: "c"
}

board = np.array([[2, 0], [12, 1], [6, 0], [-127, -127], [-127, -127]])
print(board)


class Card(BaseModel):
    name: str
    suit: int
    rank: int
    index: int  # 0<=4 for board position


cards = {}
for i, card in enumerate(board):
    print(card)
    cards[f'b{i}'] = Card(**{'name': RANK_DICT[card[0]] + SUIT_DICT[card[1]],
                             'suit': card[1],
                             'rank': card[0],
                             'index': i})
print(cards)