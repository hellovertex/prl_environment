from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from prl.environment.steinberger.PokerRL import NoLimitHoldem


def make_wrapped_env(n_players, starting_stack_sizes):
    # set env_args such that new starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=starting_stack_sizes,
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env)
    env.overwrite_args(args)
    return env


env = make_wrapped_env(3, [2000, 2000, 2000])
obs, _, done, _ = env.reset()
print(f'done = {done}')
print(f'sum obs = {sum(obs[:166])}')
print(env.env.board)
obs, _, done, _ = env.step((2,200))
print(f'done = {done}')
print(f'sum obs = {sum(obs[:166])}')
print(env.env.board)
obs, _, done, _ = env.step((1,-1))
print(f'done = {done}')
print(f'sum obs = {sum(obs[:166])}')
print(env.env.board)
obs, _, done, _ = env.step((1,-1    ))
print(env.env.board)
print(f'done = {done}')
print(f'sum obs = {sum(obs[:166])}')
print(env.obs_idx_dict['4th_board_card_suit_3'])
print(env.env.board)