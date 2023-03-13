from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.vectorizer import AgentObservationType
from prl.environment.steinberger.PokerRL import NoLimitHoldem


def make_wrapped_env(n_players,
                     starting_stack_sizes,
                     agent_observation_mode=AgentObservationType.CARD_KNOWLEDGE):
    # set env_args such that new starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=starting_stack_sizes,
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env, agent_observation_mode=agent_observation_mode)
    env.overwrite_args(args)
    return env
