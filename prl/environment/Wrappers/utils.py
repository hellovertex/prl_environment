from typing import TypeVar, Tuple, Type, List, Union

from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem

ENV_WRAPPER = TypeVar('ENV_WRAPPER', bound=EnvWrapperBase)


def init_wrapped_env(env_wrapper_cls: Type[EnvWrapperBase],
                     stack_sizes: List[float],
                     blinds: Tuple[int, int]=(25,50),  #  = [25, 50]
                     multiply_by=100,
                     scale_rewards=True,
                     disable_info=False) -> ENV_WRAPPER:  # Tuple[Wrapper, List[int]]:
    """
    Wraps a NoLimitHoldEm instance with a custom wrapper class.
    Returns the initialized (not reset yet!) environment, together with
    a list of integer starting stacks.
    i) Use multiplier of 100 to convert two-decimal floats to integer
    ii) Assumes Btn is at stack_sizes index 0.
    :param env_wrapper_cls: Custom implementation of NoLimitHoldem-Wrapper
    :param stack_sizes: List of starting stack sizes. Starts with Button.

    # keep this to make sure nobody forgets removing the decimals
    :param multiply_by: Default is 100 to convert two-decimal floats to int.
    :return: Returns the initialized (not reset yet!) environment, together with
    a list of starting stacks. Starting stacks begin with the BTN.
    """
    # get starting stacks, starting with button at index 0
    starting_stack_sizes_list = [round(float(stack) * multiply_by) for stack in stack_sizes]

    # make args for env
    args = NoLimitHoldem.ARGS_CLS(n_seats=len(stack_sizes),
                                  scale_rewards=scale_rewards,
                                  use_simplified_headsup_obs=False,
                                  starting_stack_sizes_list=starting_stack_sizes_list)
    # return wrapped env instance
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env.SMALL_BLIND = blinds[0]
    env.BIG_BLIND = blinds[1]
    wrapped_env = env_wrapper_cls(env, disable_info=disable_info)
    return wrapped_env  # todo urgent replace:, starting_stack_sizes_list

