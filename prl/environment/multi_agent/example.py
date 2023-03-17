"""Test Reward on full episode with known epxected results."""
from typing import Union

import numpy as np
import pytest
from ray.rllib.env import EnvContext

from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.multi_agent.utils import make_multi_agent_env


@pytest.fixture
def multi_2pl_2envs():
    env_cfg = {'env_wrapper_cls': AugmentObservationWrapper,
               'n_players': 2,
               'starting_stack_size': 1000,
               'num_envs': 2
               }
    env_cls = make_multi_agent_env(env_cfg)
    dummy_ctx = EnvContext(env_config={},
                           worker_index=0,  # 0 for local worker, >0 for remote workers.
                           vector_index=0,  # uniquely identify env when there are multiple envs per worker
                           remote=False,  # individual sub-envvs should be @ray.remote actors
                           num_workers=0,  # 0 for only local
                           recreated_worker=False
                           )
    return env_cls(dummy_ctx)


def assert_obs_has_empty_action_history(obs: Union[list, np.array]):
    return True


def test_2pl_2envs_episode_rewards(multi_2pl_2envs):
    obs_dict = multi_2pl_2envs.reset()
    action_dict = {0: 0, 1: 0}
    [assert_obs_has_empty_action_history(obs) for obs in obs_dict.values()]
    obs_dict_preflop_1 = multi_2pl_2envs.step(action_dict)
    [assert_obs_has_empty_action_history(obs) for obs in obs_dict.values()]

