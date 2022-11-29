import numpy as np
from gym.spaces import Box
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

from prl.environment.Wrappers.utils import init_wrapped_env


def make_multi_agent_env(env_config):
    class _RLLibSteinbergerEnv(MultiAgentEnv):
        """Single Env holding multiple agents. Implements rllib.env.MultiAgentEnv to
        become vectorizable by rllib via `num_envs_per_worker` that can be increased via
        the algorithm config. """
        # see https://docs.ray.io/en/master/rllib/rllib-env.html
        def __init__(self, config: EnvContext):
            # config: EnvContext is not called here directly,
            # but it is called in ray.rllib.algorithm._get_env_id_and_creator
            # to set up ray remote workers
            self._n_players = env_config['n_players']
            self._starting_stack_size = env_config['starting_stack_size']
            self._env_cls = env_config['env_wrapper_cls']
            self._num_envs = env_config['num_envs']
            self.envs = [self._single_env() for _ in range(self._num_envs)]
            self.action_space = self.envs[0].action_space  # not batched, rllib wants that to be for single env
            # self.observation_space = self.envs[
            #     0].observation_space  # not batched, rllib wants that to be for single env
            # self.observation_space.dtype = np.float32
            self.observation_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)
            self._agent_ids = set(range(self._num_envs))  # _agent_ids name is enforced by rllib

            MultiAgentEnv.__init__(self)
            self.dones = set()
            self.rewards = {}
            self.acting_seat = None

        def _single_env(self):
            return init_wrapped_env(self._env_cls, [self._starting_stack_size for _ in range(self._n_players)])

        @override(MultiAgentEnv)
        def reset(self):
            self.dones = set()
            self.rewards = {}
            # return only obs nothing else, for each env
            self.acting_seat = 0
            return {i: env.reset()[0] for i, env in enumerate(self.envs)}

        def cumulate_rewards(self, rew):
            # update per agent reward
            if not self.acting_seat in self.rewards:
                self.rewards[self.acting_seat] = rew
            else:
                # update rew of agent on every sub env
                for key in self.rewards[self.acting_seat].keys():
                    self.rewards[self.acting_seat][key] += rew[key]

        @override(MultiAgentEnv)
        def step(self, action_dict):
            """When implementing your own MultiAgentEnv, note that you should only return those agent IDs in an
            observation dict, for which you expect to receive actions in the next call to step().



            """
            # agent A acts a --> step(a) --> obs, rew;  rew to A, obs to B?
            obs, rew, done, info = {}, {}, {}, {}

            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.envs[i].step(action)
                if i in self.rewards:
                    # self.rewards[i] += rew[i]
                    self.rewards[i] += 0
                else:
                    self.rewards[i] = rew[i]
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.envs)

            # do we have to do a vector of cumulative rewards and select the right one to return?
            # e.g.  return the added 20 from [10,20,-5,30] and reset it to [10,0,-5,30] etc?
            # todo: fix the rewarding of agents AFTER having debug setup ready
            # self.cumulate_rewards(rew)
            # self.acting_seat = (self.acting_seat + 1) % self._n_players
            # rew = self.rewards[self.acting_seat]
            # self.rewards[self.acting_seat]
            return obs, {0: 0.01, 1: 0.01}, done, info

        @override(MultiAgentEnv)
        def render(self, mode='human'):
            pass

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if not isinstance(x, dict):
                return False
            return all(self.action_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if not isinstance(x, dict):
                return False
            return all(self.observation_space.contains(val) for val in x.values())

    return _RLLibSteinbergerEnv

