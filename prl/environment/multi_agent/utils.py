import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiBinary
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
            self.env_wrapped = self._single_env()
            self._num_agents = self._n_players  # keep naming consistency with rllib
            self.agents = env_config['agents']
            self.action_space = self.env_wrapped.action_space  # not batched, rllib wants that to be for single env
            # self.observation_space = self.agents[
            #     0].observation_space  # not batched, rllib wants that to be for single env
            # self.observation_space.dtype = np.float32
            # todo fix following monkeypatch:
            obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)
            self.observation_space = obs_space
            if 'mask_legal_moves' in env_config:
                if env_config['mask_legal_moves']:
                    self.observation_space = gym.spaces.Dict({
                        'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
                        'legal_moves': MultiBinary(3)  # one-hot encoded [FOLD, CHECK_CALL, RAISE]
                    })

            self._agent_ids = list(self.agents.keys())

            MultiAgentEnv.__init__(self)

        def _single_env(self):
            return init_wrapped_env(self._env_cls, [self._starting_stack_size for _ in range(self._n_players)])

        @override(MultiAgentEnv)
        def reset(self):
            # return only obs, nothing else
            obs, _, _, info = self.env_wrapped.reset(config=None)
            next_to_act = self.env_wrapped.env.current_player.seat_id
            legal_moves = np.array([0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            return {next_to_act: {'obs': obs, 'legal_moves': legal_moves}}

        @override(MultiAgentEnv)
        def step(self, action_dict):
            """When implementing your own MultiAgentEnv, note that you should only return those agent IDs in an
            observation dict, for which you expect to receive actions in the next call to step().

            From the rllib docs:
            # Env, in which two agents step in sequence (tuen-based game).
            # The env is in charge of the produced agent ID. Our env here produces
            # agent IDs: "player1" and "player2".
            env = TicTacToe()

            # Observations are a dict mapping agent names to their obs. Only those
            # agents' names that require actions in the next call to `step()` should
            # be present in the returned observation dict (here: one agent at a time).
            print(env.reset())
            # ... {
            # ...   "player1": [[...]],
            # ... }

            # In the following call to `step`, only those agents' actions should be
            # provided that were present in the returned obs dict:
            new_obs, rewards, dones, infos = env.step(actions={"player1": ...})

            # Similarly, new_obs, rewards, dones, etc. also become dicts.
            # Note that only in the `rewards` dict, any agent may be listed (even those that have
            # not(!) acted in the `step()` call). Rewards for individual agents will be added
            # up to the point where a new action for that agent is needed. This way, you may
            # implement a turn-based 2-player game, in which player-2's reward is published
            # in the `rewards` dict immediately after player-1 has acted.
            print(rewards)
            # ... {"player1": 0, "player2": 0}

            # Individual agents can early exit; The entire episode is done when
            # dones["__all__"] = True.
            print(dones)
            # ... {"player1": False, "__all__": False}

            # In the next step, it's player2's turn. Therefore, `new_obs` only container
            # this agent's ID:
            print(new_obs)
            # ... {
            # ...   "player2": [[...]]
            # ... }
            """

            observations, rewards, dones, infos = {}, {}, {}, {}

            # make sure only one player acted and step environment with its action
            have_played = list(action_dict.keys())
            has_played = have_played[0]
            assert len(have_played) == 1
            action = action_dict[has_played]
            obs, rews, done, info = self.env_wrapped.step(action)
            legal_moves = np.array([0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            # assign returned observation to new player,
            # dones to old player and reward for all players
            next = self.env_wrapped.env.current_player.seat_id
            observations[next] = {'obs': obs, 'legal_moves': legal_moves}
            for i, v in enumerate(rews):
                rewards[i] = v / self._n_players  # normalize v because rllib stacks rewards per round
            dones[has_played] = done
            dones["__all__"] = done
            return observations, rewards, dones, infos

        @override(MultiAgentEnv)
        def render(self, mode='human'):
            pass

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            """The name 'agent_ids' is taken from rllib's core fn, although I think env_ids would be better"""
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            return all(self.action_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            obs_dict = list(x.values())[0]
            if type(obs_dict) == dict:
                return self.observation_space['obs'].contains(obs_dict['obs'])
            else:
                return all(self.observation_space.contains(val) for val in x.values())

    return _RLLibSteinbergerEnv
