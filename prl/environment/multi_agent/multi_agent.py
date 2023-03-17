from tianshou.env import SubprocVectorEnv, BaseVectorEnv
from tianshou.policy.random import RandomPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy


class RayEnv:
    """https://docs.ray.io/en/latest/rllib/rllib-env.html"""
    # see util.make_multi_agent_env"""


class TianshouEnv(SubprocVectorEnv):
    """https://tianshou.readthedocs.io/en/master/_modules/tianshou/env/venvs.html#SubprocVectorEnv"""
    # requires stepping with a list of actions with length equal to number of environments
    # this means we have to parallelize the action sampling --> todo: write tianshou BasePolicy
    # todo ctd: the BasePolicy will get a list of observations, each corresponding to one of the n environments
    #  and will run monte carlo sampling on each of the cpu cores
    # RandomPolicy debugging how does batch look like after calling collect()
    # todo upd: need to return a dict with key `agent_id` and value `observation`
    # todo upd: in tianshouw.policy.multiagent.mapolicy.py, this observation is dispatched to each agent
    # todo upd: StakeImitationPolicy.forward() must have a way to compute and return Batch(actions) at each env step
    # todo upd2: pettingzoo environment holds reference to agent_idx which maps agent name to agent id
    #  MultiAgentPolicyManager gets a list of agents and env because it uses env to get these
    #  refs (only used in __init__). need PettingZooWrapper that
    #  provides 1) env.possible_agents 2) List of observation spaces by agent
    #  3) observation_dict = {
    #                 'agent_id': self.env.agent_selection,
    #                 'obs': observation['observation'],
    #                 'mask':
    #                 [True if obm == 1 else False for obm in observation['action_mask']]
    #             }
    #  env.last


class StableBaselinesEnv:
    """https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html"""
