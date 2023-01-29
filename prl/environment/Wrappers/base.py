import enum

import numpy as np


class ActionSpace(enum.IntEnum):
    """Under Construction"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_MIN_OR_3BB = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5


class EnvWrapperBase:

    def __init__(self, env):
        """
        Args:
            env:   The environment instance to be wrapped
        """
        self.env = env

    def reset(self, config):
        """Reset the environment with a new config.
        Signals environment handlers to reset and restart the environment using
        a config dict.
        Args:
          config: dict, specifying the parameters of the environment to be
            generated. May contain state_dict to generate a deterministic environment.
        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.
        Args:
          action: object, mapping to an action taken by an agent.
        Returns:
          observation: object, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.
        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class WrapperPokerRL(EnvWrapperBase):

    def __init__(self, env):
        super().__init__(env)
        self._player_hands = []
        self._player_who_gets_observation = None

    def reset(self, config=None):
        """
        Resets the state of the game to the standard beginning of the episode. If specified in the args passed,
        stack size randomization is applied in the new episode. If deck_state_dict is not None, the cards
        and associated random variables are synchronized FROM the given environment, so that when .step() is called on
        each of them, they produce the same result.

        Args:
            config["deck_state_dict"]:      Optional.
                                            If an instance of a PokerEnv subclass is passed, the deck, holecards, and
                                            board in this instance will be synchronized from the handed env cls.
        """
        # maybe initialize wrapper variables
        self._before_reset(config)

        # maybe initialize deck from state --> a priori know hands and board cards
        deck_state_dict = None
        if config is not None:
            deck_state_dict = config['deck_state_dict']

        # reset env
        env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)

        # Convenient access to hand cards of each player
        self._player_hands = []
        for i in range(self.env.N_SEATS):
            self._player_hands.append(self.env.get_hole_cards_of_player(i))
        self._after_reset()

        # pass env_obs to the next wrapper to either return or further augment the observation
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def int_action_to_tuple_action(self, a):
        if a == ActionSpace.FOLD:
            return (0, -1)
        elif a == ActionSpace.CHECK_CALL:
            # check or call appropriate size (automatically via pot_size)
            return (1, -1)  # when calling with pot_size, the env scales it down to the appropriate call size
        elif ActionSpace.RAISE_MIN_OR_3BB <= a <= ActionSpace.ALL_IN:
            pot_size = self.env.get_all_winnable_money()
            if a == ActionSpace.RAISE_MIN_OR_3BB:
                return (2, self.env._get_current_total_min_raise())
            elif a == ActionSpace.RAISE_HALF_POT:
                return (2, int(pot_size / 2))  # residuals of division shouldnt cause problems
            elif a == ActionSpace.RAISE_POT:
                return (2, pot_size)
            elif a == ActionSpace.ALL_IN:
                return (2, self.env.current_player.stack)
        return a

    def step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """
        if isinstance(action, int) or isinstance(action, np.integer):
            action = self.int_action_to_tuple_action(action)
        # callbacks in derived class
        self._before_step(action)
        # step environment
        env_obs, rew_for_all_players, done, info = self.env.step(action)

        self._after_step(action)
        # call get_current_obs of derived class
        return self._return_obs(env_obs=env_obs,
                                rew_for_all_players=rew_for_all_players,
                                done=done,
                                info=info)

    def step_from_processed_tuple(self, action):
        """
        Steps the environment from a tuple (action, num_chips,).

        Returns:
            obs, reward, done, info
        """
        return self.step(action)

    def step_raise_pot_frac(self, pot_frac):
        """
        Steps the environment from a fractional pot raise instead of an action as usually specified.

        Returns:
            obs, reward, done, info
        """
        processed_action = (2, self.env.get_fraction_of_pot_raise(
            fraction=pot_frac, player_that_bets=self.env.current_player))

        return self.step(processed_action)

    def _return_obs(self, rew_for_all_players, done, info, env_obs=None):
        return self.get_current_obs(env_obs=env_obs, done=done), rew_for_all_players, done, info

    # _______________________________ Override to augment observation ________________________________

    def _before_step(self, action):
        """Implement this e.g. to process or store action before stepping environment.
        This can be useful if you wrap the environment with functions that keep a buffer
        of past actions"""
        pass

    def _before_reset(self, config):
        """Implement this. to set/init wrapper variables based on config before
        resetting environment. This may be useful if you want to reset player hand
        dictionaries which are stored directly in the wrapper, based on the hand cards
        with which you initialized the env. """
        pass

    def _after_step(self, action):
        """This is called before vectorizing the observation. You can register callbacks here"""
        pass

    def _after_reset(self):
        """This is called before vectorizing the observation. You can register callbacks here"""
        pass

    def get_current_obs(self, env_obs, *args, **kwargs):
        raise NotImplementedError
