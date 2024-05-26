from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class Actions(Enum):
    UP = 0
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UPRIGHT = 4
    UPLEFT = 5
    DOWNRIGHT = 6
    DOWNLEFT = 7


class FootyFreeKickEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        self.window_size = 10 * np.array([105.0, 68.0], dtype=np.float32)  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.concatenate([self.window_size, self.window_size], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        # The agent is a circle with radius `agent_radius`
        self._agent_radius = 40.0
        # Location of the center of the agent
        self._agent_location = np.array([-1.0, -1.0], dtype=np.float32)
        # The target is a circle with radius `target_radius`
        self._target_radius = 20.0
        # Location of the center of the target
        self._target_location = np.array([-1.0, -1.0], dtype=np.float32)
        # Velocity of the target
        self._target_velocity = np.array([0., 0.0], dtype=np.float32)
        # The goal is a rectangle with width `goal_width` and height `goal_height`
        self._goal_width = self.window_size[1] // 8
        self._goal_height = self.window_size[1] // 4
        # Location of the center of the goal
        self._goal_location = np.array(
            [self.window_size[0] - self._goal_width, self.window_size[1] // 2],
            dtype=np.float32
        )
        self._agent_speed = 5.0  # The agent moves 3 pixels per step

        # We have 8 actions, corresponding to "up", "right", "left", "down",
        # "upright", "upleft", "downright", "downleft"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_agent_velocity = {
            Actions.RIGHT.value: self._agent_speed * np.array([1, 0], dtype=np.float32),
            Actions.UP.value: self._agent_speed * np.array([0, 1], dtype=np.float32),
            Actions.LEFT.value: self._agent_speed * np.array([-1, 0], dtype=np.float32),
            Actions.DOWN.value: self._agent_speed * np.array([0, -1], dtype=np.float32),
            Actions.UPRIGHT.value: self._agent_speed * np.array([1, 1], dtype=np.float32),
            Actions.UPLEFT.value: self._agent_speed * np.array([-1, 1], dtype=np.float32),
            Actions.DOWNRIGHT.value: self._agent_speed * np.array([1, -1], dtype=np.float32),
            Actions.DOWNLEFT.value: self._agent_speed * np.array([-1, -1], dtype=np.float32),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # %%
    # Constructing Observations From Environment States
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Since we will need to compute observations both in ``reset`` and
    # ``step``, it is often convenient to have a (private) method ``_get_obs``
    # that translates the environment’s state into an observation. However,
    # this is not mandatory and you may as well compute observations in
    # ``reset`` and ``step`` separately:

    def _get_obs(self):
        return np.concatenate([self._agent_location, self._target_location], dtype=np.float32)

    # %%
    # We can also implement a similar method for the auxiliary information
    # that is returned by ``step`` and ``reset``. In our case, we would like
    # to provide the manhattan distance between the agent and the target:

    def _get_info(self) -> dict:
        """
        Return the Frobenius norm of the difference between the agent's and the target's location.

        Returns:
            dict: A dictionary containing the distance between the agent and the target.
        """
        return {
            "agent_target_distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=None
            ),
            "target_goal_location": np.linalg.norm(
                self._target_location - self._goal_location, ord=None
            ),
        }

    # %%
    # Oftentimes, info will also contain some data that is only available
    # inside the ``step`` method (e.g., individual reward terms). In that case,
    # we would have to update the dictionary that is returned by ``_get_info``
    # in ``step``.

    # %%
    # Reset
    # ~~~~~
    #
    # The ``reset`` method will be called to initiate a new episode. You may
    # assume that the ``step`` method will not be called before ``reset`` has
    # been called. Moreover, ``reset`` should be called whenever a done signal
    # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
    # initialize any random number generator that is used by the environment
    # to a deterministic state. It is recommended to use the random number
    # generator ``self.np_random`` that is provided by the environment’s base
    # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
    # worry much about seeding, *but you need to remember to call
    # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
    # correctly seeds the RNG. Once this is done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s
    # location and the random sample target positions, until it does not
    # coincide with the agent’s position.
    #
    # The ``reset`` method should return a tuple of the initial observation
    # and some auxiliary information. We can use the methods ``_get_obs`` and
    # ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([
            self.np_random.uniform(0, high=self._goal_location[0] - 3 * self._goal_width),
            self.np_random.uniform(0, high=self.window_size[1])
        ],
            dtype=np.float32
        )

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while self._target_location[0] - self._goal_width <= self._agent_location[0]:
            self._target_location = np.array([
                self.np_random.uniform(0, high=self._goal_location[0] - 3 * self._goal_width),
                self.np_random.uniform(0, high=self.window_size[1])
            ],
                dtype=np.float32
            )
        self._target_velocity = np.array([0, 0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 5-tuple
    # ``(observation, reward, terminated, truncated, info)``. See
    # :meth:`gymnasium.Env.step`. Once the new state of the environment has
    # been computed, we can check whether it is a terminal state and we set
    # ``done`` accordingly. Since we are using sparse binary rewards in
    # ``GridWorldEnv``, computing ``reward`` is trivial once we know
    # ``done``.To gather ``observation`` and ``info``, we can again make
    # use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
        agent_velocity = self._action_to_agent_velocity[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + agent_velocity, np.array([0, 0], dtype=np.float32), self.window_size
        )

        # Get the observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Kick the ball if the agent is close enough to the target
        if info["agent_target_distance"] < self._agent_radius + self._target_radius:
            self._target_velocity += agent_velocity
        self._target_location = self._target_location + self._target_velocity

        # An episode is done iff the target has reached te goal
        terminated = (
                self._target_location[0] >= self._goal_location[0]
                and np.abs(self._target_location[1] - self._goal_location[1]) <= self._goal_height / 2
        )
        reward = 1 if terminated else 0  # Sparse binary reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    # %%
    # Rendering
    # ~~~~~~~~~
    #
    # Here, we are using PyGame for rendering. A similar approach to rendering
    # is used in many environments that are included with Gymnasium and you
    # can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (int(self.window_size[0]), int(self.window_size[1]))
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((int(self.window_size[0]), int(self.window_size[1])))
        canvas.fill((1, 135, 73))
        self.load_image(
            canvas,
            'gym_examples/gym_examples/envs/images/Field.jpg',
            (int(self.window_size[0]), int(self.window_size[1])),
            None
        )
        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            self._target_location.astype(int),
            int(self._target_radius),
        )
        # Then we draw the agent
        self.load_image(
            canvas,
            'gym_examples/gym_examples/envs/images/domi.png',
            (2*int(self._agent_radius),2*int(self._agent_radius)),
            tuple(self._agent_location.astype(int)-np.array([self._agent_radius, self._agent_radius], dtype=int))
        )
        # Finally, we draw the goal
        pygame.draw.rect(
            canvas,
            (220, 20, 60),
            pygame.Rect(
                list(self._goal_location.astype(int) - np.array([0, self._goal_height / 2], dtype=int)),
                (int(self._goal_width), int(self._goal_height)),
            ),
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def load_image(self, canvas, path, size, topleft=None):
        img = pygame.image.load(path)
        transimg = pygame.transform.scale(img, size)
        imgrect = transimg.get_rect()
        if topleft is not None:
            imgrect.topleft = topleft
        canvas.blit(transimg, imgrect)

    # %%
    # Close
    # ~~~~~
    #
    # The ``close`` method should close any open resources that were used by
    # the environment. In many cases, you don’t actually have to bother to
    # implement this method. However, in our example ``render_mode`` may be
    # ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# %%
# In other environments ``close`` might also close files that were opened
# or release other resources. You shouldn’t interact with the environment
# after having called ``close``.
