from gymnasium.envs.registration import register

register(
id="gym_examples/FootyFreeKick-v0",
entry_point="gym_examples.envs:FootyFreeKickEnv",
max_episode_steps=300,
)