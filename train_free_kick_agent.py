import os
import argparse
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import DQN
import torch
import gym_examples


# Create directories for logs and models
log_dir = 'logs'
model_dir = 'models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def train(num_cycles: int = 20):
    # Create environment
    env = gym.make('gym_examples/FootyFreeKick-v0', render_mode=None)

    # Set device and create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)

    # Train model
    for i in tqdm(range(num_cycles), desc='Training model'):
        model.learn(
            total_timesteps=25_000,
            log_interval=100,
            tb_log_name=f'FootyFreeKick-v0-PPO-{i}',
            progress_bar=True
        )
        # Save model
        model_path = os.path.join(model_dir, f'FootyFreeKick-v0-PPO-{i}')
        model.save(model_path)


def test(model_name: str = 'FootyFreeKick-v0-PPO-0'):
    # Create environment
    env = gym.make('gym_examples/FootyFreeKick-v0', render_mode='human')

    # Load model
    model_path = os.path.join(model_dir, model_name)
    model = DQN.load(model_path)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Train or test model.')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Train mode')
    parser_train.add_argument(
        'num_cycles',
        help='Number of training cycles',
        type=int,
        default=20
    )

    # Create the parser for the "test" command
    parser_test = subparsers.add_parser('test', help='Test mode')
    parser_test.add_argument(
        'model_name',
        help='Name of model to load',
        type=str,
        default='FootyFreeKick-v0-PPO-0'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.num_cycles)
    elif args.mode == 'test':
        test(args.model_name)
    else:
        parser.print_help()
