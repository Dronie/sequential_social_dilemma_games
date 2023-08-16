import argparse

import gym
import supersuit as ss #check Documents/SuperSuit-3.3.1 for source
import torch
import torch.nn.functional as F
# pip install git+https://github.com/Rohan138/marl-baselines3
from marl_baselines3 import IndependentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch import nn
import wandb
from wandb.integration.sb3 import WandbCallback

from social_dilemmas.envs.pettingzoo_env import parallel_env

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser("MARL-Baselines3 PPO with Independent Learning")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--play-altruistic-game",
        type=bool,
        default=True,
        help="Frame environment as an altruistic game",
    )
    parser.add_argument(
        "--altruistic-model",
        type=str,
        default="A",
        choices=["A", "B", "C", "D"],
        help="The model of atruism to use",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument(
        "--alt-alpha",
        type=float,
        default=1,
        help="Social welfare coefficient for altruistic game",
    )
    args = parser.parse_args()
    return args


wandb.init(project='independent-ppo')
wandb_config = wandb.config

# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main(args):
    # Config
    wandb_config.env_name = env_name = args.env_name
    wandb_config.num_agents = num_agents = args.num_agents
    wandb_config.rollout_len = rollout_len = args.rollout_len
    wandb_config.total_timesteps = total_timesteps = args.total_timesteps
    wandb_config.use_collective_reward = use_collective_reward = args.use_collective_reward
    wandb_config.inequity_averse_reward = inequity_averse_reward = args.inequity_averse_reward
    wandb_config.play_altruistic_game = play_altruistic_game = args.play_altruistic_game
    wandb_config.altruistic_model = altruistic_model = args.altruistic_model
    wandb_config.alpha = alpha = args.alpha
    wandb_config.beta = beta = args.beta
    wandb_config.alt_alpha = alt_alpha = args.alt_alpha

    # Training
    wandb_config.num_cpus = num_cpus = 4  # number of cpus
    wandb_config.num_envs = num_envs = 12  # number of parallel multi-agent environments
    wandb_config.num_frames = num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    wandb_config.ent_coef = ent_coef = 0.001  # entropy coefficient in loss
    wandb_config.batch_size = batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    wandb_config.lr = lr = 0.0001
    wandb_config.n_epochs = n_epochs = 30
    wandb_config.gae_lambda = gae_lambda = 1.0
    wandb_config.gamma = gamma = 0.99
    wandb_config.target_kl = target_kl = 0.01
    wandb_config.grad_clip = grad_clip = 40
    verbose = 3

    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        use_collective_reward=use_collective_reward,
        inequity_averse_reward=inequity_averse_reward,
        play_altruistic_game=play_altruistic_game,
        altruistic_model=altruistic_model,
        alpha=alpha,
        beta=beta,
        alt_alpha=alt_alpha,
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    # turns input env into a MarkovVectorEnv check Documents/SuperSuit-3.3.1 for source
    env = ss.pettingzoo_env_to_vec_env_v1(env) 
    # puts input env through MakeCPUAsyncConstructor (which turns env into ProcConcatVec) wrapped by SB3VecEnvWrapper
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    # VecMonitor is a wrapper for vectorized Gym environments
    # It monitors the episode reward, length time and other data
    env = VecMonitor(env) 
    

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = "./results/sb3/cleanup_ppo_independent"

    model = IndependentPPO(
        "CnnPolicy",
        num_agents=num_agents,
        env=env,
        learning_rate=lr,
        n_steps=rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, callback=WandbCallback())

    logdir = model.logger.dir
    model.save(logdir)
    del model
    model = IndependentPPO.load(  # noqa: F841
        logdir, "CnnPolicy", num_agents, env, rollout_len, policy_kwargs, tensorboard_log, verbose
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
