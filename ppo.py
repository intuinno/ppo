import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import gym


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n), std=0.01)
        )
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x) 


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)),default=True, nargs='?',
                        const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', 
                        const=True, help='if toggled, this experiment will be tracked with Weights and biases')
    parser.add_argument('--wandb-project-name', type=str, default="PPO", 
                        help='the wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, 
                        help="the entity (team) of wandb's project")
    parser.add_argument('--num-envs', type=int, default=4, 
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, "videos")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True, 
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key,value in vars(args).items()])),
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed+i, i, True, run_name) 
                                     for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print("envs.single_observation_space.shape: ", envs.single_observation_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    rollout_shape = (args.num_steps, args.num_envs)
    obs = torch.zeros(rollout_shape + envs.single_observation_space.shape).to(device)
    actions = torch.zeros(rollout_shape + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(rollout_shape).to(device)
    rewards = torch.zeros(rollout_shape).to(device)
    dones = torch.zeros(rollout_shape).to(device)
    values = torch.zeros(rollout_shape).to(device)

    # start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size 
    print(num_updates)
    print("next_obs.shape: ", next_obs.shape)
    print("agent.get_value(next_obs):", agent.get_value(next_obs))
    print("agent.get_value(next_obs).shape:", agent.get_value(next_obs).shape)
    print("agent.get_action_and_value(next_obs):", agent.get_action_and_value(next_obs))

    for update in range(1, num_updates+1):
        # anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) /num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # roll out
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob 

            # execute action
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                    writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                    break




    
