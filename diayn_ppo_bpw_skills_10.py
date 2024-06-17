import argparse
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from distutils.util import strtobool
from matplotlib import pyplot as plt

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="BipedalWalker-v3",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=15000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="diayn_ppo_base",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    
    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.18,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.1, # 0.0 for ppo
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # DIAYN
    parser.add_argument("--n_skills", default=10, type=int,
        help="The number of skills to learn.")
    # PCA
    parser.add_argument("--pca_num_steps", default=2500, type=int,
        help="the number of steps to run for pca chart")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(gym_id, seed, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(gym_id, render_mode="rgb_array")#, hardcore=True)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # for continuous actions        
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -400, 400)) # for bipedalWalker -- originally [-10, 10] for different env
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def record_skill_videos(z, args, agent, run_name):
    run_name = f"{run_name}/skill_{z}"
    env = make_env(args.gym_id, args.seed, args.capture_video, run_name)()
    obs, _ = env.reset()

    obs_dim = obs.shape[0]  # This extracts the observation dimensionality
    obs_array = np.zeros((args.pca_num_steps, obs_dim)) # for PCA
    obs_idx = 0
    obs_array[obs_idx] = obs
    done = False
    
    z_one_hot = np.zeros(args.n_skills)
    z_one_hot[z] = 1
    
    while not done:
        aug_obs = np.concatenate([obs, z_one_hot])
        aug_obs = torch.Tensor(np.array(aug_obs)).to(device)
        with torch.no_grad():
            action = agent.get_play_action(aug_obs)
        obs, _, done, _, _ = env.step(action.cpu().numpy()[0])
        
        obs_idx += 1
        if obs_idx < args.pca_num_steps:
            obs_array[obs_idx] = obs
    env.close()

    return obs_array

def plot_pca_over_skills(states, labels, args, pca_save_path, n_components=2):
    pca = PCA(n_components=n_components)
    plt.figure(figsize=(10, 7))

    all_labels = []
    for skill_idx in range(len(labels)):
        all_labels.append([labels[skill_idx] for i in range(args.pca_num_steps)])
    states = np.concatenate(states)
    all_labels = np.concatenate(all_labels)

    reduced_states = pca.fit_transform(states)
    scatter = plt.scatter(reduced_states[:, 0], reduced_states[:, 1], c=all_labels, cmap='viridis', marker='o', edgecolor='k', s=20, alpha=0.7)
    
    plt.title("PCA of DIAYN Latent States")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label='Skills')
    
    plt.savefig(pca_save_path)
    plt.close()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(((envs.single_observation_space.shape[0] + args.n_skills),)).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(((envs.single_observation_space.shape[0] + args.n_skills),)).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_play_action(self, x):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action
    
class Discriminator(nn.Module):
    def __init__(self, envs, num_skills):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, num_skills)
        )

    def forward(self, state):
        return self.network(state)
    
def update_discriminator(discriminator, discriminator_optimizer, states, skill, z, device):
    # Convert lists or arrays to tensors
    states = torch.Tensor(states).to(device)
    skill = torch.Tensor(skill).to(device) 

    # Reset gradients
    discriminator_optimizer.zero_grad()

    # Forward pass
    logits = discriminator(states)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = logits.size(0)  # This should be 16 in your case
    skill_indices = torch.tensor([z for b in range(batch_size)], dtype=torch.long, device=device)

    loss = loss_fn(logits, skill_indices)

    # Backward pass and optimize
    loss.backward()
    discriminator_optimizer.step()

    return -loss.item()

def intrinsic_reward(discriminator, state, skill, z, p_z, device):
    skill = skill * z
    logits = discriminator(state)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    p_z = p_z.view(1, -1) # Reshape to [1, n_skills] for broadcasting
     
    # Ensure skill_indices matches the batch size of log_probs
    batch_size = log_probs.size(0) 
    value = p_z[0, 0] # Extracting the value, e.g., 0.0500 -- fixed by DIAYN paper (as opposed to VIC)
    new_p_z = (torch.log(value + 1e-6) * torch.ones(batch_size, 1)).to(device)
    skill_indices = torch.tensor([z for b in range(batch_size)], dtype=torch.long, device=device).view(-1, 1)  # Adjust to correct batch size and shape

    rewards = log_probs - new_p_z
    selected_rewards = rewards.gather(1, skill_indices) # This selects the specific log_prob for each skill

    # Squeeze the result to remove the last dimension, resulting in a tensor of shape [batch_size]
    selected_rewards = selected_rewards.squeeze(1)
    
    return selected_rewards # returns log prob for skill used for each env

def sample_skill(num_skills, p_z):
    # Samples z using probabilities in p_z
    return np.random.choice(num_skills, p=p_z)

def concat_state_latent(obs, z, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z] = 1
    aug_obs = []
    for s in obs:
        aug_obs.append(np.concatenate([s, z_one_hot]))
    return aug_obs, z_one_hot

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{int(time.time())}"
    pca_save_path = f"pca/{run_name}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, False, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    discriminator = Discriminator(envs, args.n_skills).to(device)
    discriminator_opt = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    num_updates = args.total_timesteps // args.batch_size
    assert args.total_timesteps > args.batch_size

    # ALGO Logic: Storage setup
    discriminator_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs = torch.zeros((args.num_steps, args.num_envs) + (envs.single_observation_space.shape[0] + args.n_skills,)).to(device)# + (args.n_skills,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    
    p_z = torch.full((args.n_skills,), 1.0 / args.n_skills)
    z = sample_skill(args.n_skills, p_z)
    aug_obs, skill = concat_state_latent(next_obs, z, args.n_skills)
    aug_obs = torch.Tensor(np.array(aug_obs)).to(device)
    next_obs = torch.Tensor(next_obs).to(device)

    for update in range(1, num_updates + 1):
        print(f"---> Update (episode): {update}, z: {z}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = aug_obs
            discriminator_obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(aug_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, trunc, info = envs.step(action.cpu().numpy())

            aug_obs, skill = concat_state_latent(next_obs, z, args.n_skills)
            aug_obs = torch.Tensor(np.array(aug_obs)).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            int_reward = intrinsic_reward(discriminator, next_obs, skill, z, p_z, device).to(device).detach()
            rewards[step] = int_reward
            
            for item_idx in range(len(done)):
                episodic_return = sum(rewards[-step:])
                if done[item_idx]:
                    print(f"global_step={global_step}, episodic_return={episodic_return[item_idx]}")
                    writer.add_scalar("charts/episodic_return", episodic_return[item_idx], global_step)
                    writer.add_scalar("charts/episodic_length", step, global_step)
                    writer.add_scalar(f"skill_charts/episodic_return/skill-{z}", episodic_return[item_idx], global_step)
                    break
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(aug_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (envs.single_observation_space.shape[0] + args.n_skills,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        loss = update_discriminator(discriminator, discriminator_opt, discriminator_obs[update], skill, z, device)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/discriminator_loss_per_global", loss, global_step)
        writer.add_scalar("losses/discriminator_loss_per_update", loss, update)
        writer.add_scalar(f"skill_losses/discrminator_loss_per_update/skill-{z}", loss, update)

        z = sample_skill(args.n_skills, p_z)

    envs.close()
    writer.close()

    pca_states = []
    pca_labels = []
    for skill_idx in range(args.n_skills):
        print(f"Recording video for skill {skill_idx}")
        pca_states.append(record_skill_videos(skill_idx, args, agent, run_name))
        pca_labels.append(skill_idx)
    
    plot_pca_over_skills(pca_states, pca_labels, args, pca_save_path)