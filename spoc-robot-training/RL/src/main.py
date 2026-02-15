from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from random import random
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

try:
    from arguments import get_args
    from q_values_plot import plot_q_gaps, plot_q_values

    from networks import QNetwork
    from dqn_utils import (
        ReplayMemory,
        EpsilonGreedySchedule,
        select_action_time_aware,
        optimize_model,
        hard_update_target_net,
    )
    from funnel_reward import FunnelReward, FunnelSpec, circle_reach_robustness, funnel_gamma

except ImportError:
    from RL.src.arguments import get_args
    from RL.src.q_values_plot import plot_q_gaps, plot_q_values

    from RL.src.networks import QNetwork
    from RL.src.dqn_utils import (
        ReplayMemory,
        EpsilonGreedySchedule,
        select_action_time_aware,
        optimize_model,
        hard_update_target_net,
    )
    from RL.src.funnel_reward import FunnelReward, FunnelSpec, circle_reach_robustness, funnel_gamma

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")])
    return file_path


@dataclass
class TrainConfig:
    # core training
    num_episodes: int = 500
    batch_size: int = 128
    gamma: float = 1.0
    lr: float = 3e-4

    # exploration (PyTorch tutorial style)
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 5000.0  # larger -> slower decay

    # replay + target net
    replay_capacity: int = 50_000
    target_update_steps: int = 1000  # hard update every N gradient steps

    # episode control
    max_steps_per_episode: int = 200

    # logging
    log_every: int = 10

    randomize_init_time: bool = False # whether to randomize initial time index t in each episode 
    max_start_time: int = 20

def normalize_state(env, s, device: torch.device) -> torch.Tensor:
    """
    s: raw state from env = [x, y, theta_deg] for discrete unicycle
    Returns: feature tensor shape (1, 5): [x_norm, y_norm, sinθ, cosθ, t_norm]
    """
    x, y, theta_deg = float(s[0]), float(s[1]), float(s[2])

    # Position normalization to [0,1]
    x_norm = (x - env.x_min) / max(env.x_max - env.x_min, 1e-6)
    y_norm = (y - env.y_min) / max(env.y_max - env.y_min, 1e-6)

    # Heading encoding
    th = np.deg2rad(theta_deg)
    sin_th = np.sin(th)
    cos_th = np.cos(th)


    feat = torch.tensor([x_norm, y_norm, sin_th, cos_th],
                        dtype=torch.float32, device=device).unsqueeze(0)
    return feat


def train_DQN(
    env,
    obs_dim: int,
    n_actions: int,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Returns:
      history dict with episode returns and losses.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training DQN on device: {device}")

    #create the model folder to save the checkpoint
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d %H:%M:%S")


    # Networks take (state || time_feature) so input_dim = obs_dim + 1
    horizon = cfg.max_steps_per_episode
    policy_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
    target_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
    hard_update_target_net(target_net, policy_net)
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    memory = ReplayMemory(cfg.replay_capacity)
    eps_sched = EpsilonGreedySchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

    action_map = getattr(env, "action_space", None)
    if isinstance(action_map, list) and isinstance(action_map[0], str):
        idx_to_action = action_map
    else:
        idx_to_action = None  # assume env expects int action index

    history = {
        "episode_return": [],
        "episode_length": [],
        "loss": [],
    }

    global_grad_steps = 0

    print("Starting training...")

    for ep in range(cfg.num_episodes):
        print("Episode:", ep+1)
        
        s = env.reset()
        if isinstance(s, tuple):
            # in case env.reset() returns (state, info)
            s = s[0]

        #fixed length version:
        H = cfg.max_steps_per_episode
        fixed_length = False

        if cfg.randomize_init_time:
            t = np.random.randint(0, cfg.max_start_time + 1)
        else:
            t = 0

        env.episode_timer = t
        H_rem = H - t

        ep_return = 0.0


        for step in range(H_rem):
            
            s_norm = normalize_state(env, s, device=device)  # (1,5)
            t_norm = float(t) / float(max(horizon-1, 1))
            #print(t_norm)
            #state_t = torch.tensor(s, dtype=torch.float32, device=device)
            #print("Normalized State:", s_norm)

            # epsilon-greedy action using Q(s,a,t)
            a_idx = select_action_time_aware(
                policy_net=policy_net,
                state=s_norm,
                t=t_norm,
                n_actions=n_actions,
                eps_sched=eps_sched,
                device=device,
            )
            a_int = int(a_idx.item())
            a_env = idx_to_action[a_int] if idx_to_action is not None else a_int #convert to env action so the simulator can understand

            s_next, r, done = env.step(a_env)

            # store transition (Alg. 2 stores t)
            r_t = torch.tensor([float(r)], dtype=torch.float32, device=device)
            a_t = a_idx.to(device)  # (1,1) long
            s_t = s_norm.detach().unsqueeze(0) if s_norm.dim() == 1 else s_norm.detach()

            truncated = (step == H_rem - 1) and (not done)
            done_or_truncated = done or truncated

            if done_or_truncated:
                ns_t = None
            else:
                ns_t = normalize_state(env, s_next, device=device) #normalize next state
                ns_t = ns_t.detach().unsqueeze(0) if ns_t.dim() == 1 else ns_t.detach()

            memory.push(
                state=s_t,
                action=a_t,
                reward=r_t,
                next_state=ns_t,
                t=t_norm,
                done=bool(done_or_truncated),
                )

            ep_return += float(r)
            s = s_next
            t += 1

            # one optimization step (PyTorch tutorial style)
            loss_val = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                optimizer=optimizer,
                batch_size=cfg.batch_size,
                gamma=cfg.gamma,
                horizon=horizon,
                device=device,
            )
            if loss_val is not None:
                history["loss"].append(loss_val)
                global_grad_steps += 1

                # hard update target net periodically
                if global_grad_steps % cfg.target_update_steps == 0:
                    hard_update_target_net(target_net, policy_net)

            if done_or_truncated:
                history["episode_length"].append(step + 1)
                break

        history["episode_return"].append(ep_return)

        # #Fixed length version: if we exit the loop without break, it means we reached max steps
        # fixed_length = True
        # for step in range(H_rem):
        #     s_norm = normalize_state(env, s, device=device)         # (1, feat_dim)
        #     t_norm = float(t) / float(max(H - 1, 1))                # in [0,1]

        #     a_idx = select_action_time_aware(
        #         policy_net=policy_net,
        #         state=s_norm,
        #         t=t_norm,
        #         n_actions=n_actions,
        #         eps_sched=eps_sched,
        #         device=device,
        #     )
        #     a_int = int(a_idx.item())
        #     a_env = idx_to_action[a_int] if idx_to_action is not None else a_int

        #     # --- step env ---
        #     s_next, r, done = env.step(a_env)

        #     # we IGNORE done for episode termination; only use it for logging if you want
        #     #reached = reached or bool(done)

        #     # fixed-length termination: ONLY last step is terminal
        #     is_last = (step == H_rem - 1) 
        #     done_fixed = is_last

        #     # next_state is None only at last step
        #     if done_fixed:
        #         ns_t = None
        #     else:
        #         ns_t = normalize_state(env, s_next, device=device)  # (1, feat_dim)
        #         ns_t = ns_t.detach()

        #     # store transition
        #     r_t = torch.tensor([float(r)], dtype=torch.float32, device=device)
        #     a_t = a_idx.to(device)                                  # (1,1)
        #     s_t = s_norm.detach()                                   # already (1, feat_dim)

        #     memory.push(
        #         state=s_t,
        #         action=a_t,
        #         reward=r_t,
        #         next_state=ns_t,
        #         t=t_norm,
        #         done=bool(done_fixed),
        #     )

        #     ep_return += float(r)
        #     s = s_next
        #     t += 1

        #     loss_val = optimize_model(
        #         policy_net=policy_net,
        #         target_net=target_net,
        #         memory=memory,
        #         optimizer=optimizer,
        #         batch_size=cfg.batch_size,
        #         gamma=cfg.gamma,
        #         horizon=H,
        #         device=device,
        #     )
        #     if loss_val is not None:
        #         history["loss"].append(loss_val)
        #         global_grad_steps += 1
        #         if global_grad_steps % cfg.target_update_steps == 0:
        #             hard_update_target_net(target_net, policy_net)

        # history["episode_return"].append(ep_return)
        # history["episode_length"].append(H)

        if (ep + 1) % cfg.log_every == 0:
            avg_ret = float(np.mean(history["episode_return"][-cfg.log_every:]))
            avg_len = float(np.mean(history["episode_length"][-cfg.log_every:])) if history["episode_length"] else 0.0
            last_loss = history["loss"][-1] if history["loss"] else None
            print(
                f"[ep {ep+1:4d}/{cfg.num_episodes}] "
                f"avg_return={avg_ret:.3f} avg_len={avg_len:.1f} "
                f"eps={eps_sched.value():.3f} last_loss={last_loss}"
            )

    print("Training complete.")

    if fixed_length:
        folder_name = folder_name + '(fixed_length)'  #append to folder name if fixed length

    ckpt = {
        "policy_state_dict": policy_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "config": vars(cfg),
    }

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, "dqn_ckpt.pt")   # <-- make it a file
    torch.save(ckpt, ckpt_path)
    print(f"Saved DQN checkpoint!")

    #Plot and save the reward and loss curves in the same folder as the model
    plot_curves(history, folder_name)

    return history




def test_DQN(
    env,
    model_path: str,
    device: Optional[torch.device] = None,
):
    """
    Test a trained DQN model.

    Args:
      env: The environment to test on.
      model_path: Path to the saved model checkpoint.
      device: Device to run the model on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Testing DQN on device: {device}")

    ckpt = torch.load(model_path, map_location=device)
    #obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    
    feat_dim = 4

    policy_net = QNetwork(input_dim=feat_dim + 1, n_actions=n_actions).to(device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    policy_net.eval()

    num_of_test_episodes = 1
    max_steps_per_episode = cfg.max_steps_per_episode

    idx_to_action = getattr(env, "action_space", None)
    horizon = cfg.max_steps_per_episode

    env.reward_fn.spec.collision_penalty = 0.0  # Set collision penalty to 0 during testing to calculatye funnel function

    num_successful_episodes = 0

    #print(env.reward_fn.rho_max)  

    
    for episode in range(num_of_test_episodes):
        accumulated_reward = 0
        s = env.reset()
        satisfied = False
        #initial_t =  np.random.randint(0, cfg.max_start_time + 1)
        #print(f"Initial time index t: {initial_t}")
        initial_t = 20
        t = initial_t
        robustness_values = []
        funnel_values = []
        q_values_over_time = []
        while t < max_steps_per_episode:
            #print('Time step:', t)
            state_t = torch.tensor(s, dtype=torch.float32, device=device)
            state_norm = normalize_state(env, s, device = device)
            t_norm = float(t) / float(max(horizon-1, 1))
            t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)  # (1,1)
            with torch.no_grad():
                x = torch.cat([state_norm, t_tensor], dim=1)          # (1,5)
                q_values = policy_net(x)  
                #print(q_values.squeeze().cpu().numpy())  # shape (n_actions,)
                a_idx = q_values.argmax(dim=1).item()
                #print("Q-values:", q_values.squeeze().cpu().numpy())
            print("Q-values:", q_values)
            sorted_q, indices = torch.sort(q_values, dim=1)

            print("sorted Q-values:", sorted_q)
            print("Action indices sorted by Q-value:", indices)
            action = idx_to_action[a_idx]
            print(action)
            #print("State:", s)          
            #action = 'stay'
            s_next, reward, done = env.step(action)
            q_values_over_time.append(q_values.squeeze().cpu().numpy())
            accumulated_reward += reward
            s = s_next
            t += 1

            if done:
                satisfied = True
                break
                
                
            robustness = circle_reach_robustness((s[0], s[1]), env.goals[0]['center'], env.goals[0]['radius'])
            robustness_values.append(robustness)
            # funnel = robustness - r
            # funnel_values.append(funnel)
            
        if satisfied:
            num_successful_episodes += 1

        print(f"Test Episode {episode+1} finished in {t} steps.")
        print(f"Accumulated reward: {accumulated_reward:.3f}")

        gammas = np.array([funnel_gamma(int(t), env.reward_fn.gamma0, env.reward_fn.spec.gamma_inf, env.reward_fn.l) for t in range(initial_t, initial_t + len(robustness_values))])
        funnel_values = env.reward_fn.rho_max - gammas

        time_axis = np.arange(initial_t, initial_t + len(robustness_values))

        #plot robustness and funnel values on the same figure
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, robustness_values, label='Robustness', color='blue')
        plt.plot(time_axis, funnel_values, label='Funnel', color='orange')
        plt.hlines(y = env.reward_fn.rho_max, xmin=0, xmax=len(robustness_values)-1, colors='red', linestyles='dashed', label='Rho Max')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Robustness and Funnel Values Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        #Plot Q values:
        plot_q_values(q_values_over_time)

        ## Plot Q-value gaps over time
        #plot_q_gaps(q_values_over_time)

    print(f"Number of successful episodes: {num_successful_episodes} out of {num_of_test_episodes}")

def plot_curves(history, folder_name: str):
    rewards = history["episode_return"]
    loss = history["loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    #save the figure
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    plot_dir = os.path.join(base_dir, folder_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "dqn_reward_curve.png"), dpi = 200)
    plt.show()
    print("Reward figure saved!")

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    #save the figure
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    plot_dir = os.path.join(base_dir, folder_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "dqn_loss_curve.png"), dpi = 200)
    plt.show()
    print("Loss figure saved!")



if __name__ == "__main__":
    from simulator import Continuous2DEnv

    goals = {0: {'center': (5.2, 1.8), 'radius': 0.4, 'movement':{'type':'static'}}}

    targets = {}
    #config dictionary for the environment
    config = {
        'house_index': 30,
        'init_loc':[4.0657877922058105, 4.711010932922363, 41.999996185302734], #initial location of the agent (x, y)
        "dt": 1,
        "render": False,
		'dt_render': 0.01,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [300.0, 300.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "discrete unicycle", #dynamics model to use
		"targets": targets,
		"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
		"agent_as_point": False
    }

    args = get_args()
    
    num_episodes = 1200
    max_steps_per_episode = 60

    cfg = TrainConfig(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=128,
        eps_decay= 0.3 * (num_episodes * max_steps_per_episode),  # decay over the whole training
        randomize_init_time=True,
        max_start_time = 30
    )

    if args.mode == 'train':
        env = Continuous2DEnv(config)
        feat_dim = 4  # [x_norm, y_norm, sinθ, cosθ]
        #obs_dim = int(env.observation_space.shape[0])  # (3,) for discrete unicycle
        n_actions = len(env.action_space) if isinstance(env.action_space, list) else int(env.action_space.shape[0])

        history = train_DQN(env, feat_dim, n_actions, cfg) #train the DQN
    
    elif args.mode == 'test':
        model_path = select_model_file()
        if model_path:
            config['render'] = True # Enable rendering for testing
            config['dt_render'] = 0.03
            config["agent_as_point"] = True
            env = Continuous2DEnv(config)
            test_DQN(env, model_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            print("No model file selected. Exiting.")
