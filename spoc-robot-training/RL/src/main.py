from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from random import random
import random
from typing import Dict, Optional, Tuple

import numpy as np
import stlrom
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
    from robustness_calculator import calculate_predicate_robustness

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
    from RL.src.robustness_calculator import calculate_predicate_robustness

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
    gamma: float = 0.95
    lr: float = 3e-4

    # exploration (PyTorch tutorial style)
    eps_start: float = 0.9
    eps_end: float = 0.1
    eps_decay: float = 6000.0  # larger -> slower decay

    # replay + target net
    replay_capacity: int = 50_000
    target_update_steps: int = 500  # hard update every N gradient steps

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


# def train_DQN(
#     env,
#     obs_dim: int,
#     n_actions: int,
#     cfg: TrainConfig,
#     device: Optional[torch.device] = None,
# ) -> Dict[str, list]:
#     """
#     Returns:
#       history dict with episode returns and losses.
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"Training DQN on device: {device}")

#     #create the model folder to save the checkpoint
#     now = datetime.now()
#     folder_name = now.strftime("%Y-%m-%d %H:%M:%S")


#     # Networks take (state || time_feature) so input_dim = obs_dim + 1
#     horizon = cfg.max_steps_per_episode
#     policy_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
#     target_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
#     hard_update_target_net(target_net, policy_net)
#     target_net.eval()

#     optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
#     memory = ReplayMemory(cfg.replay_capacity)
#     eps_sched = EpsilonGreedySchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

#     action_map = getattr(env, "action_space", None)
#     if isinstance(action_map, list) and isinstance(action_map[0], str):
#         idx_to_action = action_map
#     else:
#         idx_to_action = None  # assume env expects int action index

#     history = {
#         "episode_return": [],
#         "episode_length": [],
#         "loss": [],
#     }

#     global_grad_steps = 0

#     print("Starting training...")

#     for ep in range(cfg.num_episodes):
#         print("Episode:", ep+1)
        
#         s = env.reset()
#         if isinstance(s, tuple):
#             # in case env.reset() returns (state, info)
#             s = s[0]

#         #fixed length version:
#         H = cfg.max_steps_per_episode
#         fixed_length = True

#         if cfg.randomize_init_time:
#             t = np.random.randint(0, cfg.max_start_time + 1)
#         else:
#             t = 0

#         env.episode_timer = t
#         H_rem = H - t

#         ep_return = 0.0

#         #Fixed length version: if we exit the loop without break, it means we reached max steps
#         for step in range(H_rem):
#             s_norm = normalize_state(env, s, device=device)         # (1, feat_dim)
#             t_norm = float(t) / float(max(H - 1, 1))                # in [0,1]

#             a_idx = select_action_time_aware(
#                 policy_net=policy_net,
#                 state=s_norm,
#                 t=t_norm,
#                 n_actions=n_actions,
#                 eps_sched=eps_sched,
#                 device=device,
#             )

#             a_int = int(a_idx.item())
#             a_env = idx_to_action[a_int] if idx_to_action is not None else a_int

#             # --- step env ---
#             s_next, r, done = env.step(a_env)

#             # we IGNORE done for episode termination; only use it for logging if you want
#             #reached = reached or bool(done)

#             # fixed-length termination: ONLY last step is terminal
#             is_last = (step == H_rem - 1) 
#             done_fixed = is_last

#             # next_state is None only at last step
#             if done_fixed:
#                 ns_t = None
#             else:
#                 ns_t = normalize_state(env, s_next, device=device)  # (1, feat_dim)
#                 ns_t = ns_t.detach()

#             # store transition
#             r_t = torch.tensor([float(r)], dtype=torch.float32, device=device)
#             a_t = a_idx.to(device)                                  # (1,1)
#             s_t = s_norm.detach()                                   # already (1, feat_dim)

#             memory.push(
#                 state=s_t,
#                 action=a_t,
#                 reward=r_t,
#                 next_state=ns_t,
#                 t=t_norm,
#                 done=bool(done_fixed),
#             )

#             ep_return += float(r)
#             s = s_next
#             t += 1

#             loss_val = optimize_model(
#                 policy_net=policy_net,
#                 target_net=target_net,
#                 memory=memory,
#                 optimizer=optimizer,
#                 batch_size=cfg.batch_size,
#                 gamma=cfg.gamma,
#                 horizon=H,
#                 device=device,
#             )
#             if loss_val is not None:
#                 history["loss"].append(loss_val)
#                 global_grad_steps += 1
#                 if global_grad_steps % cfg.target_update_steps == 0:
#                     hard_update_target_net(target_net, policy_net)

#         history["episode_return"].append(ep_return)
#         history["episode_length"].append(H)

#         if (ep + 1) % cfg.log_every == 0:
#             avg_ret = float(np.mean(history["episode_return"][-cfg.log_every:]))
#             avg_len = float(np.mean(history["episode_length"][-cfg.log_every:])) if history["episode_length"] else 0.0
#             last_loss = history["loss"][-1] if history["loss"] else None
#             print(
#                 f"[ep {ep+1:4d}/{cfg.num_episodes}] "
#                 f"avg_return={avg_ret:.3f} avg_len={avg_len:.1f} "
#                 f"eps={eps_sched.value():.3f} last_loss={last_loss}"
#             )

#     print("Training complete.")

#     if fixed_length:
#         folder_name = folder_name + '(fixed_length)'  #append to folder name if fixed length

#     ckpt = {
#         "policy_state_dict": policy_net.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "obs_dim": obs_dim,
#         "n_actions": n_actions,
#         "config": vars(cfg),
#     }

#     save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", folder_name)
#     os.makedirs(save_dir, exist_ok=True)

#     ckpt_path = os.path.join(save_dir, "dqn_ckpt.pt")   # <-- make it a file
#     torch.save(ckpt, ckpt_path)
#     print(f"Saved DQN checkpoint!")

#     #Plot and save the reward and loss curves in the same folder as the model
#     plot_curves(history, folder_name)

#     return history



def train_DQN(
    env,
    obs_dim: int,
    n_actions: int,
    cfg,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Train a DQN model and periodically evaluate to save the best checkpoint.

    Expected cfg additions (if missing, defaults are used):
      - eval_every: int (episodes)              default: 10
      - eval_episodes: int                      default: 5
      - save_latest_on_eval: bool               default: True
      - metric_mode: str in {"max"}             default: "max" (maximize return)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training DQN on device: {device}")

    # --- Eval/save config defaults (so you don't have to edit TrainConfig immediately) ---
    eval_every = getattr(cfg, "eval_every", 10)
    eval_episodes = getattr(cfg, "eval_episodes", 1)
    save_latest_on_eval = getattr(cfg, "save_latest_on_eval", True)

    # create the model folder to save the checkpoint
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d %H:%M:%S")

    # Networks take (state || time_feature) so input_dim = obs_dim + 1
    H = cfg.max_steps_per_episode
    policy_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
    target_net = QNetwork(input_dim=obs_dim + 1, n_actions=n_actions).to(device)
    hard_update_target_net(target_net, policy_net)
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    memory = ReplayMemory(cfg.replay_capacity)
    eps_sched = EpsilonGreedySchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

    action_map = getattr(env, "action_space", None)
    if isinstance(action_map, list) and len(action_map) > 0 and isinstance(action_map[0], str):
        idx_to_action = action_map
    else:
        idx_to_action = None  # assume env expects int action index

    history = {
        "episode_return": [],
        "episode_length": [],
        "loss": [],
        "eval_return": [],
        "best_eval_return": [],
    }

    global_grad_steps = 0
    fixed_length = True  # you already use fixed-length termination

    # --- Save dir created once so we can save during training ---
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(save_dir, "dqn_ckpt_latest.pt")
    best_ckpt_path = os.path.join(save_dir, "dqn_ckpt_best.pt")

    best_eval = -float("inf")

    def build_ckpt():
        return {
            "policy_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "config": vars(cfg),
        }

    @torch.no_grad()
    def evaluate_greedy(num_episodes: int) -> float:
        """
        Greedy evaluation: always take argmax_a Q(s,t,a).
        Uses same fixed-length horizon H, sets t=0 (no random init time) for consistency.
        """
        policy_net.eval()
        returns = []

        for _ in range(num_episodes):
            s = env.reset()
            if isinstance(s, tuple):
                s = s[0]

            # eval uses deterministic start time
            t_eval = 0
            env.episode_timer = t_eval

            ep_ret = 0.0
            for step in range(H - t_eval):
                s_norm = normalize_state(env, s, device=device)  # (1, feat_dim)
                t_norm = float(t_eval) / float(max(H - 1, 1))

                # greedy action
                q = policy_net(torch.cat([s_norm, torch.tensor([[t_norm]], device=device)], dim=1))
                a_int = int(torch.argmax(q, dim=1).item())
                a_env = idx_to_action[a_int] if idx_to_action is not None else a_int

                s_next, r, done = env.step(a_env)
                ep_ret += float(r)

                s = s_next
                t_eval += 1

            returns.append(ep_ret)

        policy_net.train()
        return float(np.mean(returns)) if returns else -float("inf")

    print("Starting training...")

    for ep in range(cfg.num_episodes):
        print("Episode:", ep + 1)

        s = env.reset()
        if isinstance(s, tuple):
            s = s[0]

        # fixed length version:
        if cfg.randomize_init_time:
            t = np.random.randint(0, cfg.max_start_time + 1)
        else:
            t = 0

        env.episode_timer = t
        H_rem = H - t

        ep_return = 0.0

        for step in range(H_rem):
            s_norm = normalize_state(env, s, device=device)  # (1, feat_dim)
            t_norm = float(t) / float(max(H - 1, 1))         # in [0,1]

            a_idx = select_action_time_aware(
                policy_net=policy_net,
                state=s_norm,
                t=t_norm,
                n_actions=n_actions,
                eps_sched=eps_sched,
                device=device,
            )

            a_int = int(a_idx.item())
            a_env = idx_to_action[a_int] if idx_to_action is not None else a_int

            # --- step env ---
            s_next, r, done = env.step(a_env)

            # fixed-length termination: ONLY last step is terminal
            done_fixed = (step == H_rem - 1)

            # next_state is None only at last step
            if done_fixed:
                ns_t = None
            else:
                ns_t = normalize_state(env, s_next, device=device).detach()

            # store transition
            r_t = torch.tensor([float(r)], dtype=torch.float32, device=device)
            a_t = a_idx.to(device)
            s_t = s_norm.detach()

            memory.push(
                state=s_t,
                action=a_t,
                reward=r_t,
                next_state=ns_t,
                t=t_norm,
                done=bool(done_fixed),
            )

            ep_return += float(r)
            s = s_next
            t += 1

            loss_val = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                optimizer=optimizer,
                batch_size=cfg.batch_size,
                gamma=cfg.gamma,
                horizon=H,
                device=device,
            )
            if loss_val is not None:
                history["loss"].append(loss_val)
                global_grad_steps += 1
                if global_grad_steps % cfg.target_update_steps == 0:
                    hard_update_target_net(target_net, policy_net)

        history["episode_return"].append(ep_return)
        history["episode_length"].append(H)

        # ---- periodic eval + save ----
        do_eval = (eval_every > 0) and ((ep + 1) % eval_every == 0)
        if do_eval:
            eval_avg = evaluate_greedy(eval_episodes)
            history["eval_return"].append(eval_avg)

            # always save latest on eval (optional)
            if save_latest_on_eval:
                torch.save(build_ckpt(), latest_ckpt_path)
                print(f"[eval @ ep {ep+1}] saved latest -> {latest_ckpt_path}")

            # save best model
            if eval_avg > best_eval:
                best_eval = eval_avg
                torch.save(build_ckpt(), best_ckpt_path)
                print(f"[eval @ ep {ep+1}] NEW BEST eval_return={eval_avg:.3f} -> {best_ckpt_path}")

            history["best_eval_return"].append(best_eval)
            print(f"[eval @ ep {ep+1}] eval_return={eval_avg:.3f} best={best_eval:.3f}")

        # your existing logging
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
        folder_name = folder_name + "(fixed_length)"

    # final save (keep your original name too)
    ckpt_path = os.path.join(save_dir, "dqn_ckpt.pt")
    torch.save(build_ckpt(), ckpt_path)
    print(f"Saved DQN checkpoint! -> {ckpt_path}")

    # Plot and save the reward and loss curves
    plot_curves(history, folder_name)

    return history


def test_DQN(env, model_path: str, horizon: int, goals: dict, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Testing DQN on device: {device}")

    ckpt = torch.load(model_path, map_location=device)
    n_actions = ckpt["n_actions"]

    feat_dim = 4
    policy_net = QNetwork(input_dim=feat_dim + 1, n_actions=n_actions).to(device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    policy_net.eval()

    idx_to_action = getattr(env, "action_space", None)

    num_of_test_episodes = 100
    num_successful_episodes = 0

    for episode in range(num_of_test_episodes):
        s = env.reset()
        accumulated_reward = 0.0
        q_values_over_time = []

        # Store trajectory positions so we can compute robustness curves after the rollout
        traj_xy = []   # list of (x,y) AFTER each step
        traj_t  = []   # list of t indices corresponding to traj_xy

        satisfied = False
        init_t = 0
        t = init_t

        while t < horizon:
            state_norm = normalize_state(env, s, device=device)
            t_norm = float(t) / float(max(horizon - 1, 1))
            t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)

            with torch.no_grad():
                x = torch.cat([state_norm, t_tensor], dim=1)
                q_values = policy_net(x)
                a_idx = int(q_values.argmax(dim=1).item())

            a_env = idx_to_action[a_idx] if idx_to_action is not None else a_idx
            #a_env = 'stay'
            #print("action:", a_env)

            s_next, reward, done = env.step(a_env)
            #print('state:', s_next)

            q_values_over_time.append(q_values.squeeze().cpu().numpy())
            accumulated_reward += float(reward)

            # log (x,y) at the *new* state
            traj_xy.append((float(s_next[0]), float(s_next[1])))
            traj_t.append(t)

            s = s_next
            t += 1

        predicate_signals = np.zeros((len(traj_xy), len(goals)))  # shape (T, num_goals)

        for goal_id in goals.keys():
            g = goals[goal_id]
            predicate_robustness = calculate_predicate_robustness(traj_xy, g['center'], g['radius'])
            predicate_signals[:, goal_id] = predicate_robustness

        task_robustness = get_task_robustness(stl_task_str, predicate_signals)
        print(f"Task robustness: {task_robustness:.3f}")

        if task_robustness >= 0:
            satisfied = True

        if satisfied:
            num_successful_episodes += 1

        print(f"Test Episode {episode+1} finished in {t} steps.")
        print(f"Accumulated reward: {accumulated_reward:.3f}")

        # -------------------------
        # Multi-task funnel plotting
        # -------------------------
        tasks = env.reward_fn.tasks  # list[TaskDef]
        T = len(traj_t)
        time_axis = np.array(traj_t, dtype=int)

        # plt.figure(figsize=(12, 6))

        # for k, td in enumerate(tasks):
        #     fr = td.reward_fn  # FunnelReward

        #     # Active interval for plotting:
        #     # use td.start (your "effective start") and td.b (end)
        #     t0 = int(td.start) if td.start is not None else int(td.a)
        #     t1 = int(td.b)

        #     # Prepare arrays filled with NaN so lines only appear in their active window
        #     rob_k = np.full(T, np.nan, dtype=float)
        #     fun_k = np.full(T, np.nan, dtype=float)

        #     # Goal for this task
        #     g = env.goals[td.goal_id]
        #     gc = g["center"]
        #     gr = g["radius"]

        #     # Fill only where time is within [t0, t1]
        #     for i, (ti, (x, y)) in enumerate(zip(time_axis, traj_xy)):
        #         if ti < t0 or ti > t1:
        #             continue

        #         # robustness wrt this task goal
        #         rob = circle_reach_robustness((x, y), gc, gr)
        #         rob_k[i] = rob

        #         # funnel lower bound for this task at global time ti
        #         # gamma uses fr.time_origin internally (you set it properly)
        #         t_local = max(int(ti) - int(fr.time_origin), 0)
        #         gam = funnel_gamma(t_local, fr.gamma0, fr.spec.gamma_inf, fr.l)
        #         fun_k[i] = fr.rho_max - gam

        #     plt.plot(time_axis, rob_k, label=f"Task {k}: robustness (goal {td.goal_id})")
        #     plt.plot(time_axis, fun_k,  linestyle="--", label=f"Task {k}: funnel bound")

        #     # mark the active window edges
        #     plt.axvline(t0, linestyle=":", linewidth=1)
        #     plt.axvline(t1, linestyle=":", linewidth=1)

        # plt.xlabel("Time step")
        # plt.ylabel("Value")
        # plt.title("Robustness vs Funnel Bounds (per task, plotted only in each task's time window)")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # Plot Q-values if you want
        #plot_q_values(q_values_over_time)

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



def get_task_robustness(task, signals):
    stl_driver =stlrom.STLDriver()
    stl_driver.parse_string(task)

    #add the samples:
    for i in range(signals.shape[0]):
        sample = [i] + signals[i].tolist()
        stl_driver.add_sample(sample)

    phi = stl_driver.get_monitor("phi") #overall task
    robustness = phi.eval_rob()

    return robustness    

if __name__ == "__main__":
    from simulator import Continuous2DEnv

    ######################################
    #HOUSE 30 GOALS AND TASKS FOR TESTING:
    ######################################
    #goals_dict = {0: {'center': (5.2, 1.8), 'radius': 0.4, 'movement':{'type':'static'}}}

    # goals_dict = {0: {'center': (5.2, 1.8), 'radius': 0.4, 'movement':{'type':'static'}}, #goal region for the agentssss
	# 1: {'center': (1.8, 4.5), 'radius': 0.4, 'movement':{'type':'static'}}
    # }

    # STL_tasks =  [{"goal_id": 0, "spec": dict(operator="F", a=0,  b=45, t_star=30, gamma_inf=-0.1, collision_penalty=0.0)},
    # {"goal_id": 1, "spec": dict(operator="F", a=46, b=85, t_star=70, gamma_inf=-0.1, collision_penalty=0.0)}]

    # #STL_tasks =  [{"goal_id": 0, "spec": dict(operator="F", a=0,  b=60, t_star=50, gamma_inf=-0.1, collision_penalty=0.0)}]

    # STL_horizon = 85 #CHANGE LATER!

    # stl_task_str = """
    # signal x, y   # signal namesss
    # mu_1 := x[t] > 0  # goal-1
    # mu_2 := y[t] > 0  # goal-2

    # phi1 := F_[0, 45] mu_1
    # phi2 := F_[46, 85] mu_2
    # phi := phi1 and phi2
    # """

    # targets = {}
    # #config dictionary for the environment
    # config = {
    #     'house_index': 30,
    #     'init_loc':[3, 0.5, 0.0], #initial location of the agent (x, y, theta_deg)
    #     "dt": 1,
    #     "render": False,
	# 	'dt_render': 0.01,
	# 	'goals': goals_dict, #goal regions for the agent
    #     "obstacle_location": [300.0, 300.0],
    #     "obstacle_size": 0.0,
    #     "randomize_loc": True, #whether to randomize the agent location at the end of each episode
	# 	'deterministic': False,
	# 	'auto_entropy':True,
	# 	"dynamics": "discrete unicycle", #dynamics model to use
	# 	"targets": targets,
	# 	"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
	# 	"agent_as_point": False,
    #     "tasks": STL_tasks,
    # }


    #####################################
    #HOUSE 9 GOALS AND TASKS FOR TESTING:
    #####################################

    goals_dict = {0: {'center': (5, 7.6), 'radius': 0.4, 'movement':{'type':'static'}}, #goal region for the agentssss
	1: {'center': (2.5, 1), 'radius': 0.4, 'movement':{'type':'static'}}}

    STL_tasks =  [{"goal_id": 0, "spec": dict(operator="F", a=0,  b=70, t_star=60, gamma_inf=-0.1, collision_penalty=1.0)},
    {"goal_id": 1, "spec": dict(operator="F", a=70, b=150, t_star=140, gamma_inf=-0.1, collision_penalty=1.0)},
    ]

    STL_horizon = 150 #CHANGE LATER!


    stl_task_str = """
    signal x, y   # signal namesss
    mu_1 := x[t] > 0  # goal-1
    mu_2 := y[t] > 0  # goal-2

    phi1 := F_[0, 70] mu_1
    phi2 := F_[70, 150] mu_2
    phi := phi1 and phi2
    """

    targets = {}
    #config dictionary for the environment
    config = {
        'house_index': 9,
        'init_loc':[1, 6.5, 90.0], #initial location of the agent (x, y)
        "dt": 1,
        "render": False,
		'dt_render': 0.01,
		'goals': goals_dict, #goal regions for the agent
        "obstacle_location": [300.0, 300.0],
        "obstacle_size": 0.0,
        "randomize_loc": True, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		"dynamics": "discrete unicycle", #dynamics model to use
		"targets": targets,
		"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
		"agent_as_point": False,
        "tasks": STL_tasks,
        "episode_len": STL_horizon,
    }



    args = get_args()
    
    num_episodes = 2000
    max_steps_per_episode = STL_horizon

    cfg = TrainConfig(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=128,
        eps_decay= 0.3 * (num_episodes * max_steps_per_episode),  # decay over the whole training
        randomize_init_time=False,
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
            config["agent_as_point"] = False
            config["randomize_loc"] = False
            env = Continuous2DEnv(config)
            test_DQN(env, model_path, horizon = STL_horizon, goals=goals_dict, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            #test_DQN(env, model_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            print("No model file selected. Exiting.")
