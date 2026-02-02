from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Replay Memory 
# ----------------------------
@dataclass
class Transition:
    state: torch.Tensor       # shape: (obs_dim,) or (1, obs_dim)
    action: torch.Tensor      # shape: (1, 1)  (discrete action index)
    reward: torch.Tensor      # shape: (1,)
    next_state: Optional[torch.Tensor]  # None if terminal
    t: float                   # time index at this transition
    done: bool


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Optional[torch.Tensor],
        t: float,
        done: bool,
    ) -> None:
        self.memory.append(Transition(state, action, reward, next_state, t, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

#Time feature utilities:

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    # Converts (obs_dim,) -> (1, obs_dim)
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _time_feature(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.float().unsqueeze(1)
    return t.float()

def concat_state_time(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    state = _ensure_2d(state)
    return torch.cat([state, _time_feature(t)], dim=1)


class EpsilonGreedySchedule:
    """
    Matches the PyTorch tutorial style:
      eps = eps_end + (eps_start - eps_end) * exp(-steps_done/eps_decay)
    """
    def __init__(self, eps_start: float, eps_end: float, eps_decay: float) -> None:
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.steps_done = 0

    def value(self) -> float:
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay)
        return eps

    def step(self) -> None:
        self.steps_done += 1


@torch.no_grad()
def select_action_time_aware(
    policy_net: nn.Module,
    state: torch.Tensor,   # (obs_dim,) or (1, obs_dim)
    t: float,
    n_actions: int,
    eps_sched: EpsilonGreedySchedule,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns action tensor shape (1, 1) with dtype long.
    """
    eps = eps_sched.value()
    eps_sched.step()


    if random.random() < eps:
        a = random.randrange(n_actions)
        return torch.tensor([[a]], dtype=torch.long, device=device)

    # Greedy action using Q(s,a,t)
    state = _ensure_2d(state).to(device)
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    x = concat_state_time(state, t_tensor)
    q_values = policy_net(x)  # (1, n_actions)
    return q_values.argmax(dim=1).view(1, 1)



def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    horizon: int,
    device: torch.device,
) -> Optional[float]:
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)

    # Unpack dataclass transitions into lists
    batch_state      = [tr.state for tr in transitions]
    batch_action     = [tr.action for tr in transitions]
    batch_reward     = [tr.reward for tr in transitions]
    batch_next_state = [tr.next_state for tr in transitions]
    batch_t          = [tr.t for tr in transitions]
    batch_done     = [tr.done for tr in transitions]  # optional

    # Prepare tensors
    state_batch = torch.cat([_ensure_2d(s) for s in batch_state]).to(device)   # (B, obs_dim)
    action_batch = torch.cat(batch_action).to(device)                         # (B, 1)
    reward_batch = torch.cat(batch_reward).to(device).view(-1)                # (B,)
    t_batch = torch.tensor(batch_t, dtype=torch.float32, device=device)          # (B,)

    # Mask for non-terminal transitions
    non_final_mask = torch.tensor(
        [ns is not None for ns in batch_next_state],
        device=device,
        dtype=torch.bool,
    )

    if non_final_mask.any():
        non_final_next_states = torch.cat(
            [_ensure_2d(ns) for ns in batch_next_state if ns is not None]
        ).to(device)  # (B_nonfinal, obs_dim)

        dt = 1.0 / (horizon - 1) #since we are using normalized time
        non_final_t_next = torch.tensor(
            [ti + dt for (ti, ns) in zip(batch_t, batch_next_state) if ns is not None],
            dtype=torch.float32,
            device=device,)  # (B_nonfinal,)
    else:
        non_final_next_states = None
        non_final_t_next = None

    # Q(s,a,t) for taken actions
    x = concat_state_time(state_batch, t_batch)                 # (B, obs_dim+1)
    q_values = policy_net(x)                                    # (B, n_actions)
    state_action_values = q_values.gather(1, action_batch).squeeze(1)  # (B,)

    # Compute target: r + gamma * max_a Q_target(s', a, t+1)
    next_state_values = torch.zeros(batch_size, device=device)  # (B,)
    if non_final_mask.any() and non_final_next_states is not None and non_final_t_next is not None:
        x_next = concat_state_time(non_final_next_states, non_final_t_next)  # (B_nonfinal, obs_dim+1)
        next_q = target_net(x_next)                                          # (B_nonfinal, n_actions)
        next_state_values[non_final_mask] = next_q.max(dim=1).values

    expected_state_action_values = reward_batch + gamma * next_state_values

    # Loss (Huber / SmoothL1) as in tutorial
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100.0)
    
    optimizer.step()

    return float(loss.item())



def hard_update_target_net(target_net: nn.Module, policy_net: nn.Module) -> None:
    target_net.load_state_dict(policy_net.state_dict())