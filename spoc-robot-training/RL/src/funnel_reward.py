from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

import numpy as np


def circle_reach_robustness(pos_xy: Tuple[float, float],
                            center_xy: Tuple[float, float],
                            radius: float) -> float:
    """
    Robustness for predicate: ||pos - center|| <= radius
    """
    pos = np.array(pos_xy, dtype=float)
    c = np.array(center_xy, dtype=float)
    return float(radius - np.linalg.norm(pos - c))


def funnel_gamma(t: int, gamma0: float, gamma_inf: float, l: float) -> float:
    return float((gamma0 - gamma_inf) * np.exp(-l * float(t)) + gamma_inf)


def choose_l(t_star: int, gamma0: float, gamma_inf: float, rho_max: float) -> float:
    """
    From the paper (Table I / Sec III-A):
      l = (1/t*) * ln((gamma0 - gamma_inf) / (rho_max - gamma_inf))
    """
    t_star = max(int(t_star), 1)
    numer = max(gamma0 - gamma_inf, 1e-8)
    denom = max(rho_max - gamma_inf, 1e-8)
    return float((1.0 / t_star) * np.log(numer / denom))


@dataclass
class FunnelSpec:
    operator: str = "F"
    a: int = 0
    b: int = 200
    t_star: Optional[int] = None
    gamma_inf: float = 0.01

    # NEW: world bounds in environment coordinates
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 10.0

    collision_penalty: float = 0.0


class FunnelReward:
    def __init__(self, goal_center, goal_radius, spec: FunnelSpec, time_origin: Optional[int] = None):
        self.goal_center = (float(goal_center[0]), float(goal_center[1]))
        self.goal_radius = float(goal_radius)
        self.spec = spec

        self.rho_max = self.goal_radius

        # Approximate rho_min using farthest CORNER of the true bounds
        corners = np.array([
            [spec.x_min, spec.y_min],
            [spec.x_min, spec.y_max],
            [spec.x_max, spec.y_min],
            [spec.x_max, spec.y_max],
        ], dtype=float)
        c = np.array(self.goal_center, dtype=float)
        max_dist = float(np.max(np.linalg.norm(corners - c[None, :], axis=1)))

        self.rho_min_approx = self.goal_radius - max_dist
        self.gamma0 = float(self.rho_max - self.rho_min_approx)

        # choose t_star
        if spec.t_star is None:
            self.t_star = int((spec.a + spec.b) // 2) if spec.operator.upper() == "F" else int(spec.a)
        else:
            self.t_star = int(spec.t_star)

        if spec.operator.upper() == "G":
            self.t_star = int(spec.a)


        # NEW: funnel start time (defaults to spec.a if not provided)
        self.time_origin = int(spec.a) if time_origin is None else int(time_origin)

        # NEW: choose l based on LOCAL t_star relative to time_origin
        t_star_local = max(self.t_star - self.time_origin, 1)
        self.l = choose_l(t_star_local, self.gamma0, spec.gamma_inf, self.rho_max)
        

        # t_star_local = self.t_star - int(self.spec.a) #shif t_star to local time within [a,b]
        # t_star_local = max(t_star_local, 1)  # avoid divide-by-zero
        # self.l = choose_l(t_star_local, self.gamma0, self.spec.gamma_inf, self.rho_max)



    def __call__(self,
                 pos_xy: Tuple[float, float],
                 t: int,
                 collided: bool = False,
                 goal_center: Optional[Tuple[float, float]] = None,
                 goal_radius: Optional[float] = None) -> float:
        """
        Args:
          pos_xy: (x,y) agent position at time t
          t: integer time step (your simulation_timer)
          collided: if True, add collision_penalty (optional)
          goal_center/goal_radius: override if goal moves (optional)

        Returns:
          funnel-shaped reward scalar (float)
        """
        center = self.goal_center if goal_center is None else (float(goal_center[0]), float(goal_center[1]))
        radius = self.goal_radius if goal_radius is None else float(goal_radius)

        # If radius changes dynamically, update rho_max accordingly (rare)
        rho_max = radius

        rho = circle_reach_robustness(pos_xy, center, radius)
        #gam = funnel_gamma(int(t), self.gamma0, self.spec.gamma_inf, self.l)

        t_local = int(t) - int(self.time_origin)

        if t_local < 0:
            # task not active yet; you can return 0 or keep it inactive elsewhere
            t_local = 0

        gam = funnel_gamma(t_local, self.gamma0, self.spec.gamma_inf, self.l)

        r = rho + gam - rho_max

        if collided and self.spec.collision_penalty != 0.0:
            r -= float(self.spec.collision_penalty)

        return float(r)

@dataclass   
class TaskDef:
    """
    One STL subtask with its own funnel spec and predicate (goal circle here).
    Active when t in [a, b].
    """
    reward_fn: "FunnelReward"        
    a: int
    b: int
    goal_id: Optional[int] = None 
    start: Optional[int] = None  # funnel time origin for this task 


class MultiTaskFunnelReward:
    """
    Implements the paper's conjunction reward rule:
      - if only one task active at time t -> use its reward
      - if multiple tasks active -> min of their rewards
      - if none active -> 0.0 (or you can choose something else)
    """
    def __init__(self, tasks: Sequence[TaskDef], none_value: float = 0.0):
        self.tasks = list(tasks)
        self.none_value = float(none_value)

        prev_b = None
        for td in self.tasks:
            if td.start is None:
                td.start = td.a if prev_b is None else prev_b
            prev_b = td.b

    def __call__(
        self,
        pos_xy: Tuple[float, float],
        t: int,
        collided: bool = False,
        # provide these if you have moving goals; otherwise ignore
        goal_centers: Optional[dict] = None,
        goal_radii: Optional[dict] = None,
    ) -> float:
        
        t = int(t)
        active_vals: List[float] = []

        for td in self.tasks: #td = task definition
            t0 = int(td.start) if td.start is not None else int(td.a)
            
            if t < t0 or t > int(td.b):
                continue

            # optional dynamic goal override
            gc = None
            gr = None
            if td.goal_id is not None and goal_centers is not None and goal_radii is not None:
                gc = goal_centers[td.goal_id]
                gr = goal_radii[td.goal_id]

            r_i = td.reward_fn(
                pos_xy=pos_xy,
                t=t,
                collided=collided,
                goal_center=gc,
                goal_radius=gr,
            )
            active_vals.append(float(r_i))

        if len(active_vals) == 0:
            return self.none_value
        if len(active_vals) == 1:
            return active_vals[0]
        return float(np.min(active_vals))


class ModifiedFunnelReward: #tries to satisfy the task by the end of the time interval
    def __init__(self, goal_center, goal_radius, spec: FunnelSpec):
        self.goal_center = (float(goal_center[0]), float(goal_center[1]))
        self.goal_radius = float(goal_radius)
        self.spec = spec

        self.rho_max = self.goal_radius

        # Approximate rho_min using farthest CORNER of the true bounds
        corners = np.array([
            [spec.x_min, spec.y_min],
            [spec.x_min, spec.y_max],
            [spec.x_max, spec.y_min],
            [spec.x_max, spec.y_max],
        ], dtype=float)
        c = np.array(self.goal_center, dtype=float)
        max_dist = float(np.max(np.linalg.norm(corners - c[None, :], axis=1)))

        self.rho_min_approx = self.goal_radius - max_dist
        self.gamma0 = float(self.rho_max - self.rho_min_approx)

        # choose t_star
        if spec.t_star is None:
            self.t_star = int((spec.a + spec.b) // 2) if spec.operator.upper() == "F" else int(spec.a)
        else:
            self.t_star = int(spec.t_star)

        if spec.operator.upper() == "G":
            self.t_star = int(spec.a)

        # same choose_l as before
        self.l = choose_l(self.t_star, self.gamma0, spec.gamma_inf, self.rho_max)


    def __call__(self,
                 pos_xy: Tuple[float, float],
                 t: int,
                 collided: bool = False,
                 goal_center: Optional[Tuple[float, float]] = None,
                 goal_radius: Optional[float] = None) -> float:
        """
        Args:
          pos_xy: (x,y) agent position at time t
          t: integer time step (your simulation_timer)
          collided: if True, add collision_penalty (optional)
          goal_center/goal_radius: override if goal moves (optional)

        Returns:
          funnel-shaped reward scalar (float)
        """
        center = self.goal_center if goal_center is None else (float(goal_center[0]), float(goal_center[1]))
        radius = self.goal_radius if goal_radius is None else float(goal_radius)

        # If radius changes dynamically, update rho_max accordingly (rare)
        rho_max = radius

        rho = circle_reach_robustness(pos_xy, center, radius)
        gam = funnel_gamma(int(t), self.gamma0, self.spec.gamma_inf, self.l)

        r = rho + gam - rho_max
        delta = 0.2 #tolerance value
        tracking_err = max(0.0, abs(r) - delta)
        #r2 = rho_max - rho


        # if collided and self.spec.collision_penalty != 0.0:
        #     r -= float(self.spec.collision_penalty)

        return float(r - 2*tracking_err)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tasks = [
            dict(
                name="Task1: F[0,60] goal1",
                spec=FunnelSpec(
                    operator="F",
                    a=0,
                    b=45,
                    t_star=30,         # global t*, but your FunnelReward now shifts internally by a
                    gamma_inf=-0.1,
                    x_min=0.0, x_max=5.77,
                    y_min=0.0, y_max=5.77,
                ),
                goal_center=(5.2, 1.8),
                goal_radius=0.4,
            ),
            dict(
                name="Task2: F[46,85] goal2",
                spec=FunnelSpec(
                    operator="F",
                    a=46,
                    b=85,
                    t_star=70,         # pick a global t* in [a,b] (e.g., closer to b for "eventually")
                    gamma_inf=-0.1,
                    x_min=0.0, x_max=5.77,
                    y_min=0.0, y_max=5.77,
                ),
                goal_center=(1.2, 4.6),
                goal_radius=0.4,
            ),
            # Add more tasks here...
        ]

        # ---- Build FunnelReward objects ----
    frs = []
    for td in tasks:
        frs.append(FunnelReward(td["goal_center"], td["goal_radius"], td["spec"]))

    # ---- Choose a global time axis that covers all windows ----
    T = max(td["spec"].b for td in tasks)
    ts = np.arange(0, T + 1)

    # ---- Compute each task's gamma(t) (SHIFTED by its own a) and funnel bound ----
    all_gammas = []
    all_funnels = []

    for fr, td in zip(frs, tasks):
        a = int(fr.spec.a)
        b = int(fr.spec.b)

        gammas = np.zeros_like(ts, dtype=float)
        funnel_lb = np.zeros_like(ts, dtype=float)

        for i, t in enumerate(ts):
            if t < a or t > b:
                gammas[i] = np.nan      # not active outside its window (for cleaner plots)
                funnel_lb[i] = np.nan
                continue

            # IMPORTANT: local time so it "starts" at t=a
            t_local = int(t) - a

            g = funnel_gamma(t_local, fr.gamma0, fr.spec.gamma_inf, fr.l)
            gammas[i] = g
            funnel_lb[i] = fr.rho_max - g

        all_gammas.append(gammas)
        all_funnels.append(funnel_lb)

        print(f"\n{td['name']}")
        print("Funnel params:")
        print(f"  window [a,b] = [{fr.spec.a},{fr.spec.b}]")
        print(f"  rho_max      = {fr.rho_max:.4f}")
        print(f"  rho_min_approx= {fr.rho_min_approx:.4f}")
        print(f"  gamma0       = {fr.gamma0:.4f}")
        print(f"  gamma_inf    = {fr.spec.gamma_inf:.4f}")
        print(f"  t_star (global)= {fr.t_star}")
        print(f"  l            = {fr.l:.6f}")

    # ---- Plot all gamma(t) curves on one figure ----
    plt.figure(figsize=(9, 5))
    for gammas, td, fr in zip(all_gammas, tasks, frs):
        plt.plot(ts, gammas, label=td["name"])
        # show activation boundaries
        plt.axvline(int(fr.spec.a), linestyle=":", linewidth=1)
        plt.axvline(int(fr.spec.b), linestyle=":", linewidth=1)

    plt.xlabel("global time t")
    plt.ylabel(r"$\gamma(t)$ (task-local time inside window)")
    plt.title("Funnel Functions for Multiple Tasks (time-shifted by each task's a)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot implied robustness lower bounds rho_max - gamma(t) ----
    plt.figure(figsize=(9, 5))
    for funnel_lb, td, fr in zip(all_funnels, tasks, frs):
        plt.plot(ts, funnel_lb, label=td["name"])
        plt.axvline(int(fr.spec.a), linestyle=":", linewidth=1)
        plt.axvline(int(fr.spec.b), linestyle=":", linewidth=1)
        plt.axhline(fr.rho_max, linestyle="--", linewidth=1, label=r"$\rho_{\max}$")
        
    plt.xlabel("global time t")
    plt.ylabel(r"Lower bound: $\rho_{\max} - \gamma(t)$")
    plt.title("Implied Robustness Lower Bounds (per-task, active only in its window)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()