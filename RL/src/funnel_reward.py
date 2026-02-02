from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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

        if collided and self.spec.collision_penalty != 0.0:
            r -= float(self.spec.collision_penalty)

        return float(r)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: set bounds and goal for your house
    spec = FunnelSpec(
        operator="F",
        a=0,
        b=80,
        t_star=50,
        gamma_inf=0.01,
        x_min=0.0,
        x_max=12.0,
        y_min=0.0,
        y_max=9.0,
    )

    goal_center = (6.0, 4.0)
    goal_radius = 0.2

    fr = FunnelReward(goal_center, goal_radius, spec)

    # time axis
    T = spec.b  # plot up to b
    ts = np.arange(0, T + 1)

    # gamma(t)
    gammas = np.array([funnel_gamma(int(t), fr.gamma0, fr.spec.gamma_inf, fr.l) for t in ts])


    # also plot the lower robustness bound: rho_max - gamma(t)
    funnel = fr.rho_max - gammas

    print("Funnel params:")
    print(f"  rho_max      = {fr.rho_max:.4f}")
    print(f"  rho_min_approx= {fr.rho_min_approx:.4f}")
    print(f"  gamma0       = {fr.gamma0:.4f}")
    print(f"  gamma_inf    = {fr.spec.gamma_inf:.4f}")
    print(f"  t_star       = {fr.t_star}")
    print(f"  l            = {fr.l:.6f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ts, gammas, label=r"$\gamma(t)$")
    plt.axhline(fr.spec.gamma_inf, linestyle="--", linewidth=1, label=r"$\gamma_\infty$")
    plt.axvline(fr.t_star, linestyle="--", linewidth=1, label=r"$t^*$")

    plt.xlabel("t (time step)")
    plt.ylabel(r"$\gamma(t)$")
    plt.title("Funnel Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: plot the implied robustness lower bound over time
    plt.figure(figsize=(8, 5))
    plt.plot(ts, funnel, label=r"$\rho_{\max} - \gamma(t)$")
    plt.axvline(fr.t_star, linestyle="--", linewidth=1, label=r"$t^*$")
    plt.axhline(fr.rho_max, linestyle="--", linewidth=1, label=r"$\rho_{\max}$")
    plt.xlabel("t (time step)")
    plt.ylabel("Lower bound on robustness")
    plt.title("Implied Robustness Lower Bound")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
