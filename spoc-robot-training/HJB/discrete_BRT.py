import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discrete action set
# ---------------------------------------------------------------------------

ACTIONS = jnp.array([
    [0.3,  0.0 ],   # move ahead
    [-0.3,   0.0 ],   # move backwards
    [0.0,   0.1 ],   # rotate left  ~6 deg/s
    [0.0,  -0.1 ],   # rotate right ~6 deg/s
    [0.0,   0.52],   # rotate left  ~30 deg/s
    [0.0,  -0.52],   # rotate right ~30 deg/s
])  # shape: (K, 2)  -- each row is [v, omega]


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class DiscreteUnicycle(hj.Dynamics):
    """
    Unicycle with a finite (discrete) set of [v, ω] actions and an additive
    positional disturbance d ∈ Ball(0, disturbance_radius).

    State : [x, y, θ]
    ODE   : ẋ = v·cos θ + d₀,  ẏ = v·sin θ + d₁,  θ̇ = ω

    For BRT the controller minimises and the disturbance maximises the value.
    """

    def __init__(self, actions, control_mode="min", disturbance_radius=0.0):
        self.actions = actions  # (K, 2)

        max_v = float(jnp.max(jnp.abs(actions[:, 0])))
        max_w = float(jnp.max(jnp.abs(actions[:, 1])))

        control_space     = hj.sets.Box(jnp.array([-max_v, -max_w]),
                                        jnp.array([ max_v,  max_w]))
        disturbance_space = hj.sets.Ball(jnp.zeros(2), disturbance_radius)

        super().__init__(control_mode, "max", control_space, disturbance_space)

    # -- ODE -----------------------------------------------------------------

    def __call__(self, state, control, disturbance, time):
        _, _, theta = state
        v, omega = control
        return jnp.array([v * jnp.cos(theta) + disturbance[0],
                           v * jnp.sin(theta) + disturbance[1],
                           omega])

    # -- Hamiltonian ---------------------------------------------------------

    def _state_dot(self, state, action):
        _, _, theta = state
        v, omega = action
        return jnp.array([v * jnp.cos(theta),
                           v * jnp.sin(theta),
                           omega])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Return the discrete action that min/max-imises ∇V·f."""
        # Control term: min over discrete actions
        h_vals = jax.vmap(
            lambda a: jnp.dot(grad_value, self._state_dot(state, a))
        )(self.actions)

        best_idx = jnp.argmin(h_vals) if self.control_mode == "min" \
                   else jnp.argmax(h_vals)

        # Disturbance term: adversarial maximiser over Ball
        # G_d = [[1,0],[0,1],[0,0]]  =>  G_d^T @ grad_value = grad_value[:2]
        optimal_disturbance = self.disturbance_space.extreme_point(grad_value[:2])

        return self.actions[best_idx], optimal_disturbance

    # -- CFL condition -------------------------------------------------------

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Upper bound on characteristic speeds in each state dimension."""
        max_v = jnp.max(jnp.abs(self.actions[:, 0]))
        max_w = jnp.max(jnp.abs(self.actions[:, 1]))
        d_rad = self.disturbance_space.radius
        # Disturbance couples into x and y only (G_d = [[1,0],[0,1],[0,0]])
        return jnp.array([max_v + d_rad, max_v + d_rad, max_w])


# ---------------------------------------------------------------------------
# Continuous dynamics (for comparison)
# ---------------------------------------------------------------------------

class ContinuousUnicycle(hj.ControlAndDisturbanceAffineDynamics):
    """
    Unicycle with a continuous Box control space and an additive positional
    disturbance d ∈ Ball(0, disturbance_radius).

    State   : [x, y, θ]
    ODE     : ẋ = v·cos θ + d₀,  ẏ = v·sin θ + d₁,  θ̇ = ω
    Control : [v, ω]  with v ∈ [v_min, v_max], ω ∈ [ω_min, ω_max]
    """

    def __init__(self, v_min, v_max, omega_min, omega_max,
                 control_mode="min", disturbance_radius=0.0):
        control_space = hj.sets.Box(
            jnp.array([v_min,    omega_min]),
            jnp.array([v_max,    omega_max]),
        )
        disturbance_space = hj.sets.Ball(jnp.zeros(2), disturbance_radius)
        super().__init__(control_mode, "max", control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        del state, time
        return jnp.zeros(3)

    def control_jacobian(self, state, time):
        del time
        _, _, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],
            [jnp.sin(theta), 0.0],
            [0.0,            1.0],
        ])

    def disturbance_jacobian(self, state, time):
        del state, time
        # Disturbance enters x and y additively: ẋ += d₀, ẏ += d₁
        return jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])


# ---------------------------------------------------------------------------
# BRT computation
# ---------------------------------------------------------------------------

def compute_brt(dynamics, grid, target_values, time_horizon, n_time_steps=51):
    """Solve the BRT backwards in time and return all snapshots."""
    solver_settings = hj.SolverSettings.with_accuracy(
        "high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )
    times = np.linspace(0.0, -time_horizon, n_time_steps)
    all_values = hj.solve(solver_settings, dynamics, grid, times, target_values,
                          progress_bar=True)
    return times, all_values


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_brt_slice(ax, grid, values, target_center, target_radius, theta_deg, title):
    ntheta    = grid.shape[2]
    theta_idx = int(round(theta_deg / 360.0 * ntheta)) % ntheta

    cf = ax.contourf(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[:, :, theta_idx].T,
        levels=50, cmap="RdBu_r",
    )
    plt.colorbar(cf, ax=ax, label="Value")
    ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[:, :, theta_idx].T,
        levels=[0], colors="black", linewidths=2,
    )
    circle = plt.Circle(target_center, target_radius,
                        fill=False, color="green", linewidth=2, linestyle="--")
    ax.add_patch(circle)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")


def visualize_brt(grid, times, all_values, target_center, target_radius,
                  theta_degs=(0, 90, 180, 270), time_index=-1):
    """Plot the BRT at a chosen time snapshot for several heading slices."""
    values = all_values[time_index]
    t_val  = times[time_index]
    n      = len(theta_degs)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, theta_deg in zip(axes, theta_degs):
        _plot_brt_slice(ax, grid, values, target_center, target_radius,
                        theta_deg, f"θ = {theta_deg}°  (t = {t_val:.1f} s)")

    plt.suptitle("Discrete-Action Unicycle BRT  |  green dashed = target, "
                 "black = BRT boundary", fontsize=11)
    plt.tight_layout()
    return fig, axes


def visualize_comparison(grid, times,
                         disc_values, cont_values,
                         target_center, target_radius,
                         theta_degs=(0, 90, 180, 270), time_index=-1):
    """
    Side-by-side comparison: discrete (top row) vs continuous (bottom row)
    BRT at the chosen time snapshot for several heading slices.
    """
    disc_snap = disc_values[time_index]
    cont_snap = cont_values[time_index]
    t_val     = times[time_index]
    n         = len(theta_degs)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    for col, theta_deg in enumerate(theta_degs):
        _plot_brt_slice(axes[0, col], grid, disc_snap,
                        target_center, target_radius, theta_deg,
                        f"Discrete  θ={theta_deg}°")
        _plot_brt_slice(axes[1, col], grid, cont_snap,
                        target_center, target_radius, theta_deg,
                        f"Continuous  θ={theta_deg}°")

        # Overlay discrete boundary on continuous panel for easy comparison
        ntheta    = grid.shape[2]
        theta_idx = int(round(theta_deg / 360.0 * ntheta)) % ntheta
        axes[1, col].contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            disc_snap[:, :, theta_idx].T,
            levels=[0], colors="orange", linewidths=1.5, linestyles="--",
        )

    plt.suptitle(
        f"BRT at t = {t_val:.1f} s  |  black = boundary, orange dashed = discrete boundary on continuous panel",
        fontsize=11,
    )
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- shared settings ----------------------------------------------------
    TIME_HORIZON  = 10.0
    N_STEPS       = 11
    target_center = np.array([0.0, 0.0])
    target_radius = 0.3
    THETA_DEGS    = (0, 90, 180, 270)

    # Continuous bounds derived from the discrete action set
    V_MIN, V_MAX         = float(jnp.min(ACTIONS[:, 0])), float(jnp.max(ACTIONS[:, 0]))
    OMEGA_MIN, OMEGA_MAX = float(jnp.min(ACTIONS[:, 1])), float(jnp.max(ACTIONS[:, 1]))
    DISTURBANCE_RADIUS   = 0.05   # m/s additive noise on x, y

    # --- shared grid --------------------------------------------------------
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([-5.0, -5.0, 0.0]),
            hi=np.array([ 5.0,  5.0, 2 * np.pi]),
        ),
        shape=(101, 101, 36),
        periodic_dims=2,
    )

    target_values = (
        jnp.linalg.norm(grid.states[..., :2] - jnp.array(target_center), axis=-1)
        - target_radius
    )

    # --- discrete BRT -------------------------------------------------------
    print("Computing discrete-action BRT …")
    disc_dynamics = DiscreteUnicycle(ACTIONS, control_mode="min",
                                     disturbance_radius=DISTURBANCE_RADIUS)
    times, disc_brt = compute_brt(disc_dynamics, grid, target_values,
                                  TIME_HORIZON, N_STEPS)

    disc_frac = float((disc_brt[-1] <= 0).sum()) / disc_brt[-1].size
    print(f"Discrete  reachable fraction: {disc_frac * 100:.1f}%")

    # --- continuous BRT -----------------------------------------------------
    print("Computing continuous-action BRT …")
    cont_dynamics = ContinuousUnicycle(V_MIN, V_MAX, OMEGA_MIN, OMEGA_MAX,
                                       control_mode="min",
                                       disturbance_radius=DISTURBANCE_RADIUS)
    _, cont_brt = compute_brt(cont_dynamics, grid, target_values,
                               TIME_HORIZON, N_STEPS)

    cont_frac = float((cont_brt[-1] <= 0).sum()) / cont_brt[-1].size
    print(f"Continuous reachable fraction: {cont_frac * 100:.1f}%")
    print(f"Continuous bounds: v ∈ [{V_MIN:.2f}, {V_MAX:.2f}], "
          f"ω ∈ [{OMEGA_MIN:.2f}, {OMEGA_MAX:.2f}], "
          f"disturbance radius = {DISTURBANCE_RADIUS}")

    # --- individual plots ---------------------------------------------------
    fig, _ = visualize_brt(grid, times, disc_brt, target_center, target_radius,
                            THETA_DEGS, time_index=-1)
    plt.savefig("discrete_brt.png", dpi=150, bbox_inches="tight")

    fig, _ = visualize_brt(grid, times, cont_brt, target_center, target_radius,
                            THETA_DEGS, time_index=-1)
    plt.savefig("continuous_brt.png", dpi=150, bbox_inches="tight")

    # --- comparison plot ----------------------------------------------------
    fig, _ = visualize_comparison(grid, times, disc_brt, cont_brt,
                                   target_center, target_radius,
                                   THETA_DEGS, time_index=-1)
    plt.savefig("brt_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
