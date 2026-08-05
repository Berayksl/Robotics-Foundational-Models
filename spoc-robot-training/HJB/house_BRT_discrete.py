"""
house_BRT_discrete.py

Computes the reach-avoid BRT for a unicycle navigating a house environment
using a *discrete* action set instead of a continuous control box.

Geometry utilities and visualisation are re-used from house_BRT.py.
The DiscreteUnicycle dynamics class is re-used from discrete_test.py.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import hj_reachability as hj
from jax import scipy as jsp

try:
    from occupancy_map import create_environment
    from geometry import _compute_env_bounds_from_geom
except ImportError:
    from .occupancy_map import create_environment
    from .geometry import _compute_env_bounds_from_geom

# Geometry helpers and visualisation re-used verbatim from house_BRT
from house_BRT import (
    create_obstacle_values_from_geometry,
    create_target_values,
    visualize_house_brt,
    visualize_brt_over_time,
)

# Discrete dynamics
from discrete_BRT import DiscreteUnicycle, ACTIONS


# ---------------------------------------------------------------------------
# BRT computation
# ---------------------------------------------------------------------------

def compute_house_brt_discrete(
    actions,
    house_index,
    target_center,
    target_radius,
    time_horizon=5.0,
    robot_radius=0.2,
    wall_thickness=0.1,
    disturbance_radius=0.0,
    grid_resolution=(101, 101, 36),
):
    """
    Compute the reach-avoid BRT (single snapshot) for a house environment
    using a discrete action unicycle.

    Parameters
    ----------
    actions            : (K, 2) jnp array of [v, omega] pairs
    house_index        : index of the house layout to load
    target_center      : (x, y) goal position
    target_radius      : radius of the goal region
    time_horizon       : backward time horizon (seconds)
    robot_radius       : robot collision margin
    wall_thickness     : wall collision thickness
    disturbance_radius : radius of additive Ball disturbance on (x, y)
    grid_resolution    : (nx, ny, ntheta)

    Returns
    -------
    grid, target_values, obstacle_values, brt_values, geom
    """
    fig, ax, geom = create_environment(house_index, render=True)
    plt.close(fig)

    x_min, x_max, y_min, y_max = _compute_env_bounds_from_geom(geom)

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([x_min, y_min, 0.0]),
            hi=np.array([x_max, y_max, 2 * np.pi]),
        ),
        shape=grid_resolution,
        periodic_dims=2,
    )

    obstacle_values = create_obstacle_values_from_geometry(
        grid, geom,
        wall_thickness=wall_thickness,
        robot_radius=robot_radius,
    )
    target_values = create_target_values(grid, target_center, target_radius)

    dynamics = DiscreteUnicycle(actions, control_mode="min",
                                disturbance_radius=disturbance_radius)

    solver_settings = hj.SolverSettings.with_accuracy(
        "high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        value_postprocessor=hj.solver.static_obstacle(-obstacle_values),
    )

    brt_values = hj.step(
        solver_settings, dynamics, grid,
        time=0.0,
        values=target_values,
        target_time=-time_horizon,
    )

    return grid, target_values, obstacle_values, brt_values, geom


def compute_house_brt_discrete_over_time(
    actions,
    house_index,
    target_center,
    target_radius,
    time_horizon=5.0,
    n_time_steps=51,
    robot_radius=0.2,
    wall_thickness=0.1,
    disturbance_radius=0.0,
    grid_resolution=(101, 101, 36),
):
    """
    Compute reach-avoid BRTs at multiple time snapshots for a house environment
    using a discrete action unicycle.

    Parameters
    ----------
    actions            : (K, 2) jnp array of [v, omega] pairs
    house_index        : index of the house layout to load
    target_center      : (x, y) goal position
    target_radius      : radius of the goal region
    time_horizon       : backward time horizon (seconds, positive)
    n_time_steps       : number of time snapshots to save
    robot_radius       : robot collision margin
    wall_thickness     : wall collision thickness
    disturbance_radius : radius of additive Ball disturbance on (x, y)
    grid_resolution    : (nx, ny, ntheta)

    Returns
    -------
    grid       : hj_reachability Grid
    times      : 1-D array, shape (n_time_steps,), values in [0, -time_horizon]
    target_values   : signed distance to target, shape (*grid_resolution)
    obstacle_values : signed distance to obstacles, shape (*grid_resolution)
    all_brt_values  : (n_time_steps, *grid_resolution) array of BRT values
    geom       : house geometry dict
    """
    geom = create_environment(house_index, render=False)

    x_min, x_max, y_min, y_max = _compute_env_bounds_from_geom(geom)

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([x_min, y_min, 0.0]),
            hi=np.array([x_max, y_max, 2 * np.pi]),
        ),
        shape=grid_resolution,
        periodic_dims=2,
    )

    obstacle_values = create_obstacle_values_from_geometry(
        grid, geom,
        wall_thickness=wall_thickness,
        robot_radius=robot_radius,
    )
    target_values = create_target_values(grid, target_center, target_radius)

    dynamics = DiscreteUnicycle(actions, control_mode="min",
                                disturbance_radius=disturbance_radius)

    solver_settings = hj.SolverSettings.with_accuracy(
        "high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        value_postprocessor=hj.solver.static_obstacle(-obstacle_values),
    )

    times = np.linspace(0.0, -time_horizon, n_time_steps)

    all_brt_values = hj.solve(
        solver_settings,
        dynamics,
        grid,
        times,
        target_values,
        progress_bar=True,
    )

    return grid, times, target_values, obstacle_values, all_brt_values, geom


# ---------------------------------------------------------------------------
# Value queries  (identical logic to house_BRT.py, just copied here so this
# module is self-contained without circular imports)
# ---------------------------------------------------------------------------

def get_brt_value(grid, brt_values, state):
    """
    Interpolated BRT value at a single state.

    Parameters
    ----------
    state : [x, y, theta_deg]
    """
    state = jnp.asarray(state, dtype=float)
    state = state.at[2].set(jnp.deg2rad(state[2]))

    indices = []
    for i in range(3):
        coord = grid.coordinate_vectors[i]
        lo, hi, n = coord[0], coord[-1], len(coord)
        s = state[i] % (2 * jnp.pi) if i == 2 else state[i]
        indices.append((s - lo) / (hi - lo) * (n - 1))

    value = jsp.ndimage.map_coordinates(
        brt_values,
        jnp.array(indices).reshape(-1, 1),
        order=1,
        mode="wrap",
    )
    return float(value[0])


def get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go):
    """
    Interpolated BRT value at a single state and time-to-go.

    Parameters
    ----------
    state       : [x, y, theta_deg]
    time_to_go  : seconds remaining (0 = must be at target now)
    """
    state = jnp.asarray(state, dtype=float)
    state = state.at[2].set(jnp.deg2rad(state[2]))

    times = np.asarray(times)
    times_flipped   = times[::-1]
    indices_flipped = np.arange(len(times))[::-1]
    time_idx = np.interp(-time_to_go, times_flipped, indices_flipped)

    indices = [time_idx]
    for i in range(3):
        coord = grid.coordinate_vectors[i]
        lo, hi, n = coord[0], coord[-1], len(coord)
        s = state[i] % (2 * jnp.pi) if i == 2 else state[i]
        indices.append((s - lo) / (hi - lo) * (n - 1))

    value = jsp.ndimage.map_coordinates(
        all_brt_values,
        jnp.array(indices).reshape(-1, 1),
        order=1,
        mode="wrap",
    )
    return float(value[0])


def get_brt_values_batch(grid, brt_values, states):
    """
    Interpolated BRT values for a batch of states.

    Parameters
    ----------
    states : (N, 3) array of [x, y, theta_deg]
    """
    states = jnp.asarray(states, dtype=float)
    states = states.at[:, 2].set(jnp.deg2rad(states[:, 2]))

    indices = []
    for i in range(3):
        coord = grid.coordinate_vectors[i]
        lo, hi, n = coord[0], coord[-1], len(coord)
        s = states[:, i] % (2 * jnp.pi) if i == 2 else states[:, i]
        indices.append((s - lo) / (hi - lo) * (n - 1))

    return jsp.ndimage.map_coordinates(
        brt_values,
        jnp.stack(indices, axis=0),
        order=1,
        mode="wrap",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    house_index   = 152
    target_center = (7.0, 5.5)
    target_radius = 0.5

    ######################################
    # BRT OVER TIME
    ######################################

    grid, times, target_values, obstacle_values, all_brt_values, geom = \
        compute_house_brt_discrete_over_time(
            actions=ACTIONS,
            house_index=house_index,
            target_center=target_center,
            target_radius=target_radius,
            time_horizon=20.0,
            n_time_steps=21,
            robot_radius=0.2,
            wall_thickness=0.1,
            disturbance_radius=0.00,
        )

    print(f"Times shape     : {times.shape}")
    print(f"BRT values shape: {all_brt_values.shape}")
    print(f"Time range      : [{times[0]:.1f}, {times[-1]:.1f}] s")

    fig, axes = visualize_brt_over_time(
        grid, times, all_brt_values, geom,
        target_center=target_center,
        target_radius=target_radius,
        theta_deg=0,
        time_indices=[0, 5, 10, 20],
    )
    plt.suptitle("Discrete-Action Unicycle — House BRT", fontsize=13)
    plt.tight_layout()
    plt.savefig("house_brt_discrete.png", dpi=150, bbox_inches="tight")
    plt.show()

    ######################################
    # Value queries at a test state
    ######################################

    state = [6.0, 2.0, 0.0]   # [x, y, theta_deg]

    for ttg in [20.0, 10.0, 2.0, 0.0]:
        v = get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go=ttg)
        inside = "inside BRT" if v <= 0 else "outside BRT"
        print(f"time_to_go={ttg:5.1f}s  value={v:+.4f}  ({inside})")
