import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import hj_reachability as hj
from matplotlib.path import Path
from jax import scipy as jsp
import jax
try:
    from occupancy_map import create_environment
    from geometry import _compute_env_bounds_from_geom
except ImportError:
    from .occupancy_map import create_environment
    from .geometry import _compute_env_bounds_from_geom


def point_to_segment_distance(points, seg_start, seg_end):
    """
    Compute distance from points (Nx2) to a line segment.
    Returns array of shape (N,).
    """
    points = np.asarray(points)
    seg_start = np.asarray(seg_start)
    seg_end = np.asarray(seg_end)
    
    v = seg_end - seg_start
    w = points - seg_start
    
    # Project onto line, clamp to segment
    c1 = np.sum(w * v, axis=-1)
    c2 = np.sum(v * v)
    
    if c2 < 1e-10:
        return np.linalg.norm(w, axis=-1)
    
    t = np.clip(c1 / c2, 0.0, 1.0)
    
    # Closest point on segment
    proj = seg_start + t[..., None] * v
    
    return np.linalg.norm(points - proj, axis=-1)


def point_in_polygon(points, polygon):
    """
    Check if points are inside a polygon.
    points: (N, 2) array
    polygon: (M, 2) array of vertices
    Returns: (N,) boolean array
    """
    path = Path(polygon)
    return path.contains_points(points)


def signed_distance_to_polygon(points, polygon):
    """
    Compute signed distance from points to a polygon.
    Negative inside, positive outside.
    """
    points = np.asarray(points)
    polygon = np.asarray(polygon)
    n_verts = len(polygon)
    
    # Distance to each edge
    min_dist = np.full(len(points), np.inf)
    for i in range(n_verts):
        seg_start = polygon[i]
        seg_end = polygon[(i + 1) % n_verts]
        dist = point_to_segment_distance(points, seg_start, seg_end)
        min_dist = np.minimum(min_dist, dist)
    
    # Sign: negative inside, positive outside
    inside = point_in_polygon(points, polygon)
    signed_dist = np.where(inside, -min_dist, min_dist)
    
    return signed_dist


def signed_distance_to_segment(points, seg_start, seg_end, thickness=0.05):
    """
    Signed distance to a wall segment (thickened by 'thickness').
    Negative within 'thickness' of segment, positive outside.
    """
    dist = point_to_segment_distance(points, seg_start, seg_end)
    return dist - thickness


def create_obstacle_values_from_geometry(grid, geom, wall_thickness=0.1, robot_radius=0.0):
    """
    Convert house geometry to obstacle signed distance function.
    
    Args:
        grid: hj_reachability Grid object
        geom: dict from create_environment containing:
              - 'object_polys': list of Nx2 polygon arrays
              - 'wall_segments': list of ((x0,z0), (x1,z1)) tuples
        wall_thickness: thickness of walls for collision
        robot_radius: inflate obstacles by this amount for safety margin
    
    Returns:
        obstacle_values: jnp array with shape matching grid
                        Negative inside obstacles, positive outside
    """
    # Extract x, y positions from grid states
    # grid.states has shape (*grid_shape, state_dim)
    # For unicycle: state = [x, y, theta]
    grid_shape = grid.states.shape[:-1]
    state_dim = grid.states.shape[-1]
    
    # Flatten spatial dimensions for easier computation
    flat_states = grid.states.reshape(-1, state_dim)
    xy_points = np.array(flat_states[:, :2])  # (N, 2) - x, y positions
    
    # Initialize with large positive distance (far from obstacles)
    min_obstacle_dist = np.full(len(xy_points), np.inf)
    
    # Process object polygons
    for poly in geom['object_polys']:
        if poly is None or len(poly) < 3:
            continue
        dist = signed_distance_to_polygon(xy_points, poly)
        min_obstacle_dist = np.minimum(min_obstacle_dist, dist)
    
    # Process wall segments
    for seg in geom['wall_segments']:
        (x0, z0), (x1, z1) = seg
        dist = signed_distance_to_segment(
            xy_points, 
            np.array([x0, z0]), 
            np.array([x1, z1]), 
            thickness=wall_thickness
        )
        min_obstacle_dist = np.minimum(min_obstacle_dist, dist)
    
    # Apply robot radius (inflate obstacles)
    min_obstacle_dist = min_obstacle_dist - robot_radius
    
    # Reshape back to grid shape
    obstacle_values = min_obstacle_dist.reshape(grid_shape)
    
    return jnp.array(obstacle_values)


def create_target_values(grid, target_center, target_radius):
    """
    Create target set signed distance function.
    Negative inside target, positive outside.
    """
    # Distance to target center in x-y plane
    xy_dist = jnp.linalg.norm(grid.states[..., :2] - jnp.array(target_center), axis=-1)
    return xy_dist - target_radius


# =============================================================================
# Example usage with the house environment
# =============================================================================

def compute_house_brt(dynamics, house_index, target_center, target_radius, 
                      time_horizon=5.0, robot_radius=0.2, wall_thickness=0.1):
    """
    Compute reach-avoid BRT for a house environment. (for a certain target time)
    
    Args:
        house_index: index of house to load
        target_center: (x, y) target location
        target_radius: radius of target region
        time_horizon: how far back to compute BRT
        robot_radius: robot collision radius (safety margin)
        wall_thickness: thickness of walls
    
    Returns:
        grid, target_values, obstacle_values, brt_values, geom
    """
    # Load house geometry
    fig, ax, geom = create_environment(house_index, render=True)
    plt.close(fig)  # Close the figure, we'll make our own
    
    x_min, x_max, y_min, y_max = _compute_env_bounds_from_geom(geom)
    
    # Create grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([x_min, y_min, 0.0]),
            hi=np.array([x_max, y_max, 2 * np.pi])
        ),
        shape=(101, 101, 36),  # Adjust resolution as needed
        periodic_dims=2
    )
    
    # Create obstacle signed distance function
    obstacle_values = create_obstacle_values_from_geometry(
        grid, geom, 
        wall_thickness=wall_thickness,
        robot_radius=robot_radius
    )
    
    # Create target signed distance function
    target_values = create_target_values(grid, target_center, target_radius)
    
    # Solver settings for reach-avoid
    solver_settings = hj.SolverSettings.with_accuracy(
        "high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        value_postprocessor=hj.solver.static_obstacle(-obstacle_values)
    )
    
    # Compute BRT
    time = 0.0
    target_time = -time_horizon
    brt_values = hj.step(solver_settings, dynamics, grid, time, target_values, target_time)
    
    return grid, target_values, obstacle_values, brt_values, geom


def compute_house_brt_over_time(dynamics, house_index, target_center, target_radius, 
                                time_horizon=5.0, n_time_steps=51, robot_radius=0.2, 
                                wall_thickness=0.1, grid_resolution=(101, 101, 36)):
    """
    Compute reach-avoid BRTs over multiple time steps for a house environment. (up to a target time)
    
    Args:
        dynamics: hj_reachability dynamics object (e.g., Unicycle)
        house_index: index of house to load
        target_center: (x, y) target location
        target_radius: radius of target region
        time_horizon: how far back to compute BRT (positive value)
        n_time_steps: number of time snapshots to save
        robot_radius: robot collision radius (safety margin)
        wall_thickness: thickness of walls
        grid_resolution: (nx, ny, ntheta) grid resolution
    
    Returns:
        grid: hj_reachability Grid object
        times: 1D array of time points (from 0 to -time_horizon)
        target_values: initial target set signed distance
        obstacle_values: obstacle signed distance function
        all_brt_values: (n_time_steps, nx, ny, ntheta) array of BRT values over time
        geom: house geometry dict
    """
    # Load house geometry
    geom = create_environment(house_index, render=False)
    
    # Get bounds
    x_min, x_max, y_min, y_max = _compute_env_bounds_from_geom(geom)
    
    # Create grid
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([x_min, y_min, 0.0]),
            hi=np.array([x_max, y_max, 2 * np.pi])
        ),
        shape=grid_resolution,
        periodic_dims=2
    )
    
    # Create obstacle signed distance function
    obstacle_values = create_obstacle_values_from_geometry(
        grid, geom, 
        wall_thickness=wall_thickness,
        robot_radius=robot_radius
    )
    
    # Create target signed distance function
    target_values = create_target_values(grid, target_center, target_radius)
    
    # Solver settings for reach-avoid
    solver_settings = hj.SolverSettings.with_accuracy(
        "high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        value_postprocessor=hj.solver.static_obstacle(-obstacle_values)
    )
    
    # Time array (negative = backward in time)
    times = np.linspace(0, -time_horizon, n_time_steps)
    
    # Compute BRTs over all time steps
    all_brt_values = hj.solve(
        solver_settings, 
        dynamics, 
        grid, 
        times, 
        target_values,
        progress_bar=True
    )
    
    return grid, times, target_values, obstacle_values, all_brt_values, geom


def visualize_house_brt(grid, target_values, obstacle_values, brt_values, geom,
                        target_center, target_radius, theta_idx=0):
    """
    Visualize the BRT overlaid on the house layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Obstacle map
    ax = axes[0]
    ax.set_title('Obstacle Map (black = obstacle)')
    
    cf = ax.contourf(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        obstacle_values[:, :, theta_idx].T,
        levels=50,
        cmap='RdBu'
    )
    plt.colorbar(cf, ax=ax, label='Signed distance')
    
    # Zero contour = obstacle boundary
    ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        obstacle_values[:, :, theta_idx].T,
        levels=[0],
        colors='black',
        linewidths=2
    )
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    
    # Right: BRT
    ax = axes[1]
    theta_val = theta_idx * (2 * np.pi / grid.shape[2])
    ax.set_title(f'Reach-Avoid BRT (θ = {np.degrees(theta_val):.0f}°)')
    
    cf = ax.contourf(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        brt_values[:, :, theta_idx].T,
        levels=50,
        cmap='RdBu_r'
    )
    plt.colorbar(cf, ax=ax, label='Value function')
    
    # BRT boundary
    ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        brt_values[:, :, theta_idx].T,
        levels=[0],
        colors='black',
        linewidths=2,
        linestyles='-'
    )
    
    # Draw target
    target_circle = plt.Circle(target_center, target_radius,
                               fill=False, color='green', linewidth=3, linestyle='--')
    ax.add_patch(target_circle)
    
    # Draw walls
    for seg in geom['wall_segments']:
        (x0, z0), (x1, z1) = seg
        ax.plot([x0, x1], [z0, z1], 'k-', linewidth=2, alpha=0.5)
    
    # Draw object outlines
    for poly in geom['object_polys']:
        if poly is None or len(poly) < 3:
            continue
        xs = list(poly[:, 0]) + [poly[0, 0]]
        ys = list(poly[:, 1]) + [poly[0, 1]]
        ax.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, axes

def visualize_brt_over_time(grid, times, all_brt_values, geom, 
                            target_center, target_radius,
                            theta_deg=0, time_indices=None):
    """
    Visualize BRT at multiple time snapshots.
    """
    if time_indices is None:
        # Default: show 6 snapshots
        time_indices = np.linspace(0, len(times)-1, 6, dtype=int)
    
    theta_idx = int(theta_deg / 360 * grid.shape[2]) % grid.shape[2]
    
    n_plots = len(time_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for ax, t_idx in zip(axes, time_indices):
        t_idx = int(t_idx)
        
        ax.contourf(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            all_brt_values[t_idx, :, :, theta_idx].T,
            levels=50, cmap='RdBu_r'
        )
        
        ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            all_brt_values[t_idx, :, :, theta_idx].T,
            levels=[0], colors='black', linewidths=2
        )
        
        # Draw target
        circle = plt.Circle(target_center, target_radius,
                           fill=False, color='green', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        # Draw walls
        for seg in geom['wall_segments']:
            (x0, z0), (x1, z1) = seg
            ax.plot([x0, x1], [z0, z1], 'k-', linewidth=1.5, alpha=0.5)

        # Draw object outlines
        for poly in geom['object_polys']:
            if poly is None or len(poly) < 3:
                continue
            xs = list(poly[:, 0]) + [poly[0, 0]]
            ys = list(poly[:, 1]) + [poly[0, 1]]
            ax.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.5)
        
        ax.set_title(f't = {times[t_idx]:.2f}s')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    # Hide unused axes
    for ax in axes[len(time_indices):]:
        ax.set_visible(False)
    
    plt.suptitle(f'BRT Evolution (θ = {theta_deg}°)', fontsize=14)
    plt.tight_layout()
    return fig, axes


def get_brt_value(grid, brt_values, state):
    """
    Get interpolated BRT value at a specific state.
    
    Args:
        grid: hj_reachability Grid object
        brt_values: value function array from hj.step
        state: [x, y, theta] where theta is in DEGREES
    
    Returns:
        Interpolated value (negative = inside BRT, positive = outside)
    """
    state = jnp.asarray(state)
    
    # Convert theta from degrees to radians
    state = state.at[2].set(jnp.deg2rad(state[2]))
    
    indices = []
    for i in range(len(state)):
        coord_vec = grid.coordinate_vectors[i]
        lo, hi = coord_vec[0], coord_vec[-1]
        n = len(coord_vec)
        
        if i == 2:  # theta is periodic
            s = state[i] % (2 * jnp.pi)
        else:
            s = state[i]
        
        idx = (s - lo) / (hi - lo) * (n - 1)
        indices.append(idx)
    
    indices = jnp.array(indices)
    
    value = jsp.ndimage.map_coordinates(
        brt_values, 
        indices.reshape(-1, 1), 
        order=1,
        mode='wrap'
    )
    
    return float(value[0])

def get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go):
    """
    Get interpolated BRT value at a specific state and time-to-go.
    
    The BRT shrinks as time_to_go decreases (approaching the target time).
    
    Args:
        grid: hj_reachability Grid object
        all_brt_values: (n_times, nx, ny, ntheta) array from hj.solve
        times: 1D array of time points (0 to -time_horizon)
        state: [x, y, theta] where theta is in DEGREES
        time_to_go: remaining time to reach target (0 = must be at target now,
                    time_horizon = full BRT available)
    
    Returns:
        Interpolated value (negative = inside BRT, positive = outside)
    """
    state = jnp.asarray(state)
    state = state.at[2].set(jnp.deg2rad(state[2]))
    
    times = np.asarray(times)
    
    # times = [0, -0.2, -0.4, ..., -20] (decreasing)
    # time_to_go = 20 should map to index len(times)-1 (t=-20, largest BRT)
    # time_to_go = 0 should map to index 0 (t=0, just target)
    #
    # Flip arrays for np.interp (needs increasing x)
    times_flipped = times[::-1]  # [-20, ..., -0.2, 0]
    indices_flipped = np.arange(len(times))[::-1]  # [100, ..., 1, 0]
    
    query_t = -time_to_go  # time_to_go=20 -> query_t=-20
    time_idx = np.interp(query_t, times_flipped, indices_flipped)
    
    # Spatial indices
    indices = [time_idx]
    for i in range(3):
        coord_vec = grid.coordinate_vectors[i]
        lo, hi = coord_vec[0], coord_vec[-1]
        n = len(coord_vec)
        
        if i == 2:  # theta periodic
            s = state[i] % (2 * jnp.pi)
        else:
            s = state[i]
        
        idx = (s - lo) / (hi - lo) * (n - 1)
        indices.append(idx)
    
    indices = jnp.array(indices).reshape(-1, 1)
    
    value = jsp.ndimage.map_coordinates(
        all_brt_values,
        indices,
        order=1,
        mode='wrap'
    )
    
    return float(value[0])


def get_brt_values_batch(grid, brt_values, states):
    """
    Get interpolated BRT values for multiple states.
    
    Args:
        grid: hj_reachability Grid object
        brt_values: value function array
        states: (N, 3) array of [x, y, theta] where theta is in DEGREES
    
    Returns:
        (N,) array of interpolated values
    """
    states = jnp.asarray(states)
    
    # Convert theta column from degrees to radians
    states = states.at[:, 2].set(jnp.deg2rad(states[:, 2]))
    
    indices = []
    for i in range(3):
        coord_vec = grid.coordinate_vectors[i]
        lo, hi = coord_vec[0], coord_vec[-1]
        n = len(coord_vec)
        
        if i == 2:
            s = states[:, i] % (2 * jnp.pi)
        else:
            s = states[:, i]
        
        idx = (s - lo) / (hi - lo) * (n - 1)
        indices.append(idx)
    
    indices = jnp.stack(indices, axis=0)
    
    values = jsp.ndimage.map_coordinates(
        brt_values,
        indices,
        order=1,
        mode='wrap'
    )
    
    return values



# =============================================================================
# Unicycle dynamics (same as before)
# =============================================================================

class Unicycle(hj.ControlAndDisturbanceAffineDynamics):
    """Unicycle: ẋ = v*cos(θ), ẏ = v*sin(θ), θ̇ = ω"""

    def __init__(self,
                 max_v=1.0,
                 max_omega=1.0,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        
        if control_space is None:
            control_space = hj.sets.Box(
                jnp.array([-max_v, -max_omega]),
                jnp.array([max_v, max_omega])
            )
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), 0.0)
        
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.array([0.0, 0.0, 0.0])

    def control_jacobian(self, state, time):
        _, _, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],
            [jnp.sin(theta), 0.0],
            [0.0, 1.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Choose a house
    house_index = 152
    
    #visualize the house layout:
    #fig, ax, geom = create_environment(house_index, render=True)
    # plt.title(f"House {house_index} - Pick a target location")
    # plt.show()
    
    # Set target based on house layout
    target_center = (7.0, 5.5)  # Adjust based on your house
    target_radius = 0.5

    ######################################
    # SINGLE TIME STEP BRT (for a specific target time)
    ######################################

    # # Create dynamics
    # dynamics = Unicycle(max_v=0.2, max_omega=1.0)
    
    # # Compute BRT
    # print("Computing BRT...")
    # grid, target_values, obstacle_values, brt_values, geom = compute_house_brt(
    #     dynamics=dynamics,
    #     house_index=house_index,
    #     target_center=target_center,
    #     target_radius=target_radius,
    #     time_horizon=20.0,
    #     robot_radius=0.2,
    #     wall_thickness=0.1
    # )
    # print("Done!")
    

    # #print("BRT values:", brt_values)
    
    # # Visualize at different heading angles
    # for theta_idx in [0, 9, 18, 27]:
    #     fig, axes = visualize_house_brt(
    #         grid, target_values, obstacle_values, brt_values, geom,
    #         target_center, target_radius, theta_idx=theta_idx
    #     )
    #     plt.show()



    # state = [7.0, 5.0, 0.0]  # [x, y, theta]
    # value = get_brt_value(grid, brt_values, state)

    # print(f"State: {state}")
    # print(f"BRT value: {value:.4f}")
    # print(f"Inside BRT (can reach target safely): {value <= 0}")

    ######################################
    # BRT OVER TIME (multiple snapshots up to a target time)
    ######################################

    # Create dynamics
    dynamics = Unicycle(max_v=0.2, max_omega=1.0)

    # Compute BRTs up to t=20 with 101 time snapshots
    grid, times, target_values, obstacle_values, all_brt_values, geom = compute_house_brt_over_time(
        dynamics=dynamics,
        house_index=house_index,
        target_center=target_center,
        target_radius=target_radius,
        time_horizon=20.0,
        n_time_steps=21, 
        robot_radius=0.2,
        wall_thickness=0.1
    )

    print(f"Times shape: {times.shape}")           
    print(f"BRT values shape: {all_brt_values.shape}")  
    print(f"Time range: [{times[0]:.1f}, {times[-1]:.1f}]")  # [0.0, -20.0]

    fig, axes = visualize_brt_over_time(
    grid, times, all_brt_values, geom,
    target_center=target_center,
    target_radius=target_radius,
    theta_deg=0,
    time_indices=[0, 5, 10, 20]  # specific snapshots
    )
    plt.show()

    state = [6, 2, 0]
    
    value_start = get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go=20.0)

    # Halfway through (10 seconds to go): smaller BRT
    value_mid = get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go=10.0)

    # Near end (2 seconds to go): even smaller BRT
    value_end = get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go=2.0)

    # At target time (0 seconds to go): just the target set
    value_final = get_brt_value_at_time(grid, all_brt_values, times, state, time_to_go=0.0)

    print(f"time_to_go=20: {value_start:.4f}")  # Largest BRT
    print(f"time_to_go=10: {value_mid:.4f}")    # Shrinking
    print(f"time_to_go=2:  {value_end:.4f}")    # Smaller
    print(f"time_to_go=0:  {value_final:.4f}")  # Just target

