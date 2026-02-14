import numpy as np

def unicycle_step(state, action, dt=0.1):
    """
    Convert SPOC discrete actions to low-level unicycle commands.
    
    Args:
        state: current state [x, y, theta] (theta in radians)
        action: string, one of 
                ['move_ahead', 'move_back', 'rotate_left', 'rotate_right', 'rotate_left_small', 'rotate_right_small']
        dt: simulation timestep (seconds)
    
    Returns:
        next_state: updated [x, y, theta] after applying action for dt
        v: linear velocity command
        omega: angular velocity command
    """

    x, y, theta_deg = state

    theta_rad = np.deg2rad(90 - theta_deg)

    # Action magnitudes
    forward_dist = 0.2       # meters
    back_dist = -0.2         # meters
    rot_large = np.deg2rad(30)  # radians
    rot_small = np.deg2rad(6)   # radians

    v = 0.0
    omega = 0.0

    if action == "m":          # move ahead
        v = forward_dist / dt
    elif action == "b":        # move back
        v = back_dist / dt
    elif action == "l":        # rotate left
        omega = rot_large / dt
    elif action == "r":        # rotate right
        omega = -rot_large / dt
    elif action == "ls":       # rotate left small
        omega = rot_small / dt
    elif action == "rs":       # rotate right small
        omega = -rot_small / dt
    else:
        raise ValueError(f"Unknown action {action}")

    # Unicycle kinematics update
    x_next = x + v * np.cos(theta_rad) * dt
    y_next = y + v * np.sin(theta_rad) * dt
    theta_rad_next = theta_rad + omega * dt

    # Convert back to THOR degrees
    theta_deg_next = 90 - np.rad2deg(theta_rad_next)
    theta_deg_next = theta_deg_next % 360  # normalize to [0,360)


    next_state = [x_next, y_next, theta_deg_next]
    return next_state, v, omega



if __name__ == "__main__":
    # Example usage
    state = [0.0, 0.0, 0.0]  # Initial position (x=0, y=0) and orientation (theta=0)
    actions = ["m", "m", "l", "m", "r", "m", "rs", "m", "b"]

    for action in actions:
        next_state, v, omega = unicycle_step(state, action)
        print(f"Action: {action}, Next State: {next_state}, v: {v:.2f}, omega: {omega:.2f}")
        state = next_state