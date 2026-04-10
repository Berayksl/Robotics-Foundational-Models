import numpy as np
from typing import Iterable, Tuple, Union

def calculate_goal_predicate_robustness(trajectory, goal_center, goal_radius):
    """
    Calculate the robustness of the whole trajectory with respect to reaching a circular goal region. #FIXME: currently only for eventually operator

    Args:
        current_state: tuple (x, y, theta) representing the current position and orientation
        goal_center: tuple (x_goal, y_goal) representing the goal position

    Returns:
        robustness: float value indicating how close the current state is to the goal center
    """
   
    distances = [np.linalg.norm(np.array(state[:2]) - np.array(goal_center)) for state in trajectory]

    radius = goal_radius # meters (radius of the goal region    )

    # # Define robustness as the negative distance to the goal center
    # robustness = np.max(radius - np.array(distances))

    return radius - np.array(distances)


def predicate_square_avoid_rho_over_trajectory(
    env,
    state_trajectory: Union[np.ndarray, Iterable[Tuple[float, float, float]]],
    obstacle_center: Tuple[float, float],
    obstacle_half_extents: Tuple[float, float]) -> np.ndarray:
    """
    Takes a trajectory of states (x,y,theta) and returns o[t] = square_avoid_rho((x,y), ...)
    for every state in the trajectory
    Returns:
        o: np.ndarray of shape (T,) where positive means outside, negative means inside.
    """

    traj = np.asarray(state_trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(
            f"state_trajectory must be (T,3) or (T,>=2). Got shape={traj.shape}"
        )

    T = traj.shape[0]
    o = np.empty(T, dtype=float)

    for i in range(T):
        pos_xy = traj[i, :2]
        o[i] = env._square_avoid_rho(pos_xy, obstacle_center, obstacle_half_extents)

    return o

