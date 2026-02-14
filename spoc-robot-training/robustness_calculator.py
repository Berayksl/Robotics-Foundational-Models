import numpy as np


def calculate_robustness(trajectory, goal_center, goal_radius):
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

    # Define robustness as the negative distance to the goal center
    robustness = np.max(radius - np.array(distances))

    print(robustness)

    return robustness


