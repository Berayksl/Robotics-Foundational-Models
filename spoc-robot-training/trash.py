from RL.src.networks import QNetwork
from RL.src.main import select_model_file, normalize_state
from RL.src.simulator import Continuous2DEnv
from RL.src.dynamics import DiscreteUnicycleDynamics
from RL.src.geometry import _swept_circle_clipped_translation
import numpy as np
import torch
import os


def fast_step_discrete_with_collision(env, state, action):
    x0, y0, th0 = float(state[0]), float(state[1]), float(state[2])

    # compute intended x1,y1,th1 (same as before)
    old_x, old_y, old_th = env.agent.x, env.agent.y, env.agent.theta
    env.agent.x, env.agent.y, env.agent.theta = x0, y0, th0
    x1, y1, th1 = env.apply_action_discrete_unicycle(action)
    env.agent.x, env.agent.y, env.agent.theta = old_x, old_y, old_th

    if action not in ["m", "b"]:
        return np.array([x0, y0, th1], dtype=np.float32), False

    r = 0.0 if env.agent_as_point else float(env.robot_R)

    # Query only nearby geometry using path AABB inflated by r
    xmin = min(x0, x1) - r; xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r; ymax = max(y0, y1) + r
    wall_ids, poly_ids = env._shash.query(xmin, ymin, xmax, ymax)

    # Build reduced lists
    wall_segments = [env.wall_segments[i] for i in wall_ids]
    object_polys   = [env.object_polys[i]   for i in poly_ids]

    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)

    p_clip, clipped = _swept_circle_clipped_translation(
        p0, p1, r,
        object_polys,
        wall_segments,
        step=0.02,
        eps=1e-4
    )

    return np.array([float(p_clip[0]), float(p_clip[1]), float(th1)], dtype=np.float32), bool(clipped)


goals = {0: {'center': (5, 7.6), 'radius': 0.4, 'movement':{'type':'static'}}, #goal region for the agentssss
1: {'center': (2.5, 1), 'radius': 0.4, 'movement':{'type':'static'}}}

STL_tasks =  [{"goal_id": 0, "spec": dict(operator="F", a=0,  b=70, t_star=60, gamma_inf=-0.1, collision_penalty=0.0)},
    {"goal_id": 1, "spec": dict(operator="F", a=70, b=150, t_star=140, gamma_inf=-0.1, collision_penalty=0.0)}]

STL_horizon = 150 #CHANGE LATER!
targets = {}
#config dictionary for the environment
config = {
    'house_index': 9,
    'init_loc':[1, 6.5, 90.0], #initial location of the agent (x, y)
    "dt": 1,
    "render": False,
    'dt_render': 0.01,
    'goals': goals, #goal regions for the agent
    "obstacle_location": [300.0, 300.0],
    "obstacle_size": 0.0,
    "randomize_loc": False, #whether to randomize the agent location at the end of each episode
    'deterministic': False,
    "dynamics": "discrete unicycle", #dynamics model to use
    "targets": targets,
    "disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
    "agent_as_point": False,
    "tasks": STL_tasks,
    "episode_len": STL_horizon,
}

env = Continuous2DEnv(config)


initial_state = np.array([4.917691230773926, 8.299999237060547, 240.0])


actions_to_execute = ['b'] * 50

s = initial_state

for action in actions_to_execute:
    s, _ = fast_step_discrete_with_collision(env, s, action)
    print(s)