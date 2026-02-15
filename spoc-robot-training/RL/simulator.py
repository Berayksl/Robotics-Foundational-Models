import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque

from matplotlib.patches import RegularPolygon
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle

try:
    from geometry import _static_circle_collision, _swept_circle_collision, _compute_env_bounds_from_geom
    from dynamics import UnicycleDynamics, SingleIntegratorDynamics, DiscreteUnicycleDynamics
    from funnel_reward import FunnelReward, FunnelSpec, ModifiedFunnelReward
    from occupancy_map import create_environment
except ImportError:
    from .geometry import _static_circle_collision, _swept_circle_collision, _compute_env_bounds_from_geom
    from .dynamics import UnicycleDynamics, SingleIntegratorDynamics, DiscreteUnicycleDynamics
    from .funnel_reward import FunnelReward, FunnelSpec, ModifiedFunnelReward
    from .occupancy_map import create_environment

#from Sequential_CBF import sequential_CBF

class Continuous2DEnv:
    def __init__(self, config):
        """
        Initializes the environment with given configuration.
        :param config: Dictionary containing environment parameters.
        """
        self.house_index = config.get("house_index", 30)
        self.dt = config.get("dt", 0.1)
        self.render = config.get("render", False)
        self.show_timer = config.get("show_timer", True)
        self.dt_render = config.get("dt_render", 0.001)
        self.goals = config.get("goals", None)  # dictionary of goals for the agent
        self.obstacles = config.get("obstacles", None)  # dictionary of obstacles
        self.random_loc = config.get("randomize_loc", True)
        self.targets = config.get("targets", None)  # dictionary of targets for the CBF
        self.init_loc = np.array(config.get("init_loc", [5.0, 5.0]))
        self.action_max = config.get("action_max")
        self.action_min = config.get("action_min")
        self.dynamics = config.get("dynamics", "unicycle")
        #self.u_agent_max = config.get("u_agent_max", None)  # max agent speed
        self.agent_as_point = bool(config.get("agent_as_point", False)) #whether the agent should be consider as a point mass or not

        if config.get("disturbance") is not None: #min and max disturbance:
            self.disturbance = True
            self.w_min = config.get("disturbance")[0] 
            self.w_max = config.get("disturbance")[1]
        else:
            self.disturbance = False

        self.initial_target_centers = {target_index: self.targets[target_index]['center'] for target_index in self.targets.keys()}
        self.simulation_timer = 0 # time steps elapsed (doesn't reset on env.reset())
        self.episode_timer = 0  # time steps in current episode (resets on env.reset())

        if self.dynamics == "unicycle":
            self.observation_space = np.zeros((3,))
            self.action_space = np.zeros((2,))
            
        elif self.dynamics == "single integrator":
            self.observation_space = np.zeros((2,))
            self.action_space = np.zeros((2,))

        elif self.dynamics == "discrete unicycle":
            self.observation_space = np.zeros((3,)) #x, y, theta
            self.action_space = ['m', 'b', 'l', 'r', 'ls', 'rs']  # discrete actions

        #self.precompute_cbf_values(resolution=100)  # Precompute CBF values for visualization

        self.robot_R = 0.18  # visual radius (tune)

        print("Initializing environment...")

        if self.render:
            self.fig, self.ax, geom = create_environment(
                                            self.house_index,
                                            show_room_outline=False,
                                            render = True
                                        )
            
            self.x_min, self.x_max, self.y_min, self.y_max = _compute_env_bounds_from_geom(geom)

            self.width  = self.x_max - self.x_min
            self.height = self.y_max - self.y_min

            #for collusion checking later:    
            self.object_polys = geom["object_polys"]
            self.wall_segments = geom["wall_segments"]

            
            self.reset() #start the simulation

            # --- set window location/size (pixels), backend-safe ---
            mng = plt.get_current_fig_manager()
            try:
                # TkAgg backend
                mng.window.wm_geometry("900x650+120+80")  # WxH+X+Y
            except Exception:
                try:
                    # Qt backend
                    # setGeometry(x, y, width, height)
                    mng.window.setGeometry(1200, 80, 1500, 1500)
                except Exception:
                    pass  # other backends may not support moving the window
 
            if self.dynamics in ["unicycle", "discrete unicycle"]:
                #self.robot_R = 0.18  # visual radius (tune)

                # robot body
                self.agent_body = Circle((self.agent.x, self.agent.y),
                                        radius=self.robot_R,
                                        facecolor="red",
                                        edgecolor="black",
                                        zorder=20)
                self.ax.add_patch(self.agent_body)

                # heading indicator (a short line from center to rim)
                (self.agent_heading_line,) = self.ax.plot([], [], 'k-', linewidth=2, zorder=21)

            else:
                self.agent_plot, = self.ax.plot([], [], 'ro', markersize=10, zorder=20)

            self.goal_plots = []
            for goal in self.goals.values():
                goal_plot = plt.Circle(goal['center'], goal['radius'], color='g', alpha=0.5, label='Goal')
                self.goal_plots.append(goal_plot)
                self.ax.add_patch(goal_plot)

            if self.show_timer:
                self.timer_text = self.ax.text(
                    0.02, 0.98, f"Time:{self.simulation_timer%301}",
                    transform=self.ax.transAxes,
                    ha='left', va='top', fontsize=25,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', alpha=0.75),
                    zorder=10
                )

            #self.target_region_plot = plt.Circle(self.target_region_center, self.target_region_radius, color='b', fill=False, linestyle='-', label='Target Region')
            #self.ax.add_patch(self.target_region_plot)
            self.safe_region_plots = []
            self.target_region_patches = []
            self.obstacle_region_patches = []
            # self.ax.legend()
            #remove the axis ticks:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            #self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=10, cache_frame_data=False)
            self.fig.show()
            self.fig.canvas.draw()

        else:
            geom = create_environment(self.house_index,
                                      show_room_outline=False,
                                      render = False)
            
            self.x_min, self.x_max, self.y_min, self.y_max = _compute_env_bounds_from_geom(geom)

            self.width  = self.x_max - self.x_min
            self.height = self.y_max - self.y_min

            #for collusion checking later:    
            self.object_polys = geom["object_polys"]
            self.wall_segments = geom["wall_segments"]

            self.reset() #start the simulation



        print(f"Environment x bounds: [{self.x_min}, {self.x_max}]")
        print(f"Environment y bounds: [{self.y_min}, {self.y_max}]")

        print(f"Environment width: {self.width}, height: {self.height}")

        #STL task for reward computation:
        goal_id = 0
        spec = FunnelSpec(
            operator="F",
            a=0,
            b=60,
            t_star=50,
            gamma_inf= -0.1,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            collision_penalty=0.0,
        )

        g = self.goals[goal_id]
        self.reward_fn = FunnelReward(g["center"], g["radius"], spec)
        #self.reward_fn = ModifiedFunnelReward(g["center"], g["radius"], spec)
        self.reward_goal_id = goal_id

        print("Environment initialized.")


    def _precompute_free_spawn_cells(self, resolution=0.10, seed_xy=None, max_cells=100_000):
        """
        Precompute valid spawn locations:
        - collision-free for a circle robot
        - inside the house (connected component of free space)
        Returns a list of (x, y) spawn points sampled from a grid.

        resolution: meters per cell (0.05–0.10 recommended)
        seed_xy: BFS seed point inside the house (defaults to init_loc or a goal center)
        """

        # Bounds: prefer true house bounds if you computed them, else fallback to width/height convention
        if hasattr(self, "x_min") and hasattr(self, "x_max") and hasattr(self, "y_min") and hasattr(self, "y_max"):
            x_min, x_max = float(self.x_min), float(self.x_max)
            y_min, y_max = float(self.y_min), float(self.y_max)
        else:
            x_min, x_max = -float(self.width), float(self.width)
            y_min, y_max = -float(self.height), float(self.height)

        # Expand a bit to be safe (optional)
        pad = 0.0
        x_min -= pad; x_max += pad
        y_min -= pad; y_max += pad

        # Build grid
        xs = np.arange(x_min, x_max + resolution, resolution, dtype=np.float32)
        ys = np.arange(y_min, y_max + resolution, resolution, dtype=np.float32)

        # Safety to avoid insane grids
        if xs.size * ys.size > max_cells:
            raise RuntimeError(
                f"Grid too large: {xs.size} x {ys.size}. "
                f"Increase resolution (e.g., 0.15) or tighten bounds."
            )

        # Mark collision-free cells
        free = np.zeros((ys.size, xs.size), dtype=bool)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                free[j, i] = not self.collides_at(x, y, use_swept=False)

        # Choose a seed that is definitely inside the house.
        if seed_xy is None:
            # Prefer init_loc if it exists and is in free space, else use first goal center.
            cand = (float(self.init_loc[0]), float(self.init_loc[1]))
            if not self.collides_at(cand[0], cand[1], use_swept=False):
                seed_xy = cand
            else:
                # fallback: first goal center (may still collide if goal is near wall)
                g0 = next(iter(self.goals.values()))
                seed_xy = (float(g0["center"][0]), float(g0["center"][1]))

        # Convert seed to nearest grid cell
        si = int(np.argmin(np.abs(xs - seed_xy[0])))
        sj = int(np.argmin(np.abs(ys - seed_xy[1])))

        if not free[sj, si]:
            # Find nearest free cell around seed (small local search)
            found = False
            R = 10
            for dj in range(-R, R + 1):
                for di in range(-R, R + 1):
                    jj = np.clip(sj + dj, 0, ys.size - 1)
                    ii = np.clip(si + di, 0, xs.size - 1)
                    if free[jj, ii]:
                        sj, si = jj, ii
                        found = True
                        break
                if found:
                    break
            if not found:
                raise RuntimeError("Could not find any free seed cell for flood-fill. Check bounds / robot_R.")

        # Flood-fill connected free-space component (inside the house)
        inside = np.zeros_like(free, dtype=bool)
        q = deque()
        q.append((sj, si))
        inside[sj, si] = True

        # 4-neighborhood is fine; 8-neighborhood is slightly more permissive
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            j, i = q.popleft()
            for dj, di in nbrs:
                jj = j + dj
                ii = i + di
                if jj < 0 or jj >= ys.size or ii < 0 or ii >= xs.size:
                    continue
                if inside[jj, ii]:
                    continue
                if not free[jj, ii]:
                    continue
                inside[jj, ii] = True
                q.append((jj, ii))

        # Store all valid cells as spawn candidates
        spawn_js, spawn_is = np.where(inside)
        self._spawn_xs = xs
        self._spawn_ys = ys
        self._spawn_cells = list(zip(spawn_is.tolist(), spawn_js.tolist()))  # (i,j) pairs
        self._spawn_resolution = float(resolution)

        if len(self._spawn_cells) == 0:
            raise RuntimeError("No valid spawn cells found. Check robot_R, bounds, or collision geometry.")
        

    # def reset(self):
    #     """Resets the environment. If random_loc=True, sample a collision-free start (with robot size)."""
    #     self.episode_timer = 0

    #     # --- bounds: prefer house-derived bounds if available ---
    #     if hasattr(self, "x_min") and hasattr(self, "x_max") and hasattr(self, "y_min") and hasattr(self, "y_max"):
    #         x_lo, x_hi = float(self.x_min), float(self.x_max)
    #         y_lo, y_hi = float(self.y_min), float(self.y_max)
    #     else:
    #         # fallback to your previous convention
    #         x_lo, x_hi = -float(self.width), float(self.width)
    #         y_lo, y_hi = -float(self.height), float(self.height)

    #     # --- robot radius for collision-aware sampling ---
    #     # If you only define robot_R when render=True, ensure a default exists:
    #     if not hasattr(self, "robot_R"):
    #         self.robot_R = 0.18  # same as your render radius

    #     # --- helper: check if too close to any goal (including robot radius) ---
    #     def too_close_to_goals(x, y, buffer=2.0):
    #         p = np.array([x, y], dtype=float)
    #         for goal in self.goals.values():
    #             c = np.array(goal["center"], dtype=float)
    #             R = float(goal["radius"]) + float(self.robot_R) + float(buffer)
    #             if np.linalg.norm(p - c) <= R:
    #                 return True
    #         return False

    #     # --- helper: free-space check using your collision checker ---
    #     def in_free_space(x, y):
    #         # collides_at returns True if collision
    #         return not self.collides_at(x, y, use_swept=False)

    #     if self.random_loc:
    #         max_tries = 5000
    #         buffer = 2.0  # your previous "goal buffer"; keep or make configurable

    #         # sample until valid
    #         for _ in range(max_tries):
    #             x = np.random.uniform(x_lo, x_hi)
    #             y = np.random.uniform(y_lo, y_hi)

    #             # # 1) not too close to goals
    #             # if too_close_to_goals(x, y, buffer=buffer):
    #             #     continue

    #             # 2) collision-free given robot radius
    #             if not in_free_space(x, y):
    #                 continue

    #             # Found valid pose
    #             if self.dynamics == "unicycle":
    #                 theta = float(self.init_loc[2] + np.random.uniform(-np.pi, np.pi))
    #                 self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
    #                 return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

    #             elif self.dynamics == "single integrator":
    #                 self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
    #                 return np.array([self.agent.x, self.agent.y], dtype=float)

    #             elif self.dynamics == "discrete unicycle":
    #                 theta_deg = float(np.rad2deg(np.random.uniform(-np.pi, np.pi)))
    #                 self.agent = DiscreteUnicycleDynamics(x=x, y=y, theta=theta_deg)
    #                 return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

    #             else:
    #                 raise ValueError(f"Unknown dynamics: {self.dynamics}")

    #         # If we get here, we failed to sample
    #         raise RuntimeError(
    #             f"reset(): Failed to sample collision-free start after {max_tries} tries. "
    #             f"Map may be too cluttered or bounds may be wrong."
    #         )

    #     # --- non-random reset (unchanged except bounds are not applied) ---
    #     if self.dynamics == "unicycle":
    #         x, y, theta = float(self.init_loc[0]), float(self.init_loc[1]), float(self.init_loc[2])
    #         self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
    #         return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

    #     elif self.dynamics == "single integrator":
    #         x, y = float(self.init_loc[0]), float(self.init_loc[1])
    #         self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
    #         return np.array([self.agent.x, self.agent.y], dtype=float)

    #     elif self.dynamics == "discrete unicycle":
    #         x, y, theta = float(self.init_loc[0]), float(self.init_loc[1]), float(self.init_loc[2])
    #         self.agent = DiscreteUnicycleDynamics(x=x, y=y, theta=theta)
    #         return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

    #     else:
    #         raise ValueError(f"Unknown dynamics: {self.dynamics}")


    def reset(self):
        """Resets the environment."""
        self.episode_timer = 0
        self.task_satisfied = False

        if self.random_loc:
            # Make sure spawn cells are available
            if not hasattr(self, "_spawn_cells") or len(self._spawn_cells) == 0:
                # fallback: build them now (but better to do in __init__)
                self._precompute_free_spawn_cells(resolution=0.30, seed_xy=(self.init_loc[0], self.init_loc[1]))

            # sample a free cell uniformly
            i, j = self._spawn_cells[np.random.randint(len(self._spawn_cells))]
            x = float(self._spawn_xs[i])
            y = float(self._spawn_ys[j])

            # OPTIONAL: jitter within the cell and re-check collision
            # (keeps distribution less "grid-like")
            jitter = 0.45 * self._spawn_resolution
            for _ in range(10):
                xj = x + np.random.uniform(-jitter, jitter)
                yj = y + np.random.uniform(-jitter, jitter)
                if not self.collides_at(xj, yj, use_swept=False):
                    x, y = xj, yj
                    break

            # OPTIONAL: avoid being inside/too-close to goals (including robot size)
            buffer = 1.0
            for goal in self.goals.values():
                c = np.array(goal["center"], dtype=float)
                extra = 0.0 if self.agent_as_point else float(self.robot_R)
                R = float(goal["radius"]) + extra + buffer
                #R = float(goal["radius"]) + float(self.robot_R) + buffer
                if np.linalg.norm(np.array([x, y]) - c) <= R:
                    # if too close, just resample once more (simple)
                    i, j = self._spawn_cells[np.random.randint(len(self._spawn_cells))]
                    x = float(self._spawn_xs[i]); y = float(self._spawn_ys[j])
                    break

            # construct agent
            if self.dynamics == "unicycle":
                theta = float(self.init_loc[2] + np.random.uniform(-np.pi, np.pi))
                self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
                return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

            elif self.dynamics == "single integrator":
                self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
                return np.array([self.agent.x, self.agent.y], dtype=float)

            elif self.dynamics == "discrete unicycle":
                theta_deg = float(np.rad2deg(np.random.uniform(-np.pi, np.pi)))
                self.agent = DiscreteUnicycleDynamics(x=x, y=y, theta=theta_deg)
                return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

            else:
                raise ValueError(f"Unknown dynamics: {self.dynamics}")
            
        else:
            # --- non-random reset (unchanged except bounds are not applied) ---
            if self.dynamics == "unicycle":
                x, y, theta = float(self.init_loc[0]), float(self.init_loc[1]), float(self.init_loc[2])
                self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
                return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

            elif self.dynamics == "single integrator":
                x, y = float(self.init_loc[0]), float(self.init_loc[1])
                self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
                return np.array([self.agent.x, self.agent.y], dtype=float)

            elif self.dynamics == "discrete unicycle":
                x, y, theta = float(self.init_loc[0]), float(self.init_loc[1]), float(self.init_loc[2])
                self.agent = DiscreteUnicycleDynamics(x=x, y=y, theta=theta)
                return np.array([self.agent.x, self.agent.y, self.agent.theta], dtype=float)

            else:
                raise ValueError(f"Unknown dynamics: {self.dynamics}")



    def compute_reward(self): #updated to include multiple goals
        pos = np.array([self.agent.x, self.agent.y])
        # List of distances to all goal centers
        dists = [np.linalg.norm(pos - np.array(g['center'])) for g in self.goals.values()]
        dist_min = min(dists)
        
        reward = -dist_min  # Dense part: encourages reaching nearest goal
        
        # Sparse bonus if agent enters any goal region
        for goal in self.goals.values():
            if dist_min <= goal['radius']:
                reward += 100
                break
        
        return reward
    
    def collides_at(self, x, y, use_swept=False, prev_xy=None):
        """
        Check if agent circle at (x,y) collides with objects or walls.
        If use_swept=True, you must pass prev_xy=(x_prev,y_prev) to check along the path.
        """
        p = np.array([x, y], dtype=np.float32)
        r = 0.0 if self.agent_as_point else float(self.robot_R)

        if use_swept:
            assert prev_xy is not None, "prev_xy must be provided for swept collision"
            p0 = np.array(prev_xy, dtype=np.float32)
            p1 = p
            return _swept_circle_collision(p0, p1, r, self.object_polys, self.wall_segments, step=0.02)

        return _static_circle_collision(p, r, self.object_polys, self.wall_segments)
    

    def apply_action_discrete_unicycle(self, action):
        """
        Matches your DiscreteUnicycleDynamics.update() exactly.
        theta_deg is your stored convention (not math angle).
        Returns (x1, y1, theta1_deg).
        """
        x, y, theta_deg = float(self.agent.x), float(self.agent.y), float(self.agent.theta)

        forward_dist = 0.2
        back_dist = -0.2
        rot_large = np.deg2rad(30)
        rot_small = np.deg2rad(6)

        v = 0.0
        omega = 0.0

        if action == "m":
            v = forward_dist
        elif action == "b":
            v = back_dist
        elif action == "l":
            omega = rot_large
        elif action == "r":
            omega = -rot_large
        elif action == "ls":
            omega = rot_small
        elif action == "rs":
            omega = -rot_small
        elif action == "stay":
            v = 0.0
            omega = 0.0
        else:
            raise ValueError(f"Unknown action {action}")

        theta_rad = np.deg2rad(90 - theta_deg)

        x1 = x + v * np.cos(theta_rad)
        y1 = y + v * np.sin(theta_rad)

        theta_rad_next = theta_rad + omega
        theta1_deg = (90 - np.rad2deg(theta_rad_next)) % 360

        return x1, y1, theta1_deg
    
    def step_discrete_with_collision(self, action):
        """
        Use this instead of directly calling self.agent.update(action)
        for the discrete-unicycle case.
        """
        x0, y0, th0 = float(self.agent.x), float(self.agent.y), float(self.agent.theta)
        x1, y1, th1 = self.apply_action_discrete_unicycle(action)

        # Keep within bounds if you want (optional). If your walls are the true boundary,
        # you can skip clipping and rely on wall segments. If you DO clip, do it before collision:
        # x1 = np.clip(x1, -self.width, self.width)
        # y1 = np.clip(y1, -self.height, self.height)

        # rotations can't collide (unless you want "rotation in place" to collide if already intersecting)
        is_translation = action in ["m", "b"]

        if is_translation:
            # swept check prevents tunneling through thin walls/edges
            hit = self.collides_at(x1, y1, use_swept=True, prev_xy=(x0, y0))
        else:
            # for rotation-only actions, keep position same
            hit = self.collides_at(x0, y0, use_swept=False)

        if hit:
            # reject: do not move
            return np.array([x0, y0, th0]), True  # collided=True

        # commit
        self.agent.x, self.agent.y, self.agent.theta = x1, y1, th1
        return np.array([x1, y1, th1]), False
    
    def step(self, action):
        """
        Takes an action (linear and angular velocity) and updates the agent's position.
        :param action: A tuple (v, w) representing linear and angular velocity.
        :return: next_state, reward, done
        """
        #make the updates:
        if self.disturbance:
            w = np.random.uniform(self.w_min, self.w_max, size=2) #sample noise
            action = action + w #add noise to the action
            #print(action)

        #update the agent's state
        if self.dynamics == "discrete unicycle":
            state, collided = self.step_discrete_with_collision(action)

        else:
            state = self.agent.update(action) #update the agent state
            self.agent.x = np.clip(self.agent.x, -self.width, self.width)
            self.agent.y = np.clip(self.agent.y, -self.height, self.height)

        #update the target region states:
        if self.targets is not None:
            for target_index in self.targets.keys():
                x_new, y_new = self.dynamic_target(self.simulation_timer, target_index)
                self.targets[target_index]['center'] = (x_new, y_new)

        #update the obstacle states:
        if self.obstacles is not None:
            for obstacle_index in self.obstacles.keys():
                x_new, y_new = self.dynamic_obstacle(self.simulation_timer, obstacle_index)
                self.obstacles[obstacle_index]['center'] = (x_new, y_new)

        #update the goal states:
        for goal_id in self.goals.keys():
            x_new, y_new = self.dynamic_goal(self.simulation_timer, goal_id)
            self.goals[goal_id]['center'] = (x_new, y_new)

        # Compute reward
        goal = self.goals[self.reward_goal_id]
        reward = self.reward_fn(
            pos_xy=(self.agent.x, self.agent.y),
            t=self.episode_timer,
            collided=(collided if self.dynamics == "discrete unicycle" else False),
            goal_center=goal["center"],  
            goal_radius=goal["radius"],
        )

        # Calculate distance to the goals
        dist_to_goals = {goal_id: np.linalg.norm(np.array([self.agent.x, self.agent.y]) - np.array(goal['center'])) for goal_id, goal in self.goals.items()}
        # Check if any goal is reached
        done = any(dist <= goal['radius'] for dist, goal in zip(dist_to_goals.values(), self.goals.values()))

        #CHANGE LATER!!!!!
        if done and not self.task_satisfied:
            #print("Goal reached!")
            reward += 500  # Add a large bonus for reaching any goal
            self.task_satisfied = True  # Mark task as satisfied to prevent multiple bonuses   
        ########################################

        if self.render:
            self.update_plot()
    

        #increment timers
        self.simulation_timer += 1
        self.episode_timer += 1

        return state, reward, done
    
    
    def update_plot(self):
        time.sleep(self.dt_render)

        # Update agent position:
        if self.dynamics in ["unicycle", "discrete unicycle"]:
            x, y = self.agent.x, self.agent.y
            th = np.deg2rad(self.agent.theta)  # degrees -> radians

            # update body
            self.agent_body.center = (x, y)

            # heading direction: theta=0 -> +y, theta=90 -> +x
            dx = np.sin(th)
            dy = np.cos(th)

            # line from center to rim (or a bit outside)
            L = self.robot_R * 1
            x2 = x + L * dx
            y2 = y + L * dy
            self.agent_heading_line.set_data([x, x2], [y, y2])

        else:
            self.agent_plot.set_data([self.agent.x], [self.agent.y])


        if self.render and self.show_timer:
            self.timer_text.set_text(f"Time:{self.simulation_timer%301}")
        
        # Update targets' positions
        for patch in self.target_region_patches:
            patch.remove()  # remove old
            
        for label in getattr(self, "target_labels", []):
            label.remove()

        self.target_region_patches = []
        self.target_labels = []
        for i, target_info in self.targets.items():
            patch = plt.Circle(target_info["center"], target_info["radius"], color=target_info['color'], fill=False, linestyle='-')
            self.ax.add_patch(patch)
            self.target_region_patches.append(patch)

            cx, cy = target_info["center"]
            target_label = target_info['label']
            label = self.ax.text(
                cx, cy, str(target_label),
                ha='center', va='center',
                fontsize=10, color=target_info['color'],
                weight='bold'
            )
            self.target_labels.append(label)

        if self.obstacles is not None:
            # Update obstacles' positions
            for patch in self.obstacle_region_patches:
                patch.remove()

            self.obstacle_region_patches = []
            for obstacle_id, obstacle_info in self.obstacles.items():
                patch = plt.Circle(obstacle_info["center"], obstacle_info["radius"], color='black', alpha=1, label='Obstacle')
                self.ax.add_patch(patch)
                self.obstacle_region_patches.append(patch)


        # Update goals' positions
        for goal_plot in self.goal_plots:
            goal_plot.remove()

        self.goal_plots = []
        for goal_id, goal_info in self.goals.items():
            patch = plt.Circle(goal_info["center"], goal_info["radius"], color='g', alpha=0.5, label='Goal')
            self.ax.add_patch(patch)
            self.goal_plots.append(patch)

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def dynamic_target(self, current_t, target_id):
        if self.targets is not None:
            movement_type = self.targets[target_id]['movement']['type']
            #TODO: add more movement types

            if movement_type == 'circular':
                x0, y0 = self.initial_target_centers[target_id]
                u_target_max = self.targets[target_id]['u_max']
                x, y = self.targets[target_id]['center'] #current position
                xc, yc = self.targets[target_id]['movement']['center_of_rotation']

                if 'theta' not in self.targets[target_id]['movement']:
                    theta = np.arctan2(y0 - yc, x0 - xc) #initial angle
                else:
                    theta = self.targets[target_id]['movement']['theta']

                turning_radius = np.linalg.norm(np.array([x0 - xc, y0 - yc]))
                omega = u_target_max / turning_radius #angular velocity
                
                # Calculate the new angle after time_elapsed
                theta += omega * self.dt
                turning_radius = u_target_max / omega

                x_new = xc + turning_radius * np.cos(theta)
                y_new = yc + turning_radius * np.sin(theta)

                #stop if hit another target:
                for other_id, other_info in self.targets.items():
                    if other_id != target_id:
                        other_center = np.array(other_info['center'])
                        dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                        if dist < (self.targets[target_id]['radius'] + other_info['radius']):
                            x_new, y_new = self.targets[target_id]['center'] #stay at the current position
                            theta -= omega * self.dt #stay at the current angle

                # Update theta in targets dictionary
                self.targets[target_id]['movement']['theta'] = theta

            elif movement_type == 'straight':
                x0, y0 = self.initial_target_centers[target_id]
                heading_angle = self.targets[target_id]['movement']['heading_angle']
                u_target_max = self.targets[target_id]['u_max']

                x_new = x0 + np.cos(heading_angle) * u_target_max * current_t
                y_new = x0 + np.sin(heading_angle) * u_target_max * current_t

            elif movement_type == "static":
                #no movement, just return the current position
                x_new, y_new = self.targets[target_id]['center']

            elif movement_type == "periodic":
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.targets[target_id]['movement']['point1'])
                point2 = np.array(self.targets[target_id]['movement']['point2'])
                u_target_max = self.targets[target_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_target_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # #stop if hit another target:
                # for other_id, other_info in self.targets.items():
                #     if other_id != target_id:
                #         other_center = np.array(other_info['center'])
                #         dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                #         if dist < (self.targets[target_id]['radius'] + other_info['radius']):
                #             x_new, y_new = self.targets[target_id]['center'] #stay at the current position

                # Update heading angle in targets dictionary
                self.targets[target_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'random_walk':
                # 0) keep your random re-heading
                if self.simulation_timer % 50 == 0:
                    heading_angle = np.random.uniform(0, 2*np.pi)
                    self.targets[target_id]['movement']['heading_angle'] = heading_angle
                else:
                    heading_angle = self.targets[target_id]['movement']['heading_angle']

                # 1) base velocity
                speed = self.targets[target_id]['u_max'] * 0.5
                v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed

                # 2) super-light separation from neighbors
                AVOID_RADIUS = 2.0   # start with 2–3; tune
                AVOID_GAIN   = 0.8   # 0.5–1.0 is typical
                p = np.array(self.targets[target_id]['center'], dtype=float)
                r_self = self.targets[target_id]['radius']

                rep = np.zeros(2, float)
                for oid, o in self.targets.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], dtype=float)
                    ro = o['radius']
                    delta = p - po
                    d = np.linalg.norm(delta)
                    R = r_self + ro + AVOID_RADIUS
                    if d < R and d > 1e-6:
                        rep += (delta / d) * (R - d) / R   # push straight away

                # apply small push, keep speed ~constant
                v = v + AVOID_GAIN * rep
                n = np.linalg.norm(v)
                #if n > 1e-6:
                    #v = v / n * speed
                heading_angle = float(np.arctan2(v[1], v[0]))

                # 3) integrate
                p_new = p + v * self.dt

                # 4) simple wall bounce
                if p_new[0] < -self.width or p_new[0] > self.width:
                    heading_angle = np.pi - heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[0] = np.clip(p_new[0], -self.width, self.width)

                if p_new[1] < -self.height or p_new[1] > self.height:
                    heading_angle = -heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[1] = np.clip(p_new[1], -self.height, self.height)

                # 5) last-resort bounce if still overlapping
                for oid, o in self.targets.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], float)
                    R = r_self + o['radius']
                    if np.linalg.norm(p_new - po) < R:
                        heading_angle += np.pi  # flip 180°
                        v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                        p_new = p + v * self.dt
                        break

                # 6) commit
                x_new, y_new = p_new.tolist()
                self.targets[target_id]['movement']['heading_angle'] = heading_angle

        return (x_new, y_new)
    

    def dynamic_goal(self, current_t, goal_id):
        if self.goals is not None:
            movement_type = self.goals[goal_id]['movement']['type']
            if movement_type == 'static':
                # No movement, just return the current position
                x_new, y_new = self.goals[goal_id]['center']
            elif movement_type == 'periodic':
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                u_goal_max = self.goals[goal_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_goal_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # Update heading angle in goals dictionary
                self.goals[goal_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'blinking':
                # Blinking movement: switch between two points every blink_duration
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                blink_duration = self.goals[goal_id]['movement']['blink_duration']

                # Determine if we are in the first or second half of the blink cycle
                if (current_t // blink_duration) % 2 == 0:
                    x_new, y_new = point1[0], point1[1]
                else:
                    x_new, y_new = point2[0], point2[1]
                #print the state change
                #print(f"Goal {goal_id} moved to ({x_new}, {y_new}) at time {current_t}")

            else:
                raise NotImplementedError(f"Movement type '{movement_type}' for goal not implemented.")
        else:
            raise ValueError("No goals defined in the environment.")

        return (x_new, y_new)
    

    def set_agent_location(self, x, y, theta=None):
        """
        Sets the agent's location to the specified coordinates manually.
        :param x: x-coordinate of the agent.
        :param y: y-coordinate of the agent.
        :param theta: orientation angle (only for unicycle dynamics).
        """
        if self.dynamics == 'unicycle':
            self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
            return np.array([self.agent.x, self.agent.y, self.agent.theta])
        elif self.dynamics == 'single integrator':
            self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
            return np.array([self.agent.x, self.agent.y])
        
if __name__ == "__main__":
    goals = {0: {'center': (5.2, 2), 'radius': 0.3, 'movement':{'type':'static'}}, #goal region for the agent
	#1: {'center': (-50, 0), 'radius': 10, 'movement':{'type':'static'}}
    }

    targets = {}
    #config dictionary for the environment
    config = {
        'house_index': 30,
        'init_loc':[1.5, 3.0, 59.999996185302734], #initial location of the agent (x, y)
        "dt": 1,
        "render": True,
		'dt_render': 0.1,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "discrete unicycle", #dynamics model to use
		"targets": targets,
		"disturbance": None #disturbance range in both x and y directions [w_min, w_max]
    }

    env = Continuous2DEnv(config)
    

    #action = (1.0, 1)  #linear vel., angular vel.

    #actions = ["l", "l", "l", "m", "r", "m", "rs", "m", "b", 'stay',"m", "m", "l", "m", "r", "m", "rs", "m", "b", 'stay', 'stay', 'stay','stay', 'stay','stay']

    #actions = ["b", "b", "b", "b", "b", "b", "b", "b", "b", "b",'b', "b", "b", "b", "b", "b", "b", "b", "b", "b",'b','b', "b", "b", "b", "b", "b", "b", "b", "b", "b",'b']

    #actions = ["r"] * 50
    #actions = ['stay'] *5

    actions = ['m']

    for i in range(100):
        for action in actions:
            state, reward, done = env.step(action)
            print(state)

            if done:
                print(f"Episode finished after {i+1} timesteps")
                break

        #print("State:", state)
        # if done:
        #     break

        #state = env.reset()

