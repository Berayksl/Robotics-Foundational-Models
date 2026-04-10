import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

def kl_project_action_distribution_with_robustness(
    *,
    env,
    current_state: np.ndarray,           # [x, y, theta]
    current_t: int,
    horizon: int,                         # STL_horizon
    q_net: torch.nn.Module,

    # STL / robustness
    stl_task_str: str,
    goals: Dict[int, Dict],               # goals[goal_id] has {'center':..., 'radius':...}
    forward_propagate,                    # (env, current_state, current_t, action, horizon, q_net) -> future_trajectory
    calculate_predicate_robustness,       # (trajectory, center, radius) -> (T,) array
    get_task_robustness,                 # (stl_task_str, predicate_signals) -> robustness

    # history
    state_trajectory: List[np.ndarray],   # past trajectory up to now (list of states)

    # foundation model distribution over actions (already probabilities)
    p_probs: np.ndarray,                  # shape (A,), sums to 1, all > 0

    # constraint
    robustness_threshold: float,          # tau

    # options
    regular_actions: Optional[List[str]] = None,
    bisection_iters: int = 40,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve:
        min_q   KL(q || p)
        s.t.    E_{a~q}[g(a)] >= robustness_threshold
                q in simplex

    Inputs:
      p_probs: foundation model action distribution p (already probabilities)
      g_i: STL robustness when taking action i now then following q_net (computed inside)

    Returns:
      q: optimal distribution (A,)
      g_vals: per-action robustness values (A,)
      lam: Lagrange multiplier used for exponential tilting (0 if q == p)
    """
    if regular_actions is None:
        regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']

    A = len(regular_actions)
    p = np.asarray(p_probs, dtype=np.float64).reshape(-1)
    assert p.shape[0] == A, "p_probs must have shape (len(regular_actions),)."

    # ---- 1) Compute g_i for each action via forward propagation + STL robustness ----
    g_vals = np.zeros((A,), dtype=np.float64)

    for i, a in enumerate(regular_actions):
        if len(state_trajectory) > horizon:
            temp_trajectory = state_trajectory[-(horizon+1):]
        else:
            future_trajectory = forward_propagate(env, current_state, current_t, a, horizon, q_net)
            temp_trajectory = state_trajectory + future_trajectory

        num_goals = len(goals)
        predicate_signals = np.ones((horizon + 1, num_goals), dtype=np.float64) * -999.0
        for goal_id in goals.keys():
            goal = goals[goal_id]
            pred_rob = calculate_predicate_robustness(temp_trajectory, goal['center'], goal['radius'])
            predicate_signals[:, goal_id] = pred_rob

        g_vals[i] = float(get_task_robustness(stl_task_str, predicate_signals))

    
    # ---- 2) If already feasible under p, return p ----
    Eg0 = float(np.dot(p, g_vals))
    print("expected robustness under p:", Eg0)
    if Eg0 >= robustness_threshold:
        return p, g_vals

    # ---- 3) Feasibility check----
    if float(np.max(g_vals)) < robustness_threshold:
        # No distribution can satisfy the constraint. Best-effort: concentrate on best action.
        print("Warning: robustness constraint is infeasible. Returning degenerate distribution on best action.")
        best = int(np.argmax(g_vals))
        q = np.zeros((A,), dtype=np.float64)
        q[best] = 1.0
        return q, g_vals

    # ---- 4) Solve for lambda with bisection: q(lambda) = softmax(log p + lambda g) ----

    def expected_g(lmbda):
        weights = p * np.exp(lmbda * g_vals)
        q = weights / weights.sum()
        return np.sum(q * g_vals)


    lo, hi = 0.0, 1.0
    while expected_g(hi) < robustness_threshold:
        hi *= 2.0

    tol = 1e-6 #tolerance for bisection convergence
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        val = expected_g(mid)

        if abs(val - robustness_threshold) < tol:
            break

        if val >= robustness_threshold:
            hi = mid
        else:
            lo = mid


    lam = hi

    weights = p * np.exp(lam * g_vals)
    q = weights / np.sum(weights)

    return q, g_vals

def kl_project_action_distribution_with_robustness_v2(
    *,
    #TRIES TO PRUNE VIOLATING ACTIONS
    env,
    current_state: np.ndarray,           # [x, y, theta]
    current_t: int,
    horizon: int,                         # STL_horizon
    q_net: torch.nn.Module,

    # STL / robustness
    stl_task_str: str,
    goals: Dict[int, Dict],               # goals[goal_id] has {'center':..., 'radius':...}
    forward_propagate,                    # (env, current_state, current_t, action, horizon, q_net) -> future_trajectory
    calculate_predicate_robustness,       # (trajectory, center, radius) -> (T,) array
    get_task_robustness,                 # (stl_task_str, predicate_signals) -> robustness

    # history
    state_trajectory: List[np.ndarray],   # past trajectory up to now (list of states)

    # foundation model distribution over actions (already probabilities)
    p_probs: np.ndarray,                  # shape (A,), sums to 1, all > 0

    # constraint
    violation_epsilon: float = 1e-6,         # epsilon

    # options
    regular_actions: Optional[List[str]] = None,
    bisection_iters: int = 40,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve:
        min_q   KL(q || p)
        s.t.    E_{a~q}[g(a)] >= robustness_threshold
                q in simplex

    Inputs:
      p_probs: foundation model action distribution p (already probabilities)
      g_i: STL robustness when taking action i now then following q_net (computed inside)

    Returns:
      q: optimal distribution (A,)
      g_vals: per-action robustness values (A,)
      lam: Lagrange multiplier used for exponential tilting (0 if q == p)
    """
    if regular_actions is None:
        regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']

    A = len(regular_actions)
    p = np.asarray(p_probs, dtype=np.float64).reshape(-1)
    assert p.shape[0] == A, "p_probs must have shape (len(regular_actions),)."

    # ---- 1) Compute g_i for each action via forward propagation + STL robustness ----
    g_vals = np.zeros((A,), dtype=np.float64)

    for i, a in enumerate(regular_actions):
        if len(state_trajectory) > horizon:
            temp_trajectory = state_trajectory[-(horizon+1):]
        else:
            future_trajectory = forward_propagate(env, current_state, current_t, a, horizon, q_net)
            temp_trajectory = state_trajectory + future_trajectory

        num_goals = len(goals)
        predicate_signals = np.ones((horizon + 1, num_goals), dtype=np.float64) * -999.0
        for goal_id in goals.keys():
            goal = goals[goal_id]
            pred_rob = calculate_predicate_robustness(temp_trajectory, goal['center'], goal['radius'])
            predicate_signals[:, goal_id] = pred_rob

        g_vals[i] = float(get_task_robustness(stl_task_str, predicate_signals))

    # ---- 2) Build violation indicator v_i = 1[g_i < 0] ----
    v = (g_vals < 0.0).astype(np.float64)

    # print("violation prob under p:", viol_p)

    # If already satisfies chance constraint, q = p
    viol_p = float(np.dot(p, v))
    if viol_p <= violation_epsilon:
        return p, g_vals
    

    # ---- 3) Feasibility check for chance constraint ----
    # If all actions violate, you cannot push violation probability below 1.
    if np.all(v > 0.5):
        print("Warning: all actions violate (g_i < 0 for all i). Returning p.")
        return p, g_vals

    # Also, if violation_epsilon is 0, the optimal is simply renormalize p over safe actions.
    # (bisection will also converge to that in the limit, but this is exact and faster.)
    safe = (v < 0.5)
    if violation_epsilon <= 0.0:
        q = np.zeros_like(p)
        mass = float(p[safe].sum())
        if mass <= 0:
            # p puts zero mass on safe actions; best you can do is choose safest by g (closest to 0)
            best = int(np.argmax(g_vals))
            q[best] = 1.0
            return q, g_vals
        q[safe] = p[safe] / mass
        return q, g_vals

    # ---- 4) Closed form for this constraint: q_i ∝ p_i * exp(-lam * v_i) ----
    # (safe actions unchanged; violating actions uniformly down-weighted by e^{-lam})
    def violation_prob(lmbda: float) -> float:
        w = p * np.exp(-lmbda * v)   # safe: *1, violate: *e^{-lmbda}
        q = w / w.sum()
        return float(np.dot(q, v))

    # We want violation_prob(lam) <= epsilon, with minimal KL => smallest lam satisfying it.
    lo, hi = 0.0, 1.0
    while violation_prob(hi) > violation_epsilon:
        hi *= 2.0

    tol = 1e-8
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        val = violation_prob(mid)

        # if abs(val - violation_epsilon) < tol:
        #     hi = mid
        #     break

        if val <= violation_epsilon:
            hi = mid
        else:
            lo = mid

    lam = hi
    w = p * np.exp(-lam * v)
    q = w / w.sum()

    return q, g_vals

def kl_project_action_distribution_with_robustness_v3(
    *,
    #TRIES TO PRUNE VIOLATING ACTIONS
    env,
    current_state: np.ndarray,           # [x, y, theta]
    current_t: int,
    horizon: int,                         # STL_horizon
    q_net: torch.nn.Module,

    # STL / robustness
    stl_task_str: str,
    goals: Dict[int, Dict],               # goals[goal_id] has {'center':..., 'radius':...}
    forward_propagate,                    # (env, current_state, current_t, action, horizon, q_net) -> future_trajectory
    calculate_predicate_robustness,       # (trajectory, center, radius) -> (T,) array
    get_task_robustness,                 # (stl_task_str, predicate_signals) -> robustness

    # history
    state_trajectory: List[np.ndarray],   # past trajectory up to now (list of states)

    # foundation model distribution over actions (already probabilities)
    p_probs: np.ndarray,                  # shape (A,), sums to 1, all > 0

    # constraint
    tau: float = 1e-6,         # epsilon

    # options
    regular_actions: Optional[List[str]] = None,
    bisection_iters: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve:
        min_q   KL(q || p)
        s.t.    E_{a~q}[g(a)] >= robustness_threshold
                q in simplex

    Inputs:
      p_probs: foundation model action distribution p (already probabilities)
      g_i: STL robustness when taking action i now then following q_net (computed inside)

    Returns:
      q: optimal distribution (A,)
      g_vals: per-action robustness values (A,)
      lam: Lagrange multiplier used for exponential tilting (0 if q == p)
    """
    if regular_actions is None:
        regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']

    A = len(regular_actions)
    p = np.asarray(p_probs, dtype=np.float64).reshape(-1)
    assert p.shape[0] == A, "p_probs must have shape (len(regular_actions),)."

    # ---- 1) Compute g_i for each action via forward propagation + STL robustness ----
    g_vals = np.zeros((A,), dtype=np.float64)

    for i, a in enumerate(regular_actions):
        if len(state_trajectory) > horizon:
            temp_trajectory = state_trajectory[-(horizon+1):]
        else:
            future_trajectory = forward_propagate(env, current_state, current_t, a, horizon, q_net)
            temp_trajectory = state_trajectory + future_trajectory

        num_goals = len(goals)
        predicate_signals = np.ones((horizon + 1, num_goals), dtype=np.float64) * -999.0
        for goal_id in goals.keys():
            goal = goals[goal_id]
            pred_rob = calculate_predicate_robustness(temp_trajectory, goal['center'], goal['radius'])
            predicate_signals[:, goal_id] = pred_rob

        g_vals[i] = float(get_task_robustness(stl_task_str, predicate_signals))

    # ---- 2) Build satisfaction indicator v_i = 1[g_i >= 0] ----
    v = (g_vals >= 0.0).astype(np.float64)


    # If already satisfies chance constraint, q = p
    viol_p = float(np.dot(p, v))
    if viol_p >= tau:
        return p, g_vals
    

    # Also, if tau is 1, the optimal is simply renormalize p over safe actions.
    # (bisection will also converge to that in the limit, but this is exact and faster.)
    safe = (v > 0.5)
    if tau == 1.0:
        q = np.zeros_like(p)
        mass = float(p[safe].sum())
        if mass <= 0:
            # p puts zero mass on safe actions; best you can do is choose safest by g (closest to 0)
            print("Warning: p puts zero mass on safe actions. Returning degenerate distribution on best action.")
            best = int(np.argmax(g_vals))
            q[best] = 1.0
            return q, g_vals
        q[safe] = p[safe] / mass
        return q, g_vals

    # ---- 4) Closed form for this constraint: q_i ∝ p_i * exp(lam * v_i) ----
    def expected_g(lmbda: float) -> float:
        w = p * np.exp(lmbda * v)  
        q = w / w.sum()
        return float(np.dot(q, v))

    # We want violation_prob(lam) <= epsilon, with minimal KL => smallest lam satisfying it.
    lo, hi = 0.0, 1.0
    while expected_g(hi) < tau:
        hi *= 2.0

    tol = 1e-8
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        val = expected_g(mid)

        # if abs(val - violation_epsilon) < tol:
        #     hi = mid
        #     break

        if val >= tau:
            hi = mid
        else:
            lo = mid

    lam = hi
    w = p * np.exp(lam * v)
    q = w / w.sum()

    return q, g_vals



def kl_projection_simple_numpy(p, g, tau, bisection_iters=50):
    #for testing
    """
    Solve:
        min_q KL(q || p)
        s.t. E_q[g] >= tau
    """

    # Already feasible
    if np.dot(p, g) >= tau:
        return p.copy(), 0.0

    # Infeasible
    if np.max(g) < tau:
        q = np.zeros_like(p)
        q[np.argmax(g)] = 1.0
        return q, np.inf

    def expected_g(lmbda):
        z = np.log(p) + lmbda * g
        z = z - np.max(z)     # numerical stability
        q = np.exp(z)
        q /= q.sum()
        return np.dot(q, g)

    lo, hi = 0.0, 1.0
    while expected_g(hi) < tau:
        hi *= 2.0

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        if expected_g(mid) >= tau:
            hi = mid
        else:
            lo = mid

    lam = hi
    z = np.log(p) + lam * g
    z = z - np.max(z)
    q = np.exp(z)
    q /= q.sum()

    return q, lam



if __name__ == "__main__":

    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    g = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    tau = 0.4

    q, lam = kl_projection_simple_numpy(p, g, tau)

    print("Test 1")
    print("E_p[g] =", np.dot(p, g))
    print("Returned q =", q)
    print("Lambda =", lam)
    print("E_q[g] =", np.dot(q, g))
