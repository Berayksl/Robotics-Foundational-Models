import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_q_gaps(q_values_over_time):
    gaps = []
    best_actions = []

    for q in q_values_over_time:
        sorted_q = sorted(q, reverse=True)
        gap = sorted_q[0] - sorted_q[1]
        gaps.append(gap)
        best_actions.append(q.argmax())

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(gaps, marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Q-value gap (best − second best)")
    plt.title("Action Gap vs Time for Fixed State")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    spread = []
    for q in q_values_over_time:
        spread.append(q.max() - q.min())

    plt.plot(spread)
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Q-value gap (max − min)")
    plt.title("Q-value gap (min vs max) vs time")
    plt.show()

    entropy = []
    for q in q_values_over_time:
        p = np.exp(q)
        p /= p.sum()
        entropy.append(-(p * np.log(p + 1e-8)).sum())

    plt.plot(entropy)
    plt.title("Action entropy vs time")
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Entropy")
    plt.show()


    q_values_over_time = np.array(q_values_over_time)
    A = q_values_over_time - q_values_over_time.mean(axis=-1, keepdims=True) #advantage values
    A = A / (q_values_over_time.std(axis=-1, keepdims=True) + 1e-6)   # per-state normalization
    normalized_spread = []
    
    for a in A:
        normalized_spread.append(a.max() - a.min())

    plt.plot(normalized_spread)
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Normalized Q-value gap (max − min)")
    plt.title("Normalized Q-value gap (min vs max) vs time")
    plt.show()

    #plot the entropy of normalized Q-values
    normalized_entropy = []
    for a in A:
        p = np.exp(a)
        p /= p.sum()
        normalized_entropy.append(-(p * np.log(p + 1e-8)).sum())
    plt.plot(normalized_entropy)
    plt.title("Action entropy of normalized Q-values vs time")
    plt.grid(True)
    plt.xlabel("Time step")
    plt.ylabel("Entropy")
    plt.show()



def plot_q_values(q_values_over_time):
    action_space = ['m', 'b', 'l', 'r', 'ls', 'rs']
    q_values_over_time = np.array(q_values_over_time)
    plt.figure(figsize=(10, 6))
    for i in range(q_values_over_time.shape[1]):
        plt.plot(q_values_over_time[:, i], label=f"Action {action_space[i]}")
    plt.xlabel("Time step")
    plt.ylabel("Q-value")
    plt.title("Q-values vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
