import torch
import matplotlib.pyplot as plt

def beta_funnel(t, T, beta_min, beta_max, p=2.0, device="cpu"):
    u = torch.clamp(torch.tensor(t / max(T, 1), device=device), 0.0, 1.0)
    return beta_min + (beta_max - beta_min) * (u ** p)


import math

def beta_exp_saturating(t, T, beta_min, beta_max, k=6.0):
    u = min(max(t / max(T, 1), 0.0), 1.0)
    g = (1.0 - math.exp(-k * u)) / (1.0 - math.exp(-k))
    return beta_min + (beta_max - beta_min) * g



def beta_logistic_saturating(t, T, beta_min, beta_max, k=12.0, u0=0.7):
    u = min(max(t / max(T, 1), 0.0), 1.0)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    s0 = sig(k*(0.0 - u0))
    s1 = sig(k*(1.0 - u0))
    su = sig(k*(u   - u0))
    g = (su - s0) / (s1 - s0)
    return beta_min + (beta_max - beta_min) * g



if __name__ == "__main__":
    # Example usage of beta_funnel
    T = 60
    beta_min = 0.1
    beta_max = 2.0
    device = "cpu"

    # beta_values = [beta_funnel(t, T, beta_min, beta_max, p=2.0, device=device).item() for t in range(T+1)]
    # plt.plot(beta_values)
    # plt.xlabel("Time step")
    # plt.ylabel("Beta value")
    # plt.show()   



    beta_values_exp = [beta_exp_saturating(t, T, beta_min, beta_max, k= -3.0) for t in range(T+1)]
    plt.plot(beta_values_exp)
    plt.xlabel("Time step")
    plt.ylabel("Beta value (Exponential Saturating)")
    plt.grid(True)
    plt.show()   


    beta_values_logistic = [beta_logistic_saturating(t, T, beta_min, beta_max, k=12.0, u0=0.7) for t in range(T+1)]
    plt.plot(beta_values_logistic)
    plt.xlabel("Time step")
    plt.ylabel("Beta value (Logistic Saturating)")
    plt.grid(True)
    plt.show() 