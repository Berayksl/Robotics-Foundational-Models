import numpy as np
import stlrom

def get_task_robustness(task, signals):
    stl_driver =stlrom.STLDriver()
    stl_driver.parse_string(task)

    #add the samples:
    for i in range(len(signals)):
        stl_driver.add_sample([i, signals[i][0], signals[i][1]])   #format [t, pred1_val, pred2_val, ...]

    phi = stl_driver.get_monitor("phi") #overall task
    phi2 = stl_driver.get_monitor("phi2") #overall task
    phi1 = stl_driver.get_monitor("phi1") #subtask 1
    print("Robustness of phi1:", phi1.eval_rob())
    print("Robustness of phi2:", phi2.eval_rob())
    robustness = phi.eval_rob()

    return robustness    

s = """
    signal x, y    # signal names
    mu_1 := x[t] > 0  # goal-1
    mu_2 := y[t] > 0   #goal-2

    phi1 := ev_[0, 40] mu_1
    phi2 := ev_[45, 80] mu_2  # eventually (or F) 
    phi := phi1 and phi2   # boolean combination 
    """



#CREATE a pesudo signal for testing (time, signal1, signal2):
time_steps = 80
signal1 = np.concatenate((-1*np.ones(15), np.ones(15)))  #satisfies mu_1 in the first 30 steps, then violates it, then satisfies it again
signal2 = np.concatenate((-1*np.ones(15),np.ones(15)))  #violates mu_2 in the first 45 steps, then satisfies it, then violates it again
signals = np.stack((signal1, signal2), axis=1)

print(signals)
print("Robustness of the task with the given signals:", get_task_robustness(s, signals))