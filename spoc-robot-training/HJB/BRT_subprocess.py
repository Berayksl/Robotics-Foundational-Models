import subprocess
import pickle
import numpy as np

import subprocess
import pickle
import numpy as np
import os
import sys

def compute_brt_in_subprocess(house_index, target_center, target_radius, time_horizon, output_path):
    """Run BRT computation in a separate Python process."""
    
    # Get the directory containing your HJB module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Or specify explicitly if HJB is elsewhere:
    # hjb_dir = "/home/bera/Desktop/Codes/STL Aware Foundational Models/SPOC/spoc-robot-training"
    
    script = f'''
import sys
sys.path.insert(0, "{current_dir}")

import numpy as np
import pickle

import house_BRT

dynamics = house_BRT.Unicycle(max_v=0.2, max_omega=1.0)

grid, times, target_values, obstacle_values, all_brt_values, geom = house_BRT.compute_house_brt_over_time(
    dynamics=dynamics,
    house_index={house_index},
    target_center={target_center},
    target_radius={target_radius},
    time_horizon={time_horizon},
    n_time_steps={int(time_horizon + 1)}, # 1 second resolution
    robot_radius=0.2,
    wall_thickness=0.1
)

# Convert JAX arrays to numpy for pickling
result = {{
    "times": np.array(times),
    "all_brt_values": np.array(all_brt_values),
    "coordinate_vectors": [np.array(v) for v in grid.coordinate_vectors],
    "grid_shape": grid.shape,
}}

with open("{output_path}", "wb") as f:
    pickle.dump(result, f)

print("BRT computation complete!")
'''
    
    # Use the same Python interpreter
    python_executable = sys.executable
    
    result = subprocess.run(
        [python_executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=current_dir  # Run from the same directory
    )
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"BRT computation failed: {result.stderr}")
    
    print(result.stdout)
    
    # Load results
    with open(output_path, "rb") as f:
        return pickle.load(f)


# Numpy-only version of get_brt_value_at_time (no JAX needed)
def get_brt_value_at_time_numpy(brt_data, state, time_to_go):
    """
    Get interpolated BRT value using numpy only.
    
    Args:
        brt_data: dict with 'times', 'all_brt_values', 'coordinate_vectors'
        state: [x, y, theta] where theta is in DEGREES
        time_to_go: remaining time to reach target
    
    Returns:
        Interpolated value (negative = inside BRT, positive = outside)
    """
    from scipy.ndimage import map_coordinates
    
    times = brt_data["times"]
    all_brt_values = brt_data["all_brt_values"]
    coord_vectors = brt_data["coordinate_vectors"]
    
    state = np.asarray(state, dtype=np.float64)
    state[2] = np.deg2rad(state[2])  # Convert theta to radians
    
    # Time index
    times_flipped = times[::-1]
    indices_flipped = np.arange(len(times))[::-1]
    query_t = -time_to_go
    time_idx = np.interp(query_t, times_flipped, indices_flipped)
    
    # Spatial indices
    indices = [time_idx]
    for i in range(3):
        coord_vec = coord_vectors[i]
        lo, hi = coord_vec[0], coord_vec[-1]
        n = len(coord_vec)
        
        if i == 2:  # theta periodic
            s = state[i] % (2 * np.pi)
        else:
            s = state[i]
        
        idx = (s - lo) / (hi - lo) * (n - 1)
        indices.append(idx)
    
    indices = np.array(indices).reshape(-1, 1)
    
    value = map_coordinates(
        all_brt_values,
        indices,
        order=1,
        mode='wrap'
    )
    
    return float(value[0])
