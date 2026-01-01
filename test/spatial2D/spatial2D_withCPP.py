import numpy as np
from scipy.ndimage import convolve
import subprocess
import os


# =============================================
# Parameters and Initial Grid
# =============================================
# input image file name
input_image_file = 'initial_grid1_cpp.txt'   
output_image_file = 'final_grid1_cpp.txt'

# Open txt file and save as numpy array
initial_grid = np.loadtxt(input_image_file, dtype=int)

# parameters
alpha = 0.06 #np.random.normal(0.06, 0.001) # upgrade rate
beta = 0.0002 #np.random.normal(0.0003, 0.00001)  # hotspot formation rate
gamma = 0.2 #np.random.normal(0.2, 0.005)  # hostpot addition rate
dt = 0.1    # time step

t0 = 0.0
t_end = 24.0
steps = int((t_end - t0) / dt)


# ============================================
# Run C++ version with parameters from Python
# ============================================

def run_cpp_simulation(alpha, beta, gamma, dt, t0, t_end, 
                       input_grid=input_image_file, 
                       output_grid=output_image_file):
    """
    Run the C++ simulation with parameters passed as command-line arguments
    
    Parameters:
    -----------
    alpha : float - upgrade rate
    beta : float - hotspot formation rate
    gamma : float - hotspot addition rate
    dt : float - time step
    t0 : float - start time
    t_end : float - end time
    input_grid : str - input grid filename
    output_grid : str - output grid filename
    """
    
    # # Check if C++ executable exists
    # if not os.path.exists('./spatial2D'):
    #     print("C++ executable not found. Compile with: g++ -std=c++11 -O2 spatial2D.cpp -o spatial2D")
    #     return None
    
    # Run the C++ simulation with parameters as arguments
    try:
        result = subprocess.run(
            ['./spatial2D', 
             str(alpha), str(beta), str(gamma), 
             str(dt), str(t0), str(t_end),
             input_grid, output_grid],
            check=True,
            capture_output=True,
            text=True
        )
        print("C++ Simulation Output:")
        print(result.stdout)
        
        # Load and return the result
        cpp_grid = np.loadtxt(output_grid, dtype=int)
        return cpp_grid
        
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ simulation: {e}")
        print(f"stderr: {e.stderr}")
        return None


# Example usage: Run C++ with same parameters as Python
if __name__ == "__main__":
    # print("\n" + "="*50)
    # print("Running C++ simulation with Python parameters...")
    # print("="*50)
    
    cpp_result = run_cpp_simulation(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dt=dt,
        t0=t0,
        t_end=t_end
    )
    
    # if cpp_result is not None:
    #     print(f"\nC++ result grid shape: {cpp_result.shape}")