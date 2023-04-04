import pickle as pkl
import numpy as np

    
def run_example_bPQE():
    
    # NOTE: random angles for demonstration only
    # Use pose estimator to extract yaw and pitch angles of a given face image
    yaw = 10
    pitch = 10

    bPQE = 100 - 100 * (abs(float(yaw)) + abs(float(pitch))) / 180
    sq_bPQE = 100 - 100 * (abs(float(yaw))**2 + abs(float(pitch))**2) / (90**2 + 90**2)
    quad_bPQE = 100 - 100 * (abs(float(yaw))**4 + abs(float(pitch))**4) / (90**4 + 90**4)
    oct_bPQE = 100 - 100 * (abs(float(yaw))**8 + abs(float(pitch))**8) / (90**8 + 90**8)

    print(f"bPQE with n = 1: {bPQE}")
    print(f"bPQE with n = 2: {sq_bPQE}")
    print(f"bPQE with n = 4: {quad_bPQE}")
    print(f"bPQE with n = 8: {oct_bPQE}")


if __name__ == "__main__":
    
    run_example_bPQE()
 
    

    

