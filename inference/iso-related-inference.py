import pickle as pkl
import numpy as np
import math

    
def run_example_iso_related_PQE():
    
    # NOTE: random angles for demonstration only
    # Use pose estimator to extract yaw and pitch angles of a given face image
    yaw = 50
    pitch = 10
        
    # convert from degree to radian
    radian_pitch = math.radians(pitch) 
    radian_yaw = math.radians(yaw) 
    # apply pose quality measures from ISO/IEC WD5 29794-5
    iso_pitch_quality = max(0, 100 * math.cos(radian_pitch)**2)
    iso_yaw_quality = max(0, 100 * math.cos(radian_yaw)**2)
    # Derive fused quality score with MIN operator
    iso_pose_quality = np.min([iso_pitch_quality, iso_yaw_quality])
    

    print(f"ISO/IEC WD5 29794-5 related PQE: {iso_pose_quality}")


if __name__ == "__main__":
    
    run_example_iso_related_PQE()
 
    

    

