import pickle as pkl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

    
def run_example_SYP_lasso_PQE():
    
    # Specify fr_model and paths
    fr_model = "arcface"
    path_to_syp_lasso = f"path/to/SYP/model/SYP-Lasso-{fr_model}.pkl"
    path_to_syp_feature_transformer =  f"path/to/SYP/feature/preparation/SYP-Lasso-{featurePrepare}.pkl"
    with open(f'/mnt/tests/eg3d/eg3d/evaluation/models/{fr_model}_fnmr_poly{exponent}_model_{alpha}.pkl', 'rb') as file:
        qpredictor = pkl.load(file)
        
    with open(f'/mnt/tests/eg3d/eg3d/evaluation/models/{fr_model}_fnmr_feature_poly{exponent}_{alpha}.pkl', 'rb') as file:
        featurePrepare = pkl.load(file)

    # NOTE: random angles for demonstration only
    # Use pose estimator to extract yaw and pitch angles of a given face image
    yaw = 15
    pitch = -15

    # Split angles into positive and negative parts
    negyaw, posyaw, negpitch, pospitch = get_neg_pos_components(yaw, pitch)

    # Stack angles as input feature
    input_pts = np.stack([negpitch, pospitch, negyaw, posyaw]).T.reshape((1, -1))
    # Convert to 2-polynom features
    poly_input_pts = featurePrepare.transform(input_pts)
    # Use SYP-Lasso to estimate quality
    q_value = qpredictor.predict(poly_input_pts)[0]
    # Clip to cosine similarity -1 to 1 range
    q_value = np.clip(q_value, -1, 1)
    # Scale into 0 to 100 range according to ISO requirements 
    q_value = (q_value + 1) / 2 * 100

    print(q_value)



def get_neg_pos_components(yaw, pitch):
    if yaw <= 0:
        negyaw = yaw
        posyaw = 0
    elif yaw >= 0:
        negyaw = 0
        posyaw = yaw
    if pitch <= 0:
        negpitch = pitch
        pospitch = 0
    elif pitch >= 0:
        negpitch = 0
        pospitch = pitch
    return(negyaw, posyaw, negpitch, pospitch)


if __name__ == "__main__":
    
    run_example_SYP_lasso_PQE()
 
    

    

