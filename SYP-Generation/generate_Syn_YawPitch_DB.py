import operator
from typing import Protocol
import numpy as np
import pickle as pkl
import os
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# configs ----------------------------------------------------------------
fr_model = "cognitec"
numIDs =  1000
matedcompScoreDir = f'/mnt/PhD-Marcel/SynPose/comparison-scores-intersect/{fr_model}_cosine_sim_synyawpi_extPoses_intersect_{numIDs}_ids.npy'
yawAnglesDir = f'/mnt/PhD-Marcel/SynPose/comparison-scores-intersect/{fr_model}_yaw_synyawpi_extPoses_intersect_{numIDs}_ids.npy'
pitchAnglesDir = f'/mnt/PhD-Marcel/SynPose/comparison-scores-intersect/{fr_model}_pitch_synyawpi_extPoses_intersect_{numIDs}_ids.npy'
nonMatedcompScoreDir = f'/mnt/PhD-Marcel/SynPose/comparison-scores-intersect/{fr_model}_nonmated_cosine_distances_synyawpi_random_intersect_{numIDs}_ids.pkl'
outdir_figs = '/mnt/PhD-Marcel/SynPose/results'
outdir_errs = '/mnt/PhD-Marcel/SynPose/error-rates'
fixed_error_percent = 0.01
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
"""
Computes the false non-match rates and false match rates for various thresholds

Input: Mated comparison scores & Non-Mated comparison scores
Output: Thresholds with corresponding FMRs and FNMRs
"""
def calculate_roc(gscores, iscores, ds_scores=False, rates=True):

    if isinstance(gscores, list):
        gscores = np.array(gscores, dtype=np.float64)

    if isinstance(iscores, list):
        iscores = np.array(iscores, dtype=np.float64)

    if gscores.dtype == np.int:
        gscores = np.float64(gscores)

    if iscores.dtype == np.int:
        iscores = np.float64(iscores)

    if ds_scores:
        gscores = gscores * -1
        iscores = iscores * -1

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    gscores = zip(gscores, [1] * gscores_number)
    iscores = zip(iscores, [0] * iscores_number)

    gscores = list(gscores)
    iscores = list(iscores)

    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
    cumul = np.cumsum(scores[:, 1])

    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    fnm = cumul[u_indices] - scores[u_indices][:, 1]
    fm = iscores_number - (u_indices - fnm)

    if rates:
        fnm_rates = fnm / gscores_number
        fm_rates = fm / iscores_number
    else:
        fnm_rates = fnm
        fm_rates = fm

    if ds_scores:
        return thresholds * -1, fm_rates, fnm_rates

    return thresholds, fm_rates, fnm_rates
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
"""
Computes the EER based on given FMR and FNMRs
Input: FMR & FNMR
Output: EER
"""
def get_eer(fmr, fnmr):
    diff = fmr - fnmr
    t2 = np.where(diff <= 0)[0]

    if len(t2) > 0:
        t2 = t2[0]
    else:
        return 0, 1, 1, 1

    t1 = t2 - 1 if diff[t2] != 0 and t2 != 0 else t2

    return (fnmr[t2] + fmr[t2]) / 2

#-------------------------------------------------------------------------
"""
Helper function to created pickleable nested defaultsdicts
"""
def nested_defaultdict():
    return defaultdict(list)
#-------------------------------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
#-------------------------------------------------------------------------
"""
Computes the EER based on given FMR and FNMRs
Input: FMR & FNMR
Output: EER
"""
def convert_to_level_dict(matedScores, yawAngles, pitchAngles):

    pose_ordered_mated = defaultdict(nested_defaultdict)
    
    
    for i in range(len(matedScores)):

        for j in np.linspace(-80, 80, int((80+80) + 1)):
            if yawAngles[i] == j:
                yaw_l = f"y{int(j)}"
            if pitchAngles[i] == j:
                pitch_l = f"p{int(j)}"
        
        pose_ordered_mated[yaw_l][pitch_l].append(matedScores[i])
        pose_ordered_mated["y0"]["p0"].append(1) 
    
    return pose_ordered_mated
#-------------------------------------------------------------------------


if __name__=="__main__":
    # read mated and non-mated comp. scores
    
    matedCompScores = np.load(matedcompScoreDir)
    yawAngles = np.load(yawAnglesDir)
    pitchAngles = np.load(pitchAnglesDir)
    with open(nonMatedcompScoreDir, 'rb') as f:
        nonMatedCompScores = pkl.load(f)
        
    matedCompScores = convert_to_level_dict(matedCompScores, yawAngles, pitchAngles)

    # initiate lists to plot line chart
    nonMated_scores = [1- nmcs for nmcs in nonMatedCompScores['nonmated-frontal-comp']]
    
    
    yaw_labels = []
    pitch_labels = []
    eers = []
    fnmrs = []
    fmrs = []
    # Compute EER / FNMR / FMR for different pairs of pitch and yaw angles
    matedCompScores = dict(sorted(matedCompScores.items()))
    for k1 in matedCompScores.keys():
        matedCompScores[k1] = dict(sorted(matedCompScores[k1].items()))
        for k2 in matedCompScores[k1].keys():
            yaw_labels.append(int(k1[1:]))
            pitch_labels.append(int(k2[1:]))
            mated_scores = matedCompScores[k1][k2]
            _, fmr, fnmr = calculate_roc(mated_scores, nonMated_scores, ds_scores = False)
            eer = get_eer(fmr, fnmr) *100
            fnmr_idx, fnmr_v = find_nearest(fnmr, fixed_error_percent)
            fmr_idx, fmr_v = find_nearest(fmr, fixed_error_percent)
            fnmrs.append(0 if fnmr[fmr_idx] == 1.0 else fnmr[fmr_idx]*100)
            fmrs.append(0 if fmr[fnmr_idx] == 1.0 else fmr[fnmr_idx]*100)
            eers.append(eer)
    

    # save error rates
    yaw_labels = np.array(yaw_labels)
    np.save(os.path.join(outdir_errs, f"{fr_model}-extPoses-intersect-yaw-labels-{numIDs}-ids.npy"), yaw_labels)
    pitch_labels = np.array(pitch_labels)
    np.save(os.path.join(outdir_errs, f"{fr_model}-extPoses-intersect-pitch-labels-{numIDs}-ids.npy"), pitch_labels)
    eers = np.array(eers)
    np.save(os.path.join(outdir_errs, f"{fr_model}-extPoses-intersect-eers-{numIDs}-ids.npy"), eers)
    fnmrs = np.array(fnmrs)
    np.save(os.path.join(outdir_errs, f"{fr_model}-extPoses-intersect-fnmrs-{numIDs}-ids.npy"), fnmrs)
    fmrs = np.array(fmrs)
    np.save(os.path.join(outdir_errs, f"{fr_model}-extPoses-intersect-fmrs-{numIDs}-ids.npy"), fmrs)
    
    # results = np.array([yaw_labels, pitch_labels, eers, fnmrs, fmrs])
    # np.savetxt("data.csv", results, delimiter=",", fmt='%s')
#-------------------------------------------------------------------------
