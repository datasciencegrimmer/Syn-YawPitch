# Syn-YawPitch

This repository includes the official code to the paper "Pose Impact Estimation on Face Recognition using 3D-Aware Synthetic Data with Application to Quality Assessment". We provide our scripts for generating our synthetic Syn-YawPi database together with our best performing SYP-Lasso pose quality estimators - pretrained on four different face recognition systems: ArcFace, MagFace, CurricularFace, and the COTS face recognition system.


## 1) Generation of Syn-YawPitch DB

To reconstruct our proposed Syn-YawPitch database, we provide a numpy array of selected seeds that generate the same facial images with EG3D, making sure to comply to the minimum distance criterium.  

All requirements can be adopted from the official EG3D repository: https://github.com/NVlabs/eg3d

## 2) Pose Quality Prediction with our SYP-Lasso models

! Documentation and code follows!

