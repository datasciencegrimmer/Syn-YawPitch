# Syn-YawPitch

This repository includes the official code to the paper "Pose Impact Estimation on Face Recognition using 3D-Aware Synthetic Data with Application to Quality Assessment". We provide our scripts for generating our synthetic Syn-YawPi database together with our best performing SYP-Lasso pose quality estimators (PQEs) - trained on four different face recognition systems (FRSs): ArcFace, MagFace, CurricularFace, and a COTS FRS.


## Syn-YawPitch Generation

To reconstruct our proposed Syn-YawPitch database, we provide a numpy array of seeds (Syn-YawPitch/selected_seeds.npy) that generates the same facial images when given to the EG3D generator. The seeds are selected to draw 1000 latent vectors that keep a minimum distance in the latent space to each other in order to prevent unnatural look-alike rates across the generated non-mated samples. 

**NOTE** Our Syn-YawPitch DB builds upon the EG3D face image generator and 6DRepNet pose estimator: In order to run the provided script given here, make sure to install their requirements too:

 - https://github.com/NVlabs/eg3d
 - https://github.com/thohemp/6drepnet

## Pose Quality Estimation

To use our proposed SYP-Lasso PQE, an example script is given under inference/SYP-lasso-inference.py. Given a facial image, the SYP-Lasso regression model first requires to estimate the head poses (yaw and pitch angle), from which it estimates the pose quality in the range between 0 to 100 following the specifications of ISO/IEC WD5 29794-5. We recommend the usage of 6DRepNet (https://github.com/thohemp/6drepnet) to estimate the head poses, as it has proven well-compatible with our SYP-Lasso regression model according to our evaluation results. 


### Baseline Quality Estimation

Alongside our proposed SYP-Lasso regression model, we provide additional inference scripts used as baseline pose estimators in our paper: bPQE (inference/bPQE-inference.py) and ISO/IEC WD5 29794-5 related PQE (inference/iso-related-inference.py).


## Citation

If you use our code or pre-trained models, please reference the following work: 

@article{grimmer2023pose,
  title={Pose Impact Estimation on Face Recognition using 3D-Aware Synthetic Data with Application to Quality Assessment},
  author={M. Grimmer and C. Rathgeb and C. Busch},
  journal={arXiv preprint arXiv:2303.00491},
  year={2023}
}


## Special Credicts

We highlight the great work of EG3D and 6DRepNet: 

@inproceedings{chan2022efficient,
  title={Efficient geometry-aware 3D generative adversarial networks},
  author={E.R. Chan and C.Z. Lin and M.A. Chan and K. Nagano and B. Pan and S. De Mello and O. Gallo and L.J. Guibas and J. Tremblay and S. Khamis and others},
  booktitle={Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition},
  pages={16123--16133},
  year={2022}
}

@inproceedings{hempel20226d,
  title={6d rotation representation for unconstrained head pose estimation},
  author={T. Hempel and A.A. Abdelrahman and A. Al-Hamadi},
  booktitle={2022 IEEE Intl. Conf. on Image Processing (ICIP)},
  pages={2496--2500},
  year={2022},
  organization={IEEE}
}



