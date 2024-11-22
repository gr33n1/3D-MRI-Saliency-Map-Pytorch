# 3D-MRI-Saliency-Map-Pytorch
## This repository contains the code relative to the paper "Deep learning Modeling of Structural Brain MRI in Chronic Head and Neck Pain Following Mild TBI".
#### [paper link] 

### Please cite the paper when referring to this repository.

#### [citation]

### About
The Model presented is an example with random weights.
Please use your own model architecture and weights.

Images were taken from IXI Dataset https://brain-development.org/ixi-dataset/

The images were normalized with spm with the following pseudocode

https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.spm.preprocess.html#normalize12

    from nipype.interfaces import spm

    norm12 = spm.Normalize12()
    norm12.inputs.image_to_align = input_path
    norm12.inputs.affine_regularization_type = 'mni'
    norm12.inputs.out_prefix = 'w'
    norm12.run()

Image current input size for saliency_map.py (1, 30, 95, 79) (batch, num_slices, height, width)

### Run
    python saliency_map.py
