# rfov reconstruction for non-Cartesian PROPELLER MRI

## Iterative non-Cartesian recosntruction with split-Bregman (SB) method
Split-Bregman method can include various regularization terms, including total variation (TV). 
This software is a example to use SB for TV regularization. 
rFOV allows rapid reconstruction, if a restricted area is needed. 

## Usage
run the test_scripts in the tests folder.
$cd tests
$./test_scripts

If you see images, the reconstruction should have been successful. 

## Using GPU
To use GPU, relevant CUDA/pyopencl/reikna-0.6.4 software should have been installed.
If you see "reconstruction accomplished using GPU", GPU has been for the reconstruction. 
Otherwise, there must be something wrong. 

