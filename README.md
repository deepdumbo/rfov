# rfov reconstruction for non-Cartesian PROPELLER MRI

## Iterative non-Cartesian recosntruction with split-Bregman (SB) method
Split-Bregman method can include various regularization terms, including total variation (TV). 
This software is a example to use SB for TV regularization. 
rFOV allows rapid reconstruction, if only a restricted area is needed. 

## Usage
Run the test_scripts in the tests folder.
$cd tests
$./test_scripts

If you see images, the reconstruction should have been successful. 
If you see "reconstruction accomplished using GPU", GPU has been for the reconstruction.
Otherwise, CPU has been used instead of GPU.

# Dependency
Untile the date of update, the software was tested on the following hardware software:
Gentoo 3.19.3-aufs, x11 with nvidia driver enabled. 
Intel i7 CPU, 16GB memory, Nvidia 4200m  (1G mem) or Nvidia GTX560, 
python 2.7.9-r1, numpy 1.9.0-r1, scipy 0.15.1, matplotlib 1.4.3, nvidia-cuda-sdk-6.5.14, nvidia-drivers-352.21, pyFFTW-0.9.2 and PyFFTW3 0.2.1

## Using GPU
To use GPU, relevant CUDA/pyopencl/reikna-0.6.4 software should have been installed.
	
