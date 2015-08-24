# rfov reconstruction for non-Cartesian PROPELLER MRI

## Iterative non-Cartesian recostruction with split-Bregman (SB) method
Split-Bregman method can include various regularization terms, including total variation (TV).   
This software uses SB method for TV regularization.    
rFOV allows rapid reconstruction, if only a restricted area is needed.    

## Usage
Run the test_scripts in the tests folder.   
$cd tests  
$./test_scripts  

If you see images, the reconstruction should have been successful.   
If you see "reconstruction accomplished using GPU", GPU has been applied to the reconstruction.  
Otherwise, CPU is used rather than GPU.   

# Dependency
The software was tested on the following hardware software:   
Gentoo 3.19.3-aufs, X11 with nvidia driver enabled.    
Intel i7 CPU, 16GB memory, Nvidia 4200m and Nvidia GTX560,   
python 2.7.9-r1, numpy 1.9.0-r1, scipy 0.15.1, matplotlib 1.4.3, nvidia-cuda-sdk-6.5.14, nvidia-drivers-352.21, pyFFTW-0.9.2 and PyFFTW3 0.2.1   

# Warning
Do not load the open-source nouveau driver because nouveau driver is not compatible with nvidia/CUDA. 


## Using GPU
To use GPU, relevant CUDA/pyopencl/reikna-0.6.4 software should be available.  
	
