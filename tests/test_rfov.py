
'''
A demonstration for reduced field-of-view reconstruction 
the main() command is at the end of this python file
From Linux bash environment, call:
$ python2.7 test_rfov.py

'''

 
# setting path
import sys 
sys.path.append('..')
sys.path.append('CsTransform/utils') 
sys.path.append('CsTransform') 

# importing required modules
from CsTransform.utils.utils import *
from CsTransform.pynufft import *


def rfov_2D(gpu_type):
 
    import numpy 

 
    N = 384
    Nd =(N,N) # image space size
    # create Shepp-Logan phantom
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(6,6) # interpolation size
    import KspaceDesign.Propeller
    nblade =26# numpy.round(N*3.1415/24)
#     print('nblade',nblade)
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,24,  nblade,   theta,  1,  -1)
    om = propObj.om
#     print('m',numpy.size(om))
    # load k-space points
#     om = numpy.loadtxt('om.txt')
#     om = numpy.loadtxt('om.gold')
     
    #create object
#     import cPickle
    NufftObj = pynufft(om, Nd,Kd,Jd )
#     f = open('nufftobj.pkl','wb')
#     cPickle.dump(NufftObj,f,2)
#     f.close()
# # #      
#     f = open('nufftobj.pkl','rb')
#     NufftObj = cPickle.load(f)
#     f.close()
#     precondition = 2
#     factor = 0.01
#     NufftObj.factor = factor
#     NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    data_shape = numpy.shape(data)
    power_of_data= numpy.max(numpy.abs(data))

    data = data + 1e-3*power_of_data*(numpy.random.randn(data_shape[0],data_shape[1])+1.0j*numpy.random.randn(data_shape[0],data_shape[1]))
    
    LMBD =1
    nInner=2
    nBreg = 25
    
    mu=1.0
    gamma = 0.001
    
    # Setting reduced FOV
    rFOV = 96
    NufftObj = pynufft(om, (rFOV,rFOV),Kd,Jd,n_shift=(0,0))
 
    # Shift the center of the image by applying a linear phase on k-space 
    NufftObj.linear_phase(  (-128, 0) )     
    
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 0



    # shrinkage threshold 
    NufftObj.beta = 1 
    
    # Attempt to set GPU computing (warning: cuda hardware and reikna/pyopencl are needed!)
    if gpu_type=='GPU': 
        NufftObj.initialize_gpu()
    else:
        pass
    
    # debug level
    NufftObj.debug = 2 # 2 for showing splitting variables and text outputs, 1 for text outputs



    
    image_blur = NufftObj.backward2(data)  
    

     
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
 
    # Normalize the contrast of images
    mean_image_recon = numpy.mean(numpy.real(image_recon[41:62, 8:24 ].flatten()))
    
    image_recon = numpy.real(image_recon)/mean_image_recon/5.0# *1.3
    mean_image_blur = numpy.mean(numpy.real(image_blur[41:62,  8:24  ].flatten()))
    image_blur=numpy.real(image_blur[...,0])/mean_image_blur/5.0#*1.15
    
    

    # Display images
    import matplotlib#.pyplot
    
    # Set grey color maps 
    cm = matplotlib.cm.gray
    
    # Set image window
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1) 
    
    
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.subplot(1,2,1)
    matplotlib.pyplot.imshow((image_recon).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('denoising by shrinkage')    

    matplotlib.pyplot.subplot(1,2,2)
    matplotlib.pyplot.imshow(numpy.real(image_blur).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('density compensation')  
 
    matplotlib.pyplot.show()
    
    return NufftObj.gpu_flag # return the GPU to see whether GPU computing has been used 

def test_2D():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=-0.0, vmax=1) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
    import cPickle
#     image = numpy.loadtxt('phantom_256_256.txt') 
#     matplotlib.pyplot.imshow(image[224-128:224, 64:192] ,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()    
#     image[128,128]= 1.0   
    N = 256
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(6,6) # interpolation size
    import KspaceDesign.Propeller
    nblade =  16#numpy.round(N*3.1415/24/2)
#     print('nblade',nblade)
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,24,  nblade,   theta,  1,  -1)
    om = propObj.om
     
    NufftObj = pynufft(om, Nd,Kd,Jd )
#         
         
#     f = open('nufftobj.pkl','wb')
#     cPickle.dump(NufftObj,f,2)
#     f.close()
# # #      
#     f = open('nufftobj.pkl','rb')
#     NufftObj = cPickle.load(f)
#     f.close()

#     f2 = open('dense_density.pkl','wb')
#     cPickle.dump(NufftObj.st['w'],f2,-1)
#     f2.close()
#     print('size of sparse amtrix',NufftObj.st['p'].data.nbytes + NufftObj.st['p'].indptr.nbytes + NufftObj.st['p'].indices.nbytes)
#     print('size of dense matrix', NufftObj.st['w'].nbytes)


#     NufftObj = pynufft(om, (64,64),(512,512),Jd) 
    precondition = 0
    factor = 0.001
    NufftObj.factor = factor
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
#     data= NufftObj.true_forward(image )  
    
    data_shape = (numpy.shape(data))
    power_of_data= numpy.max(numpy.abs(data))
#     purpose = 2 # noisy
    purpose = 1 # noisy
#     purpose = 0 # noise-free
    data = data + purpose*1e-3*power_of_data*(numpy.random.randn(data_shape[0],data_shape[1])+1.0j*numpy.random.randn(data_shape[0],data_shape[1]))
    LMBD =1
    nInner=2
    nBreg = 25
    
    mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    rFOV = 96
    NufftObj = pynufft(om, (rFOV,rFOV),Kd,Jd,n_shift=(0,0))
 
    # Shift the center of the image by applying a linear phase on k-space 
    NufftObj.linear_phase(  (-128, 0) ) 
        
    NufftObj.st['senseflag'] = 1
    NufftObj.initialize_gpu() 
    NufftObj.debug=2 #depicting the intermediate images
#     NufftObj = pynufft(om, (64,64),(96,96),Jd)#,n_shift=(192,96))
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    # Setting reduced FOV

    image_blur = NufftObj.backward2(data)  
    

#     image_recon_inner_10 = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma, 10, nBreg)
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
    image_recon = Normalize(image_recon.real)
    image_blur=Normalize(numpy.real(image_blur[...,0]))*1.2 #4.9

#     my_scale = numpy.mean( image_blur[round(25*N/256):round(46*N/256), round(109*N/256):round(149*N/256)].flatten())
#     image_blur =image_blur / my_scale /5.0 
    
    my_scale = numpy.mean( image_recon[round(25*N/256):round(46*N/256), round(109*N/256):round(149*N/256)].flatten())
    image_recon =image_recon / my_scale /5.0 
#     print(my_scale)
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
#     if purpose == 0:
#         image_recon = Normalize(image_recon.real)*1.05#2.2 # *1.3
#     elif purpose == 1:
#         image_recon = Normalize(image_recon.real)*1.1#54#2.2 # *1.3
# #     image_recon_inner_10 = Normalize(numpy.real(image_recon_inner_10))#*  1.1# *1.3
#     image_blur=Normalize(numpy.real(image_blur[...,0]))*1.2#4.9
#     if purpose == 0:
#         image_recon = Normalize(image_recon.real)*1.3#2.2 # *1.3
#     elif purpose == 1:
#         image_recon = Normalize(image_recon.real)*1.0#54#2.2 # *1.3
#     elif purpose == 2:
#         image_recon = Normalize(image_recon.real)*1.1#54#2.2 # *1.3     

#     image_recon_inner_10 = Normalize(numpy.real(image_recon_inner_10))#*  1.1# *1.3
    disp_image = numpy.concatenate( 
                                       (
#                         numpy.concatenate( ((  image_recon.real) , numpy.real(-image_recon  + image) ), axis = 0 ),
                        numpy.concatenate( ((  image_recon.real) , numpy.abs(image_recon  - image) ), axis = 0 ),
                        numpy.concatenate( ((  image_blur.real) ,numpy.abs(image_blur  - image) ), axis = 0  ) ), axis =1 )
    import scipy.misc 
    scipy.misc.imsave('/home/sram/Cambridge_2012/WORD_PPTS/PROP/generalize_inverse/MRM/FIGS/noise_XX.png',disp_image)
    fig, ax = matplotlib.pyplot.subplots()
    cax = ax.imshow( disp_image 
                     , norm = norm,
            cmap= cm,interpolation = 'nearest')
    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['< 0', '0.5', '> 1'])# vertically oriented colorbar
    matplotlib.pyplot.savefig('/home/sram/Cambridge_2012/WORD_PPTS/PROP/generalize_inverse/MRM/FIGS/noise_XX.eps', bbox_inches='tight')
    
#     matplotlib.pyplot.figure(1)
#     matplotlib.pyplot.subplot(n_x,n_y,tt)        
#     matplotlib.pyplot.imshow(numpy.abs(image_recon.real) ,
#                             norm = norm,
#             cmap= cm,interpolation = 'nearest')  
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('blurred image') 
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.imshow( numpy.abs(image_recon[85:215,73:203].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('residual error') 
#      
#     matplotlib.pyplot.show() 
    
if __name__ == '__main__':
    import cProfile

    print('Hello World!')
    print('This is a python test file to test the rFOV method')
    print('You must have numpy, scipy, matplotlib installed')
    print('pyFFTW for better performance ')
    print('Nvidia video card, CUDA and pyopencl and reikna-0.6.4 are needed if you would like to use GPU computing')
    print('Change debug level NufftObj.debug: 2 for showing splitting variables, 1 for text outputs')
#     test_2D()
    gpu_flag = rfov_2D('GPU')
    if gpu_flag == 0:
        print('reconstruction accomplished using CPU')
    if gpu_flag == 1:
        print('reconstruction accomplished using GPU') 
