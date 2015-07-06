import sys
sys.path.append('..')
sys.path.append('CsTransform/utils') 
from CsTransform.utils.utils import *
sys.path.append('CsTransform') 
from CsTransform.nufft import *
from CsTransform.pynufft import *
def test_radial():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
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
    import KspaceDesign.Radial

    nblade =  180
    theta=  68.754 #
    propObj=KspaceDesign.Radial.Radial( N ,  nblade,   theta,   -1)
    om = propObj.om
    # load k-space points
#     om = numpy.loadtxt('om.txt')
#     om = numpy.loadtxt('om.gold')
     
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd)
#     NufftObj = pynufft(om, (256,256),(512,512),Jd) 
#     precondition = 1
    factor = 0.1
    NufftObj.factor = factor
    NufftObj.nufft_type = 1
#     NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    LMBD =1
    nInner=2
    nBreg =25
    
    mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1
    NufftObj = pynufft(om, (96,96),(512,512),Jd,n_shift=(-64,0))
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    NufftObj.beta=0.1
    NufftObj.initialize_gpu()
    image_blur = NufftObj.backward2(data)  
    
     
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
    image_recon = Normalize(numpy.real(image_recon))#*  1.1# *1.3
    image_blur=Normalize(numpy.real(image_blur[...,0]))#*1.15
    print(numpy.shape(image_recon),numpy.shape(image_blur))
    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()
 

    # display images

    
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.imshow((image_recon).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('Total variation')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.imshow(numpy.real(image_blur).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('density compensation')  

    matplotlib.pyplot.show()
    


if __name__ == '__main__':
    import cProfile

    test_radial()