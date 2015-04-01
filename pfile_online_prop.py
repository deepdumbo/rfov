import numpy
import argparse
import binascii
from CsTransform.pynufft import *
from CsTransform.nufft import *
import scipy.fftpack
import GeRaw.dicom_generate_uid
# from pynufft import *
import pyfftw
# import reikna
# import os
# import numpy
# import string
import commands
#     return ''.join([str(i-1) if i < 11 else '.' for pair in [(ord(c) >> 4, ord(c) & 15) for c in uid] for i in pair if i > 0])
      
# def remove_leading_zeros(input_x):
# #     output = input
#     indx = 0
#     while input_x[indx] == '0':
#         indx =indx+1
#     output = input_x[indx:]
#     
#     return output
# 
# def guid_to_uid(ipt_root,guid_32bit):
#     guid = ''
# #     print('guid_32bit size',numpy.size(guid_32bit))
#     for jj in range(0,len(guid_32bit)):
#         tmp="%010.0f"%(float(guid_32bit[jj]))#.astype(numpy.float128))
#         guid=guid+tmp
# #         print(guid)
#     
#     guid = guid[:12] +guid[13:]
#     guid = remove_leading_zeros(guid)
#     guid = ipt_root + '.2.' + guid
#     
#     return guid
# 
# def dicom_generate_uid(uid_type):
#     '''
#     translate from matlab's dicom_generate_uid
#     only linux is supported
#     
#     '''
#     cmd = 'cat /proc/sys/kernel/random/uuid'
# #     t=os.system(cmd)
#     u = commands.getoutput(cmd)
# #     u = 'ab8ae6cb-0ae4-4f04-ac0a-26200f7a35c0'
# #     print('systemuuid=',u)    
#     u=u.replace('-','')
# #     guid_32bit= numpy.empty((4))
#     guid_32bit = ()
# 
#     guid_32bit = guid_32bit +(int(u[0:8],16),)
#     print((u[0:8]))
#     print("%010.0f"%int(u[0:8],16))
#     print("%010.0f"%float(int(u[0:8],16)))
#     guid_32bit = guid_32bit +(int(u[8:16],16),)
#     guid_32bit = guid_32bit +(int(u[16:24],16),)
#     guid_32bit = guid_32bit +(int(u[24:32],16),)
# 
# #     print(guid_32bit)
# #     print('systemuuid=',u)
# 
# #     ipt_root='1.2.840.113619.2.5'
#     ipt_root = '1.3.6.1.4.1.9590.100.1'
#     guid = guid_to_uid(ipt_root,guid_32bit)
# 
# #     if uid_type == 'instance':
# #         pass
# #     elif uid_type == 'series':
# #         pass
# #     elif uid_type == 'study':
# #         pass
#     return guid
def zero_padding(input_x,size):
#     out_image = input_x
    xres,yres = numpy.shape(input_x)
    if xres > size:
        print('xres > size! Increase size')
    if yres > size:
        print('yres > size! Increase size')
          
    out_x = numpy.zeros((size,size),dtype = numpy.complex64)
     
    input_k= (scipy.fftpack.fft2((input_x)))
     
#     matplotlib.pyplot.imshow((input_k.real), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=200.0))
#     matplotlib.pyplot.show()     
#     out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
#     out_x[-xres/2:, 0:yres/2] = input_k[-xres/2, 0:yres/2] 
#     out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
#     out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]    
    out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
    out_x[-xres/2:, 0:yres/2] = input_k[-xres/2:, 0:yres/2]  
    out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
    out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]  
 
    out_image= scipy.fftpack.ifft2(out_x)*(size*1.0/xres)*(size*1.0/yres)
#     matplotlib.pyplot.imshow(numpy.abs(out_image), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=100.0))
#     matplotlib.pyplot.show()        
#     matplotlib.pyplot.imshow(numpy.abs(out_x), cmap=cmap)
#     matplotlib.pyplot.show()
    return out_image
def foo5(data_filename):
    import GeRaw.pfileparser
    import numpy,matplotlib.pyplot
    cm=matplotlib.cm.gray
    import scipy.fftpack
    import KspaceDesign.Propeller
   
    mu= 1.0
    #f=numpy.reshape(numpy.append(X.T,X.T),(st['M'],2),order='F')
    
    LMBD = 1.0
    gamma = 0.001
    nInner= 2
    nBreg=25
    precondition = 1 # best
    J = 6
#     ft_shift_phase=0 
    
    
#     propObj_32=GeRaw.pfileparser.propBinary(args.data_filename,
#                                      0,  (xres,coil_num,etl,blade))
    propObj_32 = GeRaw.pfileparser.AddenbrookesProp(data_filename)
    
    xres = propObj_32.xres
    etl = propObj_32.etl
    blade= propObj_32.nblades
    rot_angle = propObj_32.theta
    nslices=propObj_32.nslices
    sense_fact = 1.0
    coil_num = propObj_32.ncoils    
    clockwiseness = rot_angle/numpy.abs(rot_angle)
#     propObj_32.sortProp((xres,etl,blade,coil_num))
    
    if numpy.mod(coil_num, 1 ) != 0:
        print('coil number is impossible, \n')
        print('Data size is not integer times of image dimension, \n')
        print('Please check filename\n')
    else:
        print('number of complex numbers is', coil_num)  
            

    print('rhrawsize of pfile = ',)
    print('rhrawsize of guess = ',propObj_32.rhrawsize)

    if propObj_32.prop_encode == 3:
        ck32=propObj_32.prop
    else:
        ck32=propObj_32.prop#[:,0,:]

    
    Jd=[J,J]
    Nd=[xres,xres]
#     Nd = [256,256]
#     Nd = [192,192]
    Kd=[xres*2,xres*2]
#     Kd = [384,384]
    n_shift = [0, 0 ]
#     n_shift=[64,-64]#[xres/2, xres/2]
#     n_shift = [-64,0]
    import CsTransform.pynufft 

    ck32 = numpy.reshape(ck32,(numpy.size(ck32[...,0,0]), coil_num, nslices),order='F')

#     print('sys.getsizeof',sys.getsizeof(NufftObj_1.st))
    
    
    outmat = numpy.empty((xres,xres,nslices,),dtype = numpy.complex64)
    outmat2 = numpy.empty((xres,xres,nslices,),dtype = numpy.complex64) # density-compensation
    
    # create the NUFFT kernel and reconstruction


    mythread = 1#multiprocessing.cpu_count()/4
         
#     ceil_times =  nslices
  
              
#     import pp  
#     job_server = pp.Server()
#     jobs = []  
#     jobs2 = []   
    import dispatch_nufft
    
    # execute reconstruction for parallel slices
    outmat, outmat2 = dispatch_nufft.dispatch_nufft(nslices,mythread,xres, etl, blade, propObj_32, 
                   sense_fact, clockwiseness,Nd,Kd,Jd,n_shift,precondition,
                   ck32,mu, LMBD, gamma, nInner, nBreg,outmat, outmat2,
                   )
    
    
#         del NufftObj_1
#         del propObj

#         job_server.wait()             
#         job_server.destroy()   
           
#     import pp  
# #     job_server = pp.Server()
#     jobs = []  
#     jobs2 = []    
#     for tt in xrange(0,ceil_times):
#         NufftObj_1 =()
#         propObj = ()
# #         job_server = pp.Server()
#         job_server = pp.Server()
# #         jobs = []       
#              
# #         if ceil_times == floor_times:
#         if tt < ceil_times -1:
#             end_slice = mythread
#         elif tt == ceil_times -1:
#             end_slice =  nslices - (ceil_times -1)*mythread
#      
#         for uu in range(0,end_slice):
#                         
#             real_slice = tt * mythread + uu
#             
#             propObj=propObj+(KspaceDesign.Propeller.Propeller(xres, etl,blade, propObj_32.result_angle[real_slice], sense_fact, clockwiseness),)
#                 
#             om=propObj[uu].om    
#             NufftObj_1=NufftObj_1  + (CsTransform.pynufft.pynufft(om,Nd,Kd,Jd,n_shift=n_shift ), )   #,Kd,n_shift)
#             NufftObj_1[uu].precondition = precondition
# #             NufftObj_1[uu].initialize_gpu() # tricky, parallel python does not populate any attribute about GPU computing  
#             
#             
#             jobs.append(job_server.submit(NufftObj_1[uu].pseudoinverse, (ck32[...,real_slice], mu, LMBD, gamma, nInner, nBreg),
# #                             modules = ('pyfftw','numpy','pynufft','scipy','reikna'),
#                              globals = globals()))
#             jobs2.append(job_server.submit(NufftObj_1[uu].pseudoinverse2, (ck32[...,real_slice], ),
# #                  modules = ('pyfftw','numpy','pynufft','scipy','reikna'),
#                  globals = globals()))
# #             del NufftObj_1
# #         job_server.wait()
#         import scipy.misc
#         for uu in range(0,end_slice):
#             real_slice = tt * mythread + uu
#             tmp_imag = jobs[real_slice]()
#             tmp_imag2 = jobs2[real_slice]()
#             outmat[...,real_slice].real = scipy.misc.imresize(numpy.real(tmp_imag),(xres,xres))#[::-1,::-1].T
#             outmat[...,real_slice].imag = scipy.misc.imresize(numpy.imag(tmp_imag.imag),(xres,xres))
#             outmat2[...,real_slice].real = scipy.misc.imresize(numpy.real(tmp_imag2),(xres,xres))#[::-1,::-1].T
#             outmat2[...,real_slice].imag = scipy.misc.imresize(numpy.imag(tmp_imag2),(xres,xres))            
#         del NufftObj_1
#         del propObj
# 
# #         job_server.wait()             
#         job_server.destroy()   
    print('outmat.shape 234234',outmat.shape)
    outmat=outmat/numpy.max(numpy.abs(outmat[:]))
    outmat2=outmat2/numpy.max(numpy.abs(outmat2[:]))
    import scipy.io
    import string
    import os
    import re
    pattern = 'P[0-9]{5}.7'
    regexp = re.compile(pattern)
    pfilename = regexp.search(data_filename)
    print('pfilename = ', pfilename.group())
    script_dir = os.path.dirname('/home/recon/pfiles/')
    print('data_filename',data_filename)
    rel_path = pfilename.group()+'.matlist.txt'
    abs_file_path = os.path.join(script_dir, rel_path)
    text_file = open(abs_file_path,'w')
  
    return ( propObj_32,outmat, outmat2) 

def load_fseprop_parm(filename): # load parameter file
    try:
        data = numpy.loadtxt(filename)
    except:
        raise IOError('Cannot find file!'+filename)

#    if success:
    return data
def check_fseprop_parm(fseprop_parm):
#     check the validaty of fseprop_parm
    if numpy.size(fseprop_parm) != 5:
        print('fseprop.parm should has 5 variables!')
        raise 
    
#   check coil number
#    if numpy.mod(fseprop_parm[0],1) == 0: #  integers?
#        print('coil number is correct!')
#    else:
#        raise ValueError('coil number is not integer!')
        
#    check xres dimension of x
    if numpy.mod(fseprop_parm[0],1) == 0: #  integers?
        print('xres is correct!')
    else:
        raise TypeError('xres is not integer!')        

#    check ETL
    if numpy.mod(fseprop_parm[1],1) == 0: #  integers?
        print('ETL is correct!')
    else:
        raise ValueError('ETL is not integer!')  
    
#    check blades number
    if numpy.mod(fseprop_parm[2],1) == 0: #  integers?
        print('blade number is correct!')
    else:
        raise ValueError('blade number is not integer!')      

#    check rotation angle
    if numpy.abs(fseprop_parm[3]*fseprop_parm[2]-180.0)<0.01: #  integers?
        print('rotation angle is '+ str(fseprop_parm[3])+'=180.0/'+str(fseprop_parm[2]))
    else:
        print('rotation angle is not optimal!') 
#    check sense factor
    if numpy.mod(fseprop_parm[4],1) == 0 and fseprop_parm[4]>=1: #  integers?
        print('sense factor is correct!')
    else:
        raise ValueError('sense factor is not positive integer!')  
#    check clockwiseness
#    if numpy.mod(fseprop_parm[6],1) == 0 and fseprop_parm[6]*fseprop_parm[6]==1: #  integers?
#        print('clockwiseness is correct!')
#    else:
#        raise ValueError('clockwiseness is not positive integer!')  

def check_fseprop_ctrl(fseprop_ctrl):

#     check the validaty of fseprop_parm
    if numpy.size(fseprop_ctrl) != 7:
        print('fseprop.ctrl should has 5 variables!')
        raise 
    
#   check mu
    if fseprop_ctrl[0] == 1: #  integers?
        print('mu is correct!')
    else:
        raise ValueError('mu is not 1.0!')
        
#    check LMBD 
    if fseprop_ctrl[1] <1: #  integers?
        print('LMBD is optimal')
    else:
        raise ValueError('LMBD is not optimal!')        

#    check gamma
    if fseprop_ctrl[1] <1: #  integers?
        print('gamma is optimal')
    else:
        raise ValueError('gamma is not optimal!')   
    
#    check inner iteration number of split-bregman method
    if numpy.mod(fseprop_ctrl[3],1) == 0 and fseprop_ctrl[3]>=1: #  integers?
        print('nInner is correct!')
    else:
        raise ValueError('nInner is not integer!')      

#    check outer iteration number of split-bregman method
    if numpy.mod(fseprop_ctrl[4],1) == 0 and fseprop_ctrl[4]>=1: #  integers?
        print('nBreg is correct!')
    else:
        print('nBreg is correct!') 

#    check interpolator number J
    if numpy.mod(fseprop_ctrl[5],1) == 0: #  integers?
        print('interpolator number is correct!')
    else:
        raise ValueError('interpolator number is not integer!')   

    if fseprop_ctrl[6] == 0: #  integers?
        print('shift is turn-off!')
    elif fseprop_ctrl[6] == 1: #  integers?
        print('shift is turn-on!')
    else:
        raise ValueError('shift is not 0 or 1!')

def accept_inputs():
    help_text='Doing 2D Propeller MRI recon on Bob'
    
    sign_off='author:Jyh-Miin Lin, jyhmiinlin@gmail.com or jml86@cam.ac.uk'

    
    parser=argparse.ArgumentParser(description=help_text, epilog=sign_off)

    parser.add_argument('--data', '-d', dest='data_filename', action='store',metavar='data_name', help='raw data filename')
#     parser.add_argument('--temp', '-t', dest='template_dicom', action='store',metavar='template_dicom', help='dicom filename')
#     parser.add_argument('--parameter', '-p', dest='parameter_filename', action='store',metavar='parm_name', default='fseprop.parm',help='image parameter filename')
#     parser.add_argument('--control', '-c', dest='control_filename', action='store',metavar='ctrl_name', default='fseprop.ctrl',help='PROPELLER reconstruction control file')
    
    return parser.parse_args()

def check_args(args): # check arguments
       
    if args.data_filename == None:
        raise NameError('Raw data must be given!')
    print('data file = '+ args.data_filename)   
#     print('control file = '+ args.control_filename)
#     print('parameter file = '+ args.parameter_filename)    

def dicom_change_time(dataObj,plan):
    ##############################change date/time #####################
    plan.SeriesDate = str(1900+int(dataObj.hdr['rdb']['scan_date'][6:]))+dataObj.hdr['rdb']['scan_date'][0:2]+dataObj.hdr['rdb']['scan_date'][3:5]
    plan.StudyDate = plan.SeriesDate
    plan.AcquisitionDate = plan.SeriesDate
    plan.ContentDate = plan.SeriesDate
    
    plan.SeriesTime = dataObj.hdr['rdb']['scan_time'][0:2]+dataObj.hdr['rdb']['scan_time'][3:5]+'00'
    plan.StudyTime = plan.SeriesTime
    plan.AcquisitionTime = plan.SeriesTime
    plan.ContentTime = plan.SeriesTime    
    return (dataObj,plan)
def dicom_change_station(dataObj, plan):
    plan[0x00081010].value = dataObj.hdr['exam']['ex_sysid'] # MXMR
    plan[0x00091002].value = dataObj.hdr['exam']['ex_suid'] # MX
    plan.InstitutionName = dataObj.hdr['exam']['hospname'] # hospital name
    return (dataObj, plan)
def dicom_change_sequence(dataObj, plan):
    plan[0x0008103e].value = dataObj.hdr['series']['se_desc'] # description
    plan[0x0019109c].value = (dataObj.hdr['image']['psdname']+'            ')[0:12] # sequence name
    plan[0x0019109e].value =  (dataObj.hdr['image']['psd_iname']+'                  ')[0:18] # FSE
    plan.ProtocolName = dataObj.hdr['series']['prtcl'] # protocol
    return (dataObj, plan)
def dicom_change_geometry(dataObj, plan, run_slice ):
    if dataObj.hdr['rdb']['position'] == 0: #head first-supine
        plan.PatientPosition ='HFS'
        
        
    if dataObj.hdr['image']['swappf'] ==0:  # swap phase frequency
        plan[0x0019108f].value='0'
    else:
        plan[0x0019108f].value='1'
         
    if dataObj.hdr['image']['freq_dir'] == 1:  # frequency direction
        plan[0x00181312].value='ROW' # need to debug
    elif dataObj.hdr['image']['freq_dir'] == 2:
        plan[0x00181312].value='COL'# need to debug
            
    true_slice = dataObj.hdr['data_acq_tab'][run_slice]['slice_in_pass']
#     true_slice = run_slice +1
#     plan.InStackPositionNumber = true_slice.astype(numpy.uint32)
#     plan[0x00209057].value = true_slice.astype(numpy.uint32) #in-stock position number
    plan.InStackPositionNumber = run_slice +1
    plan[0x00209057].value = run_slice +1 #in-stock position number

    plan.ImagePositionPatient = [str(-dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][0,0]),
                                 str(-dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][1,0]),
                                 str(dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][2,0]- dataObj.hdr['rdb']['scancent'])
                                 ]
    print(str(dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][2,2]))
    slice_position = ( - dataObj.hdr['rdb']['scancent']+ 
                       dataObj.hdr['series']['start_loc'] + 
                       (run_slice +1) *dataObj.hdr['image']['slthick'] + 
                       run_slice*dataObj.hdr['image']['scanspacing'] )
    plan.SliceLocation = str(slice_position)
    
    if slice_position < 0:
        plan.PositionReferenceIndicator = 'I'
    elif slice_position >= 0:
        plan.PositionReferenceIndicator = 'S'
    
    
#     str(dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][2,2])
    import struct
    plan[0x00271041].value = struct.pack('f',slice_position)
#     str(dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][2,2])
    
    v1 =( dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][:,0] -
           dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][:,2] ) 

    v2 =     ( - dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][:,1] +
            dataObj.hdr['data_acq_tab'][run_slice ]['gw_point'][:,0] )
    
    v1 = v1/numpy.sqrt(numpy.sum(v1**2))
    v2 = v2/numpy.sqrt(numpy.sum(v2**2))
    
#     if sum(v2*[1,0,0]) >sum(v1*[1,0,0]):
#         v1,v2 = v2, v1
    if plan[0x00181312].value=='ROW': # tricky part
        plan[0x00200037].value = [ -v2[0], -v2[1], -v2[2], v1[0], v1[1], -v1[2] ]
    elif plan[0x00181312].value=='COL':
        plan[0x00200037].value = [  v1[0],  v1[1],  -v1[2], v2[0], v2[1], -v2[2] ]
    return (dataObj, plan, true_slice)

def dicom_change_image(dataObj, plan, c):
    xres = dataObj.hdr['rdb']['da_xres']
    
#     if dataObj.hdr['image']['freq_dir'] == 0: # frequency direction = A-P

#     if plan[0x0019108f].value=='0': # swap phase/frequency
#         replace_image = c[:,::-1]#.T
#     else: 
#         replace_image = c[::-1,::-1].T # yet to debug
#     print('plan[0x0019108f].value=',plan[0x0019108f].value)
    
    if plan[0x00181312].value=='ROW':
        if plan[0x0019108f].value=='0': # swap phase/frequency
            replace_image = c[:,::-1]#.T
        else: 
            replace_image = c[::-1,::-1].T # yet to debug
#     print('plan[0x0019108f].value=',plan[0x0019108f].value)
        replace_image =  replace_image[::-1,:].T
    else: # don't modify
        if plan[0x0019108f].value=='0': # swap phase/frequency
            replace_image = c[:,::-1]#.T
        else: 
            replace_image = c[::-1,::-1].T # yet to debug
#         print('plan[0x0019108f].value=',plan[0x0019108f].value)        
        pass
    
#     elif dataObj.hdr['image']['freq_dir'] == 1: # frequency direction = A-P
#         replace_image = c#[::-1,::-1].T
#     elif dataObj.hdr['image']['freq_dir'] == 2: # frequency direction = A-P
#         replace_image = c        
    if xres <= 256:
        new_xres = 256
    elif xres <=512:
        new_xres = 512
    elif xres <= 1024:
        new_xres = 1024
#     new_xres = xres   
    
#     d=scipy.misc.imresize(replace_image,(xres,xres))
    d = zero_padding(replace_image, new_xres)

    plan.Rows= new_xres
    plan.Columns = new_xres
    plan.pixel_array = numpy.zeros((new_xres,new_xres),dtype=numpy.int16)
    plan.PixelData=numpy.zeros((new_xres,new_xres),dtype=numpy.int16).tostring()

    plan.pixel_array[:] = d[:]  
    plan.PixelData = plan.pixel_array.tostring() 
    
    plan[0x7fe00000].value = 8+ 2*numpy.size(d)
    
    return (dataObj,plan)
def unpack_uid(uid):
    """Convert packed PFile UID to standard DICOM UID."""
    return ''.join([str(i-1) if i < 11 else '.' for pair in [(ord(c) >> 4, ord(c) & 15) for c in uid] for i in pair if i > 0])

def gendicom(args, dataObj, outmat,outmat2):
    try:
        import dicom
        #import dicom.tag
        import dicom.UID
        from dicom.dataset import Dataset, FileDataset
    #     from dicom.tag import Tag
    except:
        print('Cannot find pydicom;')
        print('Try to install pydicom via')
        print('"pip install -U pydicom"')
        raise
    
    import matplotlib.pyplot
    import numpy

#     xres = propObj_32.xres
#     etl = propObj_32.etl
#     blade= propObj_32.nblades
#     rot_angle = propObj_32.theta
#     nslices=dataObj.nslices
#     import scipy.io                        
#     for zz in range(0,nslices):
#         fig_mag =( 2000*numpy.abs(outmat[:,:,zz]) ).astype(numpy.int16) 
#         fig_real =( 2000*numpy.real(outmat[:,:,zz]) ).astype(numpy.int16)
#         fig_imag =( 2000*numpy.imag(outmat[:,:,zz]) ).astype(numpy.int16)
#         mat_filename = args.data_filename+'_'+str(zz+1).zfill(4)+'.mat'
#         scipy.io.savemat(mat_filename, 
#                          mdict={'fig_mag':fig_mag, 'fig_real':fig_real,'fig_imag':fig_imag  }, 
#                          oned_as={'column'})
#         text_file.write(format(pfilename.group()+'_'+str(zz+1).zfill(4)+'.mat'+'\n'))
#     text_file.close()
    print('closing file')    
    print('off_data',dataObj.hdr['rdb']['off_data'])
#     sense_fact = 1.0
#     coil_num = propObj_32.ncoils  
        
    # plan=dicom.read_file("32etl.dcm")
#     plan=dicom.read_file(args.template_dicom)
#     plan = dicom.read_file('deploy2/i479.dcm') # template
    plan = dicom.read_file('deploy2/8etl.dcm') # template
    print(plan)
#     print('plan.group length',(plan.get('PixelDataGroupLength','missing')))
#     print(plan.dir("pixel"))
#     print(plan.dir("Rows"))    
#     print(plan.dir("length"))    
#     print('plan name',plan[0x10,0x00].name)
#     print('plan value',plan[0x10,0x00].value)
#     print('plan tag',plan[0x10,0x00].tag)
#     print(plan.PatientName)
#     print(plan.dir('UID'))
#     print(plan.FrameOfReferenceUID)
#     print('plan.Rows',plan.Rows)
#     print('plan.Columns',plan.Columns)
    import scipy.misc
#     c=(plan.pixel_array)
#     import scipy.io
#     c = scipy.io.loadmat('P64512.7_0001.mat')['fig_mag']
#     c = scipy.io.loadmat(args.data_filename+'_0001.mat')['fig_mag']

    
#     plan[0x00020012].value = '1.2.840.113619.6.283' 

    plan.StudyInstanceUID = unpack_uid(dataObj.hdr['exam']['study_uid'])
#     plan.SeriesInstanceUID = unpack_uid(dataObj.hdr['series']['series_uid'])
    tmp_uid = unpack_uid(dataObj.hdr['series']['series_uid'])
 
    if tmp_uid[-2] == '.':
        plan.FrameOfReferenceUID = tmp_uid[0:-1]+str(int(tmp_uid[-1:])-dataObj.hdr['series']['se_no']+1)          
    elif tmp_uid[-3] == '.':
        plan.FrameOfReferenceUID = tmp_uid[0:-2]+str(int(tmp_uid[-2:])-dataObj.hdr['series']['se_no']+1)  
    else:#if tmp_uid[-4] == '.':
        plan.FrameOfReferenceUID = tmp_uid[0:-3]+str(int(tmp_uid[-3:])-dataObj.hdr['series']['se_no']+1)      
 
    plan[0x00081140].value[0][0x00081155].value=unpack_uid(dataObj.hdr['series']['refImgUID1'])
    plan[0x00081140].value[1][0x00081155].value=unpack_uid(dataObj.hdr['series']['refImgUID2'])
    plan[0x00081140].value[0][0x00081150].value=unpack_uid(dataObj.hdr['image']['sop_uid'])
    plan[0x00081140].value[1][0x00081150].value=unpack_uid(dataObj.hdr['image']['sop_uid'])    
#     ##############################change date/time #####################
    (dataObj,plan) = dicom_change_time(dataObj,plan)
#     plan.SeriesDate = str(1900+int(dataObj.hdr['rdb']['scan_date'][6:]))+dataObj.hdr['rdb']['scan_date'][0:2]+dataObj.hdr['rdb']['scan_date'][3:5]
#     plan.StudyDate = plan.SeriesDate
#     plan.AcquisitionDate = plan.SeriesDate
#     plan.ContentDate = plan.SeriesDate
#     
#     plan.SeriesTime = dataObj.hdr['rdb']['scan_time'][0:2]+dataObj.hdr['rdb']['scan_time'][3:5]+'00'
#     plan.StudyTime = plan.SeriesTime
#     plan.AcquisitionTime = plan.SeriesTime
#     plan.ContentTime = plan.SeriesTime
    #####################################change series description #####
    (dataObj, plan) = dicom_change_sequence(dataObj, plan)
#     plan[0x0008103e].value = dataObj.hdr['series']['se_desc']
#     ###################################sequence name/ mode ######
#     plan[0x0019109c].value = dataObj.hdr['image']['psdname'][0:12]
#     plan[0x0019109e].value =  dataObj.hdr['image']['psd_iname'][0:18]
#     plan.ProtocolName = dataObj.hdr['series']['prtcl']
    #####################################change station name e.g. MXMR ##### 
    (dataObj, plan) = dicom_change_station(dataObj, plan)  
#     plan[0x00081010].value = dataObj.hdr['exam']['ex_sysid'] # MXMR
#     plan[0x00091002].value = dataObj.hdr['exam']['ex_suid'] # MX
#     plan.InstitutionName = dataObj.hdr['exam']['hospname'] # hospital name
    #####################################change coil name #####
    plan[0x00181250].value = dataObj.hdr['image']['cname']
    ########################################change SAR ############
    
    plan[0x00181316].value = str(dataObj.hdr['image']['saravg'])
    plan[0x00191084].value =  str(dataObj.hdr['image']['sarpeak'])[0:6]

    ###########slice thickness
    plan.SliceThickness = str(dataObj.hdr['image']['slthick'])
    plan.SpacingBetweenSlices = str(dataObj.hdr['image']['scanspacing'])
    
    ###########TR ###### 
    plan.RepetitionTime = str(dataObj.hdr['image']['tr']/1000.0)
    plan.EchoTime = str(dataObj.hdr['image']['te']/1000.0)
    #############Physician ###
    plan[0x00080090].value = str(dataObj.hdr['exam']['refphy'])
    ######## patient information 
    plan[0x00101030].value = str(dataObj.hdr['exam']['patweight']/1000.0)
    plan.PatientName = dataObj.hdr['exam']['patnameff']
    plan[0x00100020].value =dataObj.hdr['exam']['patidff']
    if dataObj.hdr['exam']['patsex'] == 0:
        plan[0x00100040].value = 'M'
    elif dataObj.hdr['exam']['patsex'] == 1:
        plan[0x00100040].value = 'F' 
        

    plan[0x00100030].value = dataObj.hdr['exam']['dateofbirth']
    ####### image para
    #     print(plan[0x00280030][0])
    #     print(plan[0x00280030][1])
    xres = dataObj.hdr['rdb']['da_xres']
    
    plan[0x00280030].value =[ str(dataObj.hdr['image']['pixsize_X']),str(dataObj.hdr['image']['pixsize_Y'] )]
    plan[0x00181310].value = [0,xres, xres, 0]
    plan[0x00281050].value = int(1117)
    plan[0x00281051].value = int(2235)
    import struct 
    plan[0x00280106].value = struct.pack('h',0)
    plan[0x00280107].value = struct.pack('h',2235)
    #     ####### fov
    plan.ReconstructionDiameter = str(float(dataObj.hdr['image']['dfov'])) 
#     plan[0x0019101e].value = str(float(dataObj.hdr['image']['dfov']))[0:10] 
    #     
    plan.AcquisitionNumber = str(dataObj.hdr['image']['scanactno'])
#     plan.SeriesNumber =str( dataObj.hdr['series']['se_no'] )
    plan.StudyID =str( dataObj.hdr['exam']['ex_no'] )
    
    plan.ImagesInAcquisition = str(dataObj.nslices)
#     plan.LocationsInAcquisition = dataObj.nslices
    plan[0x0021104f].value =  str(dataObj.nslices) # locationsinacquisition
    plan.EchoTrainLength = str( dataObj.hdr['image']['echo_trn_len'])
    plan.InversionTime = str(dataObj.hdr['image']['ti'])
    plan.FlipAngle=str(90)

    ############bandwidth
   
    plan.PixelBandwidth=2000.0*dataObj.hdr['rdb']['bw']/dataObj.hdr['rdb']['da_xres']
    
    
    ############ image position
    nslices=dataObj.nslices
    import os
    import shutil
    directory_name = args.data_filename+'.sdcopen'
    try:
        shutil.rmtree(directory_name)
    except:
        pass
    
    try:
        os.makedirs(directory_name)
    except OSError,err:
        if err.errno!=17:
            raise
        
#     tmp_abs = numpy.abs(outmat[...].flatten())
# 
#     B = numpy.zeros(300, int)
#     for ii in xrange(300):
#         nonzero = tmp_abs > 0
#         idx = numpy.argmin(tmp_abs[nonzero])
#         B[ii]=tmp_abs[idx]
#         tmp_abs[idx]=0
    
#     mean_background_noise = numpy.mean(B)
#     slice_image= numpy.mean(numpy.real(outmat),-1 )
#     slice_shape = numpy.shape(slice_image)
# #         negative_ind= outmat[:,:,true_slice-1].real <= 0
#     
# #         negative_fil= slice_image*0.0
# #         negative_fil[outmat[:,:,true_slice-1].real <= 0] = slice_image[outmat[:,:,true_slice-1].real <= 0]
# #         tmp_abs1 = outmat[1, slice_shape[1]/2-5:slice_shape[1]/2+5,true_slice-1]
# #         tmp_abs2 = outmat[slice_shape[0]/2-5:slice_shape[0]/2+5, 1,true_slice-1]
# #         tmp_abs3 = outmat[-2,slice_shape[1]/2-5:slice_shape[1]/2+5,true_slice-1]
# #         tmp_abs4 = outmat[slice_shape[0]/2-5:slice_shape[0]/2+5,-2,true_slice-1]
# #          
# #         mean_background_noise = ( numpy.mean(tmp_abs1)  + 
# #                                   numpy.mean(tmp_abs2)  +
# #                                   numpy.mean(tmp_abs3)  +
# #                                   numpy.mean(tmp_abs4) ) /4.0  
# #         std_background_noise = ( ( numpy.std(tmp_abs1)**2  + 
# #                                   numpy.std(tmp_abs2)**2  +
# #                                   numpy.std(tmp_abs3)**2  +
# #                                   numpy.std(tmp_abs4)**2 ) /4.0 )**0.5                                        
# #         slice_image[outmat[:,:,true_slice-1].real <  mean_background_noise] = 0
#                
# #         slice_image = numpy.real(slice_image)
#     tmp_abs1 = slice_image[1, slice_shape[1]/2-5:slice_shape[1]/2+5]
#     tmp_abs2 = slice_image[slice_shape[0]/2-5:slice_shape[0]/2+5, 1]
#     tmp_abs3 = slice_image[-2,slice_shape[1]/2-5:slice_shape[1]/2+5]
#     tmp_abs4 = slice_image[slice_shape[0]/2-5:slice_shape[0]/2+5,-2]
#     
#     max_value = numpy.max(slice_image.flatten())
# 
#         
#     mean_background_noise = ( numpy.mean(tmp_abs1)  + 
#                               numpy.mean(tmp_abs2)  +
#                               numpy.mean(tmp_abs3)  +
#                               numpy.mean(tmp_abs4) ) /4.0
#     std_background_noise = ( ( numpy.std(tmp_abs1)**2  + 
#                               numpy.std(tmp_abs2)**2  +
#                               numpy.std(tmp_abs3)**2  +
#                               numpy.std(tmp_abs4)**2 ) /4.0 )**0.5 
#                           
# #         slice_image = ( slice_image - 
# #                         mean_background_noise-
# #                         std_background_noise)/(max_value-
# #                                                mean_background_noise-std_background_noise
# #                                                )
#     stack_image = ( numpy.real(outmat) -mean_background_noise )/(max_value - mean_background_noise)
#     
# #     slice_image[slice_image<=0] = 0    
#     
# #         negative_fil = negative_fil*(1.0 - slice_image)
# #         slice_image = slice_image +    1.4*negative_fil.real    
# #         slice_image[negative_fil > 0.80] = 0
#                        
#     stack_image = ( 2235* stack_image ).astype(numpy.uint16) 
    cmd = 'cat /proc/sys/kernel/random/uuid'
    ipt_root = '1.3.6.1.4.1.9590.100.1'
    for run_slice in range(0,nslices):
        (dataObj, plan, true_slice) = dicom_change_geometry(dataObj, plan, run_slice )
        print('true slice=',true_slice)
        print('run slice=',run_slice)
        
        tmp_uid = unpack_uid(dataObj.hdr['image']['image_uid'])
        if tmp_uid[-2] == '.':
            plan.SOPInstanceUID = tmp_uid[0:-1]+str(int(tmp_uid[-1:])+true_slice)     
            plan.file_meta[0x00020003].value = tmp_uid[0:-1]+str(int(tmp_uid[-1:])+true_slice) 
        elif tmp_uid[-3] == '.':
            plan.SOPInstanceUID = tmp_uid[0:-2]+str(int(tmp_uid[-2:])+true_slice) 
            plan.file_meta[0x00020003].value  = tmp_uid[0:-2]+str(int(tmp_uid[-2:])+true_slice)
        else:#if tmp_uid[-4] == '.':
            plan.SOPInstanceUID = tmp_uid[0:-3]+str(int(tmp_uid[-3:])+true_slice)  
            plan.file_meta[0x00020003].value  = tmp_uid[0:-3]+str(int(tmp_uid[-3:])+true_slice) 
#         plan.MediaStorageSOPInstanceUID = plan.SOPInstanceUID
        plan.InstanceNumber =  str(run_slice +1) 
        tmp_implementationUID = plan.file_meta[0x00020003].value[0:20]
        tmp_implementationUID2=tmp_implementationUID[0:15] +str(6) +tmp_implementationUID[16:20]
        plan.file_meta[0x00020012].value = tmp_implementationUID2 
        
        for xx in xrange(0,2):  # store iterative and DCF 
            if xx == 0 :
                slice_image= numpy.real(outmat[:,:,true_slice-1])
                plan.SeriesNumber =str( dataObj.hdr['series']['se_no'] )
                plan.SeriesInstanceUID = unpack_uid(dataObj.hdr['series']['series_uid'])
            else:
                uuid = commands.getoutput(cmd)
                guid = GeRaw.dicom_generate_uid.dicom_generate_uid(uuid,ipt_root)
                slice_image= numpy.real(outmat2[:,:,true_slice-1])
                plan.SeriesNumber =str( dataObj.hdr['series']['se_no']*100 )
                plan.SOPInstanceUID =guid 
                plan.file_meta[0x00020003].value =guid
                plan.SeriesInstanceUID = unpack_uid(dataObj.hdr['series']['series_uid']) + '.1' # series uid must be different
                
            slice_shape = numpy.shape(slice_image)
    
            # Image Normalization
            
#             tmp_abs1 = slice_image[1, slice_shape[1]/2-5:slice_shape[1]/2+5]
#             tmp_abs2 = slice_image[slice_shape[0]/2-5:slice_shape[0]/2+5, 1]
#             tmp_abs3 = slice_image[-2,slice_shape[1]/2-5:slice_shape[1]/2+5]
#             tmp_abs4 = slice_image[slice_shape[0]/2-5:slice_shape[0]/2+5,-2]
#              
#             max_value = numpy.max(slice_image.flatten())
#      
#                  
#             mean_background_noise = ( numpy.mean(tmp_abs1)  + 
#                                       numpy.mean(tmp_abs2)  +
#                                       numpy.mean(tmp_abs3)  +
#                                       numpy.mean(tmp_abs4) ) /4.0
            max_value = numpy.percentile( slice_image, 98)   
            mean_background_noise = numpy.percentile( slice_image, 1)                         
            std_background_noise =0.0# ( ( numpy.std(tmp_abs1)**2  + 
#                                       numpy.std(tmp_abs2)**2  +
#                                       numpy.std(tmp_abs3)**2  +
#                                       numpy.std(tmp_abs4)**2 ) /4.0 )**0.5 
                                   
            slice_image = 1.0*( slice_image - 
                            mean_background_noise-
                            std_background_noise)/(max_value-
                                                   mean_background_noise-std_background_noise )  
                            # define the default window level 1.35: magic number
    
                                
            c = ( 2235*numpy.real(slice_image) ).astype(numpy.int16)
    #         slice_image = stack_image[...,true_slice-1]
            (dataObj, plan   ) = dicom_change_image(dataObj, plan, c)
    
            plan.save_as(directory_name+'/_new'+str(run_slice +1 + xx*nslices)+'.dcm')
    
    


        
    # print('plan.PixelData',plan.PixelData )
    # print('plan.pixel_array',plan.pixel_array.tostring() )
#     print(plan)
#     ps = dicom.read_file(directory_name+'/_new'+str(run_slice+1)+'.dcm')
#     
# #     print('ps.SOPInstanceUID',ps[0x00080018].value)
#     print('ps.StudyInstanceUID',ps.StudyInstanceUID )
#     print('ps.SeriesInstanceUID',ps.SeriesInstanceUID)
#     
#     print('ps.SeriesTime',ps.SeriesTime)
#     print('ps.SeriesDate',ps.SeriesDate)
#     print('ps.SliceThickness',ps.SliceThickness)
#     print('ps.SpaceBetweenSlices',ps.SpacingBetweenSlices  )
#     print('ps.SliceLocation',ps.SliceLocation)
#     print('Repetition Time   ',ps.RepetitionTime   )
#     print('ps.Reconstruction Diameter',ps.ReconstructionDiameter )
#     
#     print(ps[0x0008,0x0060].value)
#     
#     # ps.SliceLocation = 993.1
#     print('Image location',ps[0x00271040].value)
#     print(ps[0x00201041].value)
#     print(ps[0x18, 0x50].value)
#     print('spacing ',ps[0x0018, 0x0088].value)
#     # print(ps)
#     print('InStackPositionNumber',ps[0x00201002].value)
#     dddd=ps[0x00201041].value
#     
#     print('psx027_1041',str(dddd))
    
#     matplotlib.pyplot.imshow(ps.pixel_array)
#     matplotlib.pyplot.show()
    #plan.save_as("rtplan2.dcm")     
if __name__ == "__main__":

    #load arguments
    args=accept_inputs()
    check_args(args)
    
    #load image parameters
#     fseprop_parm= load_fseprop_parm(args.parameter_filename) 
#     check_fseprop_parm(fseprop_parm) # check parameters
#     
#     # load recon parameters
#     fseprop_ctrl = load_fseprop_parm(args.control_filename)  
#     check_fseprop_ctrl(fseprop_ctrl) # check recon control
    print(args.data_filename)
#     template_dicom = '8etl.dcm'

    (dataobj,outmat,outmat2)=foo5(args.data_filename)
    print('foo5')
    
    gendicom(args, dataobj, outmat,outmat2)
    
    print('success gendicom ?')
#     foo5('fse_prop_slquant2_bl16_etl16.pfile')
#     foo5('GeRaw/pfile_raw_data_fse_bl16_etl16.pfile')
#    cProfile.run("testSparse2DStack()")
#     cProfile.run("foo_prime()")
#    testSparse2DStack()