import numpy as np
cimport numpy as cnp
cimport cython
cimport pycarfac_d as pyc
from libc.stdlib cimport malloc, free

################################################################################
# Unitility functions (dealing with memory, etc.)
################################################################################
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)


################################################################################
# Wrapped C Functions
################################################################################
cpdef carfac_nap(double [:] signal, 
                 double fs,
                 int num_sections = 72,
                 double x_lo = 0.1,
                 double x_hi = 0.9,
                 double damping = 0.2,
                 double hpf_cutoff = 20.0,
                 double tau_in = 10.0e-3,
                 double tau_out = 0.5e-3,
                 double tau_ihc = 80.0e-6,
                 double scale = 0.1,
                 double offset = 0.04,
                 double b = 1.0):
    '''
    Generates a neural activity pattern (NAP) from an input signal. Uses Richard
    Lyon's CARFAC-AGC model.

    At some point should modify so that non-uniform b values can be passed as a
    vector.
    '''
    # set up data structures
    cdef pyc.carfacagc_state_s *car_data
    cdef pyc.bm_parameters_s bmp
    cdef pyc.ihc_parameters_s ihcp
    cdef pyc.ohc_parameters_s ohcp
    cdef int block_size = len(signal)

    # set up parameters
    cdef cnp.ndarray[double, ndim=1, mode='c'] b_vals_np
    cdef double * b_vals
    b_vals_np = np.ndarray((num_sections), buffer=np.ones(num_sections)*b,
            dtype=np.double, order='C')
    b_vals = &b_vals_np[0]

    
    bmp.num_sections = num_sections
    bmp.x_lo = x_lo
    bmp.x_hi = x_hi
    bmp.damping = damping

    ihcp.hpf_cutoff = hpf_cutoff 
    ihcp.tau_in = tau_in
    ihcp.tau_out = tau_out
    ihcp.tau_ihc = tau_ihc

    ohcp.scale = scale
    ohcp.offset = offset
    ohcp.b = b_vals  # make sure this doesn't cause trouble!!

    # initialize and process CARFAC
    car_data = pyc.carfacagc_init( &bmp, &ihcp, &ohcp, block_size, fs)
    success = pyc.carfacagc_process_block(car_data, &signal[0]) 
    if success != 0:
        print("Failed to process signal through CARFAC-AGC.")

    # wrangle data into numpy array
    cdef int num_channels = car_data.num_sections
    cdef int num_points = car_data.block_size

    cdef double *out = car_data.ihc_out
    cdef cnp.ndarray output_np = np.asarray(<double[:num_channels, :num_points]>out)
    PyArray_ENABLEFLAGS(output_np, cnp.NPY_OWNDATA)

    cdef double *f_vals = car_data.f
    cdef cnp.ndarray f_vals_np = np.asarray(<double[:num_channels]>f_vals)
    cdef cnp.ndarray f_vals_np_cp = f_vals_np.copy()
    # PyArray_ENABLEFLAGS(f_vals_np, cnp.NPY_OWNDATA)

    # clean up memory and get out
    pyc.carfacagc_free_except_signal(car_data)
    return output_np, f_vals_np_cp


cpdef carfac_sai(double[:,::1] nap,
                 double fs, 
                 double trig_win_t = 0.010,
                 double adv_t = 0.005,
                 int num_trig_win = 4):
    '''
    Generates the frames stabilized auditory image (SAI) animation form a neural
    activity pattern (NAP).
    '''
    
    ## Set up parameters
    cdef pyc.sai_parameters_s saip
    cdef int num_sections = len(nap)
    cdef int num_samples = len(nap[0])
    saip.trig_win_t = trig_win_t
    saip.adv_t = adv_t
    saip.num_trig_win = num_trig_win
    saip.num_sections = num_sections
    saip.num_samples = num_samples


    ## Generate the SAI frames
    cdef pyc.sai_s * sai
    sai = pyc.sai_generate(&nap[0,0], fs, & saip)
    cdef int num_frames = sai.num_frames
    cdef int frame_len_n = sai.frame_len_n


    ## Data ownership shuffling, etc.

    # SAI data
    cdef double * I     # sai images
    I = sai.images
    cdef cnp.ndarray I_np = np.asarray(<double[:num_frames, :num_sections,
        :frame_len_n]>I)
    PyArray_ENABLEFLAGS(I_np, cnp.NPY_OWNDATA)

    # Frame time data
    cdef double * t     # times for each frame 
    t = sai.times
    cdef cnp.ndarray t_np = np.asarray(<double[:num_frames]>t)
    cdef cnp.ndarray t_np_cp = t_np.copy()
    # PyArray_ENABLEFLAGS(t_np, cnp.NPY_OWNDATA)

    # Delay time data
    cdef double * delay_t
    delay_t = sai.delay_t
    cdef cnp.ndarray delay_t_np = np.asarray(<double[:frame_len_n]>delay_t)
    cdef cnp.ndarray delay_t_np_cp = delay_t_np.copy()
    # PyArray_ENABLEFLAGS(delay_t_np, cnp.NPY_OWNDATA)


    ## Free memory and return values
    pyc.sai_free_except_images(sai)

    return I_np, t_np_cp, delay_t_np_cp
    
