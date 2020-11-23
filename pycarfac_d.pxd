'''
Declaration file for any pure C functions or structures defined in ccarfac.c and
ccarfac.h
'''

cdef extern from "carfac.h":
    ctypedef struct bm_parameters_s:
        int num_sections
        double x_lo
        double x_hi
        double damping


    ctypedef struct ihc_parameters_s:
        double hpf_cutoff
        double tau_in
        double tau_out
        double tau_ihc

    ctypedef struct ohc_parameters_s:
        double scale
        double offset
        double * b

    ctypedef struct sai_parameters_s:
        double trig_win_t 
        double adv_t
        int num_trig_win
        int num_sections
        int num_samples

    ctypedef struct sai_s:
        int num_frames
        int num_sections
        int frame_len_n
        double * images
        double * delay_t
        double * times

    ctypedef struct carfacagc_state_s:
        int num_sections
        int block_size
        # derived parameters from BM settings
        double * f
        double * a0
        double * c0
        double * r
        double * r1
        double * h
        double * g
        # derived parameters from IHC settings
        double q
        double c_in
        double c_out
        double c_ihc
        # derived parameters from OHC settings
        double scale
        double offset
        double * b
        double * d_rz
        # AGC parameters
        double * c_agc
        double * sa
        double * sb
        double * sc
        # state parameters
        double * bm
        double * bm_hpf
        double * ihc_out
        double * ihc_state
        double * w0
        double * w1
        double * w1_old
        double * trans
        double * acc8
        double * acc16
        double * acc32
        double * acc64
        double * agc
        double * agc0
        double * agc1
        double * agc2
        double * agc3
        # various other internal variables (so processors doesn't have to keep allocating)
        double * ihc_new
        double * z
        double * v_mem
        double * v_ohc
        double * sqrd
        double * nlf
        double prev
        double w0_new
    

    carfacagc_state_s * carfacagc_init(bm_parameters_s *bmp,
                                       ihc_parameters_s *ihcp,
                                       ohc_parameters_s *ohcp,
                                       int block_size, 
                                       double fs)

    int carfacagc_free(carfacagc_state_s * cs)

    int carfacagc_free_except_signal(carfacagc_state_s * cs)

    int carfacagc_process_block(carfacagc_state_s * cs, double * sig)

    sai_s * sai_init(int num_frames, int num_sections, int frame_len_n)

    int sai_free(sai_s * sai)

    int sai_free_except_images(sai_s * sai)

    sai_s * sai_generate(double * nap,
                         double fs, 
                         sai_parameters_s * sai_params)
