import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import pycarfac as pyc

################################################################################
# Animation Utilities
################################################################################
class MyNormalize(mcolors.Normalize):
    def __call__(self, value, clip=None):
        f = lambda x,a: (2*x)**a*(2*x<1)/2. +(2-(2*(1-1*x))**a)*(2*x>=1)/2.
        return np.ma.masked_array(f(value,0.5))

def animate_SAI(I, fs, times, colormap = cm.binary, adv_time = 50):
    num_frames = I.shape[0]
    num_sect = I.shape[1]
    frame_len_n = I.shape[2]
    delays = np.flip(np.arange(frame_len_n)/fs,0)

    # Set up the figure
    fig = plt.figure()
    im = plt.imshow(np.flip(I[0,:,:],0),cmap=colormap,
            norm=MyNormalize(vmin=-0.3, vmax=1.2), origin='lower', aspect='auto')
    # text_labels = ["%.1f" % (np.flip(f,0)[k]) for k in range(num_sect)]
    # ticks = []
    # labels = []
    # for k in range(num_sect):
    #     if k % 8 == 0:
    #         ticks.append(k)
    #         labels.append(text_labels[k])
    # plt.yticks(ticks,labels)
    # plt.ylabel("Channel CF")
    adv = frame_len_n//4
    # ticks = np.arange(4)*adv
    # labels = ["%.0f" % (delays[tick]*1000) for tick in ticks]
    # plt.xticks(ticks, labels)
    # plt.xlabel('delay t (in ms)')

    # Animation update function
    def update(frame_num):
        plt.title("t=%.3f" % (times[frame_num]) )
        im.set_array(np.flip(I[frame_num,:,:],0))
        return im

    # Let 'er rip
    anim = FuncAnimation(fig, update, frames=range(1,num_frames), interval=adv_time, repeat=True)
    plt.show()
    
################################################################################
# Stimulus Generating Utilities
################################################################################
def sin_complex(freqs, dur, fs, phase=None):
    num_p = len(freqs)
    num_samps = int(dur*fs)
    if phase == None:
        phase = np.zeros(num_p)
    assert len(phase) == num_p
    t = np.arange(0,num_samps)/fs
    sig = np.zeros(num_samps)
    for (f, p) in zip(freqs, phase):
        sig += np.sin(2.0*np.pi*f*t + p)
    return sig

def damped_env(T, tau, num_samples, fs):
    t_period = np.arange(0,T,1/fs) 
    period = np.exp(-t_period/tau)
    n_period = len(period)
    env = [period[k%n_period] for k in range(num_samples)]
    return env


################################################################################
# Main Script
################################################################################
if __name__=="__main__":

    ## Signal parameters
    fs = 32000.0
    dur = 0.5
    num_samps = int(dur*fs)
    t = np.arange(0,num_samps)/fs
    num_p = 4
    f0 = 200.0
    freqs = [f0*k for k in range(1,num_p+1)]
    phase = [2*np.pi*np.random.random() for _ in range(1,num_p+1)]
    # phase = [0 for _ in range(1,num_p+1)]
    f_env = 50.0
    T_env = 1.0/f_env
    tau_env = T_env/2.0

    ## Generate Test Signal
    np.random.seed(0)
    signal = sin_complex(freqs, dur, fs, phase=phase)
    signal *= damped_env(T_env, tau_env, num_samps, fs)
    signal *= 0.75

    ## Generate NAP
    nap, channel_f_vals = pyc.carfac_nap(signal, fs, b=0.01, num_sections=40)
    num_sections = nap.shape[0]

    ## Generate SAI
    sai, frames_t, delays_t = pyc.carfac_sai(nap, fs)

    ## Plot Results
    fig = plt.figure(figsize=(9,6))
    ax = fig.get_axes()
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("CF (Hz)", fontsize=12)
    p = 0.10 # offset scaling factor
    skip_step = 4
    ytick_vals = np.arange(num_sections)*p
    ytick_vals = ytick_vals[::skip_step]
    ytick_labels = ["{:.0f}".format(f) for f in np.flip(channel_f_vals)]
    ytick_labels = ytick_labels[::skip_step]
    plt.yticks(ytick_vals, ytick_labels)
    for k in range(num_sections):
        plt.fill_between(t, 0, nap[k,:num_samps]+p*(num_sections-k), facecolor='w',
                edgecolor='k', linewidth=0.6)
    
    ## Animate results
    num_frames = sai.shape[0]
    animate_SAI(sai, fs, frames_t, colormap=cm.binary,
            adv_time=50)

    plt.show()
