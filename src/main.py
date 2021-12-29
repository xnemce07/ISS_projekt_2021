from math import pi
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import cmath as cm


segment_index = 4


def task_detect(task_list, task_no):
    if(task_list == ''):
        return False
    if(task_list == 'a' or task_list == 'all'):
        return True
    arr = task_list.split(',')
    task_no_str = str(task_no)
    for task in arr:
        if(task_no_str == task):
            return True
    return False

def myDFTmat(N):
    Wn = cm.exp(-1j* 2 * pi / N )
    DFTmat = np.zeros((N,N), dtype='complex')
    for n in range(0,N-1):
        Wnn = pow(Wn,n)
        for k in range(0,n):
            res = pow(Wnn,k)
            DFTmat[n][k] = res
            DFTmat[k][n] = res
    return DFTmat

def myDFT(x):
    mat = myDFTmat(len(x))
    return np.matmul(mat,x)





print("Choose task numbers separated by ',' or type 'all' for everything")
task_list = input()

#--------------------------------------------------------------------#
#-------------------------------TASK 1-------------------------------#
#--------------------------------------------------------------------#

fs, data = wavfile.read('audio/xnemce07.wav')
length_s = len(data)/fs
x_axis_whole = np.arange(0, length_s, length_s/len(data))


if(task_detect(task_list, 1)):
    print(f'SAMPLE RATE: {fs}')
    print(f'LENGTH IN SAMPLES: {len(data)}')
    print(f'LENGTH IN SECONDS: {length_s}')
    print(f'MAX VALUE: {max(data)}')
    print(f'MIN VALUE: {min(data)}')
    plt.plot(x_axis_whole, data)
    plt.xlabel('Seconds')
    plt.title('xnemce07.wav')
    plt.draw()
    plt.savefig('out/plot-xnemce07.png')
    plt.show()

#--------------------------------------------------------------------#
#-------------------------------TASK 2-------------------------------#
#--------------------------------------------------------------------#

data = data-np.mean(data)
data = data / max(np.abs(data))

if(task_detect(task_list, 2)):
    print(f'MEAN: {np.mean(data)}')
    plt.plot(x_axis_whole, data)
    plt.ylim(-1, 1)
    plt.xlabel('Seconds')
    plt.title('xnemce07.wav - normalized')
    plt.savefig('out/plot_normalized-xnemce07.png')
    plt.draw()
    plt.show()

n = 1024  # group size
m = 512  # overlap size
data_matrix = [data[i:i+n] for i in range(0, len(data), n-m)]


segment_length_s = 1024/fs
x_axis_segment = np.arange(0, segment_length_s, segment_length_s/1024)

if(task_detect(task_list, 2)):
    plt.plot(x_axis_segment, data_matrix[segment_index])
    plt.ylim(-1, 1)
    plt.xlabel('Time [s]')
    plt.title(f'xnemce07.wav - segment {segment_index}')
    plt.savefig(f'out/plot_segment_{segment_index}-xnemce07.png')
    plt.draw()
    plt.show()


#--------------------------------------------------------------------#
#-------------------------------TASK 3-------------------------------#
#--------------------------------------------------------------------#
spec = np.fft.fft(data_matrix[segment_index])
spec_axis = np.arange(0, fs, fs/len(spec))

my_spec = myDFT(data_matrix[segment_index])

if(task_detect(task_list, 3)):
    plt.plot(spec_axis[:len(spec)//2], np.abs(spec[:len(spec)//2]))
    plt.xlabel('Frequency [Hz]')
    plt.title(f'segment {segment_index} - spectral analysis')
    plt.draw()
    plt.savefig(f'out/segment_{segment_index}-spectral_analysis.png')
    plt.show()

    plt.plot(spec_axis[:len(spec)//2], np.abs(my_spec[:len(spec)//2]))
    plt.xlabel('Frequency [Hz]')
    plt.title(f'MY segment {segment_index} - My spectral analysis')
    plt.draw()
    plt.savefig(f'out/segment_{segment_index}-spectral_analysis_mine.png')
    plt.show()

#--------------------------------------------------------------------#
#-------------------------------TASK 4-------------------------------#
#--------------------------------------------------------------------#


freqs, t, sgr = signal.spectrogram(data, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20)

if(task_detect(task_list, 4)):
    plt.pcolormesh(t, freqs, sgr_log)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spectral power density [dB]', rotation=270, labelpad=15)
    plt.draw()
    plt.savefig('out/spectral_power_density.png')
    plt.show()
#--------------------------------------------------------------------#
#-------------------------------TASK 5-------------------------------#
#--------------------------------------------------------------------#

freqs = [0, 0, 0, 0]
freqs[0] = 860
freqs[1] = freqs[0] * 2
freqs[2] = freqs[0] * 3
freqs[3] = freqs[0] * 4


if(task_detect(task_list, 5)):
    for i in range(len(freqs)):
        print(f'f{i} = {freqs[i]}')


#--------------------------------------------------------------------#
#-------------------------------TASK 6-------------------------------#
#--------------------------------------------------------------------#

fx = np.arange(0, len(data), 1)
fy = 0

for freq in freqs:
    fy += np.cos(freq/fs * 2 * np.pi * fx)

if(task_detect(task_list, 6)):
    f, t, sgr = signal.spectrogram(fy, fs, nperseg=1024, noverlap=512)

    sgr_log = 10 * np.log10(sgr+1e-20)
    plt.pcolormesh(t, f, sgr_log)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label(
        'Spectral power density - 4 cos [dB]', rotation=270, labelpad=15)
    plt.draw()
    plt.savefig('out/4cos_spectral_power_density.png')
    plt.show()
    wavfile.write('audio/4cos.wav', fs, fy/50)


#--------------------------------------------------------------------#
#-------------------------------TASK 7-------------------------------#
#--------------------------------------------------------------------#

wp_segment = 50
ws_segment = 5

filtered_data = data
filt_a = 1
filt_b = 1

for freq in freqs:
    N, wn = signal.buttord([freq-wp_segment, freq+wp_segment],
                           [freq-ws_segment, freq+ws_segment], 3, 40, False, fs)
    b, a = signal.butter(N, wn, 'bandstop', False, 'ba', fs)
    filt_a = np.convolve(filt_a, a)
    filt_b = np.convolve(filt_b, b)


#--------------------------------------------------------------------#
#-------------------------------TASK 8-------------------------------#
#--------------------------------------------------------------------#

if(task_detect(task_list,8)):
    z, p, k = signal.tf2zpk(filt_b, filt_a)

    plt.figure(figsize=(4,3.5))


    ang = np.linspace(0, 2*np.pi,100)
    plt.plot(np.cos(ang), np.sin(ang))


    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='zeroes')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='ples')

    plt.gca().set_xlabel('Real part $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginary part $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.draw()
    plt.savefig('out/filter-zeroaes_and_poles.png')
    plt.show()

#--------------------------------------------------------------------#
#-------------------------------TASK 9-------------------------------#
#--------------------------------------------------------------------#


if(task_detect(task_list,9)):
    w, H = signal.freqz(filt_b, filt_a)
    _, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_title('Modulus of filter frequency response $|H(e^{j\omega})|$')

    ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_title('Argument of filter frequency response $\mathrm{arg}\ H(e^{j\omega})$')

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.draw()
    plt.savefig('out/filter_frequency_response.png')
    plt.show()

#--------------------------------------------------------------------#
#-------------------------------TASK 10------------------------------#
#--------------------------------------------------------------------#

if(task_detect(task_list,10)):
    filtered_data = signal.filtfilt(filt_b, filt_a, filtered_data)
    wavfile.write('audio/clean_bandstop.wav', fs, np.real(filtered_data))
    plt.plot(filtered_data)
    plt.draw()
    plt.savefig('out/clean_signal.png')
    plt.show()
