import numpy as np
import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, lfilter

segment_index = 1


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
# TODO: make own implementation of dft
spec = np.fft.fft(data_matrix[segment_index])
spec_axis = np.arange(0, fs, fs/len(spec))
#spec_axis = np.arange(0,len(spec)//2)



if(task_detect(task_list, 3)):
    plt.plot(spec_axis[:len(spec)//2], np.abs(spec[:len(spec)//2]))
    plt.xlabel('Frequency [Hz]')
    plt.title(f'segment {segment_index} - spectral analysis')
    plt.draw()
    plt.savefig(f'out/segment_{segment_index}-spectral_analysis.png')
    plt.show()


#--------------------------------------------------------------------#
#-------------------------------TASK 4-------------------------------#
#--------------------------------------------------------------------#


freqs, t, sgr = spectrogram(data, fs, nperseg=1024, noverlap=512)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
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
freqs[0] = (np.argmax(spec[:99]) - 1)/len(spec)*fs
# freqs[0] = 860
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
#fy = np.cos(f[0]/fs * 2 * np.pi * fx) + np.cos(f[1]/fs * 2 * np.pi * fx) + np.cos(f[2]/fs * 2 * np.pi * fx) + np.cos(f[3]/fs * 2 * np.pi * fx)

for freq in freqs:
    fy += np.cos(freq/fs * 2 * np.pi * fx)

if(task_detect(task_list, 6)):
    f, t, sgr = spectrogram(fy, fs, nperseg=1024, noverlap=512)

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
    wavfile.write('audio/4cos.wav', fs, fy/50)  # TODO: level this thing


#--------------------------------------------------------------------#
#-------------------------------TASK 6-------------------------------#
#--------------------------------------------------------------------#


filt = np.full(512, 1)

filt_frame = 80

filt_frame_tmp = filt_frame/2/fs * 1024
for freq in freqs:
    filt[(int)((freq/fs * 1024) - filt_frame_tmp):(int)
         ((freq/fs * 1024) + filt_frame_tmp)] = 0

filt2 = np.append(filt, np.flip(filt))
plt.plot(spec_axis, filt2)
plt.show()

imp = np.fft.ifft(filt2)
imp = np.fft.fftshift(imp)

plt.plot(imp)
plt.show()

filtered_data = lfilter(imp, [1], data)
wavfile.write('audio/clean_spec.wav', fs, np.real(filtered_data))


plt.plot(filtered_data)
plt.show()
