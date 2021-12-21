import numpy as np
import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt

frame_index = 1

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

samplerate, data = wavfile.read('audio/xnemce07.wav')
length_s = len(data)/samplerate
x_axis_whole = np.arange(0, length_s, length_s/len(data))


if(task_detect(task_list, 1)):
    print(f'SAMPLE RATE: {samplerate}')
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
    plt.ylim(-1,1)
    plt.xlabel('Seconds')
    plt.title('xnemce07.wav - normalized')
    plt.savefig('out/plot_normalized-xnemce07.png')
    plt.draw()
    plt.show()

n = 1024  # group size
m = 512  # overlap size
data_matrix = [data[i:i+n] for i in range(0, len(data), n-m)]

print(f'MATRIX LEN: {len(data_matrix)}')
print(f'PART LEN: {len(data_matrix[1])}')
frame_length_s = 1024/samplerate
x_axis_frame = np.arange(0, frame_length_s, frame_length_s/1024)

if(task_detect(task_list, 2)):
    plt.plot(x_axis_frame, data_matrix[frame_index])
    plt.ylim(-1,1)
    plt.xlabel('Seconds')
    plt.title(f'xnemce07.wav - frame {frame_index}')
    plt.savefig(f'out/plot_frame_{frame_index}-xnemce07.png')
    plt.draw()
    plt.show()