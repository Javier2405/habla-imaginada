import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

column_names = ["Tiempo", "Canal 1", "Canal 2", "Canal 3", "Canal 4", "Canal 5", "Canal 6",
                            "Canal 7", "Canal 8", "Canal 9", "Canal 10", "Canal 11", "Canal 12",
                            "Canal 13", "Canal 14", "Canal 15", "Canal 16", "EMG 1",
                            "EMG 2", "Timestamp", "Marca"]
def filter_data():
    #csv_data = pandas.read_csv('./S1/Archivos csv/S1_habla_imaginada_4.csv', names=column_names)
    #csv_data = pandas.read_csv('./S1/Archivos csv/S1_habla_no_pronunciada_4.csv', names=column_names)
    csv_data = pandas.read_csv('./S1/Archivos csv/S1_habla_pronunciada_2.csv', names=column_names)

    #filter for EMG 4-200 Hz EMG
    b, a = signal.iirfilter(4, [4, 200], fs=1200, rs=60, btype="bandpass", ftype='butter')

    #filter for EEG 1-40 Hz Canales
    d, c = signal.iirfilter(4, [1,40], fs=1200, rs=60, btype="bandpass", ftype='butter')

    output_signals_columns = []
    output_signals_columns.append(csv_data.get("Tiempo"))
    for i in range(1, len(column_names) - 2):
        if column_names[i] == "EMG 1" or column_names[i] == "EMG 2":
            output_signals_columns.append(signal.filtfilt(b, a, csv_data.get(column_names[i])))
        else:
            output_signals_columns.append(signal.filtfilt(d, c, csv_data.get(column_names[i])))
    output_signals_columns.append(csv_data.get("Timestamp"))
    output_signals_columns.append(csv_data.get("Marca"))
    
    print(output_signals_columns[0][0])
    print(output_signals_columns[1][0])
    print(output_signals_columns[2][0])

    np.savetxt("filtered_data.txt", output_signals_columns)
    return output_signals_columns

def plot_channel(signals, channel):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(signals[channel+2])
    ax.set_title('Butterworth bandpass frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.grid(which='both', axis='both')
    plt.show()

def generate_segments_by_tag(data,samp_rate,sec_per_win):
    #samp_rate = 256
    samps = data.shape[1]
    n_channels = data.shape[0]

    print("Analizando")
    print('Número de muestras: ', data.shape[1])
    print('Número de canales: ', data.shape[0])
    print('Duración del registro: ', samps / samp_rate, 'segundos')
    #print(data)

    # Time channel
    time = data[0, :]

    # Data channels
    chann1 = data[1, :]
    chann2 = data[2, :]
    chann3 = data[3, :]
    chann4 = data[4, :]
    chann5 = data[5, :]
    chann6 = data[6, :]
    chann7 = data[7, :]
    chann8 = data[8, :]
    chann9 = data[9, :]
    chann10 = data[10, :]
    chann11 = data[11, :]
    chann12 = data[12, :]
    chann13 = data[13, :]
    chann14 = data[14, :]
    chann15 = data[15, :]
    chann16 = data[16, :]

    #EMG channels
    emg1 = data[17, :]
    emg2 = data[18, :]

    #Marks cahnnel
    mark = data[20, :]

    marks = []
    #EEG
    channels = [chann1,chann2, chann3, chann4, chann5, chann6, chann7, chann8, chann9, chann10, chann11, chann12, chann13, chann14, chann15, chann16]

    #EMG
    #channels = [emg1,emg2]

    training_samples = {}

    #Get data ranges from different marks
    #0 = nothing
    #1 = attention
    #2-6 = words
    #7 = relaxation
    for i in range(0, samps): 
        if mark[i] > 1: 
            if mark[i] != 7:
                if(mark[i] not in marks):
                    marks.append(mark[i])
                #print("Current mark: "+str(mark[i]))
                iniSamp = i
                condition_id = mark[i]
            else:
                if not condition_id in training_samples.keys():
                    training_samples[int(condition_id)] = []
                training_samples[int(condition_id)].append([iniSamp, i])
            
    marks.sort()
    print(marks)
    print("My data splited:")
    print(training_samples)

    # Power Spectral Density (PSD) (1 second of training data)
    #sec_per_win = 1

    win_size = samp_rate * sec_per_win

    powers = []
    tags=[]
    ch1 = True
    index = 0

    for chann in channels:
        for m in marks:
            for i in range(len(training_samples[m])):
                for j in range(training_samples[m][i][0],training_samples[m][i][1],win_size):
                    ini_samp = j
                    end_samp = j + win_size
                    x = chann[ini_samp : end_samp]
                    t = time[ini_samp : end_samp]
                    power, freq = plt.psd(x, NFFT = win_size, Fs = samp_rate)
                    plt.clf()

                    #EMG
                    #start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
                    #end_freq = next(x for x, val in enumerate(freq) if val >= 200.0)

                    #EEG
                    start_freq = next(x for x, val in enumerate(freq) if val >= 1.0)
                    end_freq = next(x for x, val in enumerate(freq) if val >= 40.0)
                    
                    start_index = np.where(freq >= start_freq)[0][0]
                    end_index = np.where(freq >= end_freq)[0][0]
                    
                    power = power[start_index:(end_index+1)]
                    freq = freq[start_index:(end_index+1)]

                    if ch1 :
                        powers.append(list(power))
                    else:
                        powers[index] = powers[index] + list(power)
                        #print("power?: "+str(powers[index]))
                        index+=1

            print("len powers in "+str(m)+" = "+str(len(powers)))
            aux = len(tags)
            for i in range(len(powers)-aux):
                tags.append(int(m))
        index = 0
        ch1 = False
    x = np.array(powers)
    y = np.array(tags)

    #x is an array with all the powers of each window created

    print(x.shape)
    print(y.shape)
    #print(x)
    #print(y)

    # Escribir doc con psds para clasificador
    #np.savetxt("Analisis de datos/emg_trainning/pron/x_4.txt",x)
    #np.savetxt("Analisis de datos/emg_trainning/pron/y_4.txt",y)

    np.savetxt("Analisis de datos/eeg_trainning/pron/x_4.txt",x)
    np.savetxt("Analisis de datos/eeg_trainning/pron/y_4.txt",y)

#output_signals_columns = filter_data()

#plot_channel(output_signals_columns,1)

loaded = np.loadtxt("filtered_data.txt")

generate_segments_by_tag(loaded,1200, 1)
