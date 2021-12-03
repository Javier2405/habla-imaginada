#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Read data file
#data = np.loadtxt("Entrenamiento//Izquierda - Derecha - Cerrado 1.txt") 
data = np.loadtxt("Entrenamiento//Abierto - Cerrado - Normal 1.txt")
#data = np.loadtxt("Entrenamiento//Abierto - Cerrado 1.txt")

samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]

print("Analizando")
'''print('Número de muestras: ', data.shape[0])
print('Número de canales: ', data.shape[1])
print('Duración del registro: ', samps / samp_rate, 'segundos')
print(data)'''

# Time channel
time = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

posturas = []
posturas_final=[]
channels = [chann1,chann2]

# Mark data
mark = data[:, 6]

training_samples = {}
for i in range(0, samps): 
    if mark[i] > 0: 
        #print("Marca", mark[i], 'Muestra', i, 'Tiempo', time[i]) 

        if  (mark[i] > 100) and (mark[i] < 200):
            posturas_final.append(mark[i])
            if(mark[i] not in posturas):
                posturas.append(mark[i])
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
            training_samples[int(condition_id)].append([iniSamp, i])

#print('Rango de muestras con datos de entrenamiento:', training_samples)
posturas.sort()
#print(posturas_final)

# Power Spectral Density (PSD) (1 second of training data)
sec_per_win = 1

win_size = 256 * sec_per_win

powers = []
tags=[]
ch1 = True
index = 0

for chann in channels:
    for postura in posturas:
        for i in range(len(training_samples[postura])):
            for j in range(training_samples[postura][i][0],training_samples[postura][i][1],win_size):
                ini_samp = j
                end_samp = j + win_size
                x = chann[ini_samp : end_samp]
                t = time[ini_samp : end_samp]
                power, freq = plt.psd(x, NFFT = win_size, Fs = samp_rate)
                plt.clf()

                start_freq = next(x for x, val in enumerate(freq) if val >= 4.0);
                end_freq = next(x for x, val in enumerate(freq) if val >= 60.0);
                
                start_index = np.where(freq >= start_freq)[0][0]
                end_index = np.where(freq >= end_freq)[0][0]
                
                power = power[start_index:(end_index+1)]
                freq = freq[start_index:(end_index+1)]

                if ch1 :
                    powers.append(list(power))
                else:
                    powers[index] = powers[index] + list(power)
                    index+=1

        #print(len(powers))
        aux = len(tags)
        for i in range(len(powers)-aux):
            tags.append(int(postura))
    index = 0
    ch1 = False
x = np.array(powers)
y = np.array(tags)
'''print(x.shape)
print(y.shape)
print(x)
print(y)'''

# Escribir doc con psds para clasificador
np.savetxt("x_training.txt",x)
np.savetxt("y_training.txt",y)

#SVM LINEAR
clf_linear = svm.SVC(kernel = 'linear')
clf_linear.fit(x, y)

#Train SVM rbf base radial
clf_rbf = svm.SVC(kernel = 'rbf')
clf_rbf.fit(x,y)

#Train k-NN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

#Train decision tree
clf_tree = DecisionTreeClassifier(random_state=0)

kf = KFold(n_splits=5, shuffle = True)

acc_linear = 0
acc_rbf = 0
acc_neigh = 0
acc_tree = 0
acc_red = 0
acc_red2 = 0

n_features=len(x[0])
y=y-101

for train_index, test_index in kf.split(x):

    # Training phase linear
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf_linear.fit(x_train, y_train)

    # Test phase linear
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_linear.predict(x_test)
    print(x_test)
    print(y_test)

    # Calculate confusion matrix and model performance linear
    cm = confusion_matrix(y_test, y_pred)
    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)  
    #print('acc_i linear = ', acc_i)

    acc_linear += acc_i 

    # Training phase rbf
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf_rbf.fit(x_train, y_train)

    # Test phase rbf
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_rbf.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)      
    #print('acc_i rbf = ', acc_i)

    acc_rbf += acc_i 

    # Training phase k-NN
    x_train = x[train_index, :]
    y_train = y[train_index]
    neigh.fit(x_train, y_train)

    # Test phase k-NN
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = neigh.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)     
    #print('acc_i k-NN = ', acc_i)

    acc_neigh += acc_i 

    # Training phase tree
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf_tree.fit(x_train, y_train)

    # Test phase tree
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_tree.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)     
    #print('acc_i tree = ', acc_i)

    acc_tree += acc_i 

    # Training phase red neuronal multicapa
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(4, activation='relu'))
    if(len(posturas)==2):
        clf.add(Dense(1, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        clf.fit(x_train, y_train, epochs=200, batch_size=8, verbose=0)  

        # Test phase red neuronal multicapa
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = (clf.predict(x_test) > 0.5).astype("int32")
    else:
        clf.add(Dense(3, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer='adam')
        y_train = np_utils.to_categorical(y_train) 
        clf.fit(x_train, y_train, epochs=200, batch_size=8, verbose=0)  

        # Test phase red neuronal multicapa
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = np.argmax(clf.predict(x_test), axis=-1)

    cm = confusion_matrix(y_test, y_pred)

    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)  

    acc_red += acc_i    
    #print('acc_i red multicapa = ', acc_i)

     # Training phase red neuronal una capa
    x_train = x[train_index, :]
    y_train = y[train_index]
    

    clf = Sequential()

    if(len(posturas)==2):
        clf.add(Dense(1, input_dim=n_features, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        clf.fit(x_train, y_train, epochs=200, batch_size=8, verbose=0)  

        # Test phase red neuronal una capa
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = (clf.predict(x_test) > 0.5).astype("int32")
    else:
        clf.add(Dense(3, input_dim=n_features, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer='adam')
        y_train = np_utils.to_categorical(y_train) 
        clf.fit(x_train, y_train, epochs=200, batch_size=8, verbose=0)  

        # Test phase red neuronal una capa
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = np.argmax(clf.predict(x_test), axis=-1)
        

    cm = confusion_matrix(y_test, y_pred)

    if(len(posturas)==3):
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    elif (len(posturas)==2):
        acc_i = (cm[0,0]+cm[1,1])/len(y_test)  

    acc_red2 += acc_i    
    #print('acc_i red una capa = ', acc_i)




acc_linear = acc_linear/5
acc_rbf = acc_rbf/5
acc_neigh = acc_neigh/5
acc_tree = acc_tree/5
acc_red = acc_red/5
acc_red2 = acc_red2/5

#Accuracy
print("-----------------------------")
print('ACC linear = ', acc_linear)
print('ACC rbf = ', acc_rbf)
print('ACC k-NN = ', acc_neigh)
print('ACC decision tree = ', acc_tree)
print('ACC red multicapa = ', acc_red)
print('ACC red una sola capa = ', acc_red2)
print("------------------------------")

