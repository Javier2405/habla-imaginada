import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix

def joinArrays():
    paths=["Analisis de datos/emg_trainning/img/","Analisis de datos/eeg_trainning/img/","Analisis de datos/emg_trainning/nopron/","Analisis de datos/eeg_trainning/nopron/","Analisis de datos/emg_trainning/pron/","Analisis de datos/eeg_trainning/pron/"]
    for path in paths:
        x=np.concatenate((np.loadtxt(path+"x_1.txt"),np.loadtxt(path+"x_2.txt"),np.loadtxt(path+"x_3.txt"),np.loadtxt(path+"x_4.txt")))
        y=np.concatenate((np.loadtxt(path+"y_1.txt"),np.loadtxt(path+"y_2.txt"),np.loadtxt(path+"y_3.txt"),np.loadtxt(path+"y_4.txt")))
        np.savetxt(path+"x_final.txt",x)
        np.savetxt(path+"y_final.txt",y)

def get_precision(cm):
    precisions=[]
    for i in range(len(cm)):
        tp=cm[i,i]
        totalpp=cm[0,i]+cm[1,i]+cm[2,i]+cm[3,i]+cm[4,i]
        precision=tp/totalpp
        precisions.append(precision)
    #print("Precision: "+str(precisions))
    return precisions

def get_recall(cm):
    recalls=[]
    print(cm)
    for i in range(len(cm)):
        tp=cm[i,i]
        totalap=cm[i,0]+cm[i,1]+cm[i,2]+cm[i,3]+cm[i,4]
        recall=tp/totalap
        recalls.append(recall)
    #print("Recall: "+str(recalls))
    return recalls

def get_f1_score(recall, precision):
    prom=0
    f1s=[]
    for i in range(len(recall)):
        f1=2*((precision[i]*recall[i])/(precision[i]+recall[i]))
        prom+=f1
        f1s.append(f1)
    prom/=len(recall)
    return f1s

def get_mean(array):
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    for i in range(len(array)):
        x1+=array[i][0]
        x2+=array[i][1]
        x3+=array[i][2]
        x4+=array[i][3]
        x5+=array[i][4]
    x1/=len(array)
    x2/=len(array)
    x3/=len(array)
    x4/=len(array)
    x5/=len(array)
    return [x1,x2,x3,x4,x5]


#join data from different csv files
#joinArrays()

data={}

#################
### IMAGINADA ###
#################

#EMG
emg_img_x=np.loadtxt("Analisis de datos/emg_trainning/img/x_final.txt")
emg_img_y=np.loadtxt("Analisis de datos/emg_trainning/img/y_final.txt")

#EEG
eeg_img_x=np.loadtxt("Analisis de datos/eeg_trainning/img/x_final.txt")
eeg_img_y=np.loadtxt("Analisis de datos/eeg_trainning/img/y_final.txt")

data["imaginada"]={
    "emg":{
        "x":emg_img_x,
        "y":emg_img_y
    },
    "eeg":{
        "x":eeg_img_x,
        "y":eeg_img_y
    }
}

###################
### PRONUNCIADA ###
###################

#EMG
emg_pron_x=np.loadtxt("Analisis de datos/emg_trainning/pron/x_final.txt")
emg_pron_y=np.loadtxt("Analisis de datos/emg_trainning/pron/y_final.txt")

#EEG
eeg_pron_x=np.loadtxt("Analisis de datos/eeg_trainning/pron/x_final.txt")
eeg_pron_y=np.loadtxt("Analisis de datos/eeg_trainning/pron/y_final.txt")

data["pronunciada"]={
    "emg":{
        "x":emg_pron_x,
        "y":emg_pron_y
    },
    "eeg":{
        "x":eeg_pron_x,
        "y":eeg_pron_y
    }
}

######################
### NO PRONUNCIADA ###
######################

#EMG
emg_nopron_x=np.loadtxt("Analisis de datos/emg_trainning/nopron/x_final.txt")
emg_nopron_y=np.loadtxt("Analisis de datos/emg_trainning/nopron/y_final.txt")

#EEG
eeg_nopron_x=np.loadtxt("Analisis de datos/eeg_trainning/nopron/x_final.txt")
eeg_nopron_y=np.loadtxt("Analisis de datos/eeg_trainning/nopron/y_final.txt")

data["no pronunciada"]={
    "emg":{
        "x":emg_nopron_x,
        "y":emg_nopron_y
    },
    "eeg":{
        "x":eeg_nopron_x,
        "y":eeg_nopron_y
    }
}

######################
### CLASIFICADORES ###
######################

#SVM LINEAR
clf_linear = svm.SVC(kernel = 'linear')
#clf_linear.fit(x, y)

#Train SVM rbf base radial
clf_rbf = svm.SVC(kernel = 'rbf')
#clf_rbf.fit(x,y)

#Train k-NN
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(x, y)

#KFold
splits=5
kf = KFold(n_splits=splits, shuffle = True)

#Repeat the kfold
rep=5

####################################
### TRAINING AND TEST WITH KFOLD ###
####################################

f = open("Analisis de datos/results_final4.txt", "w")

for type in data:
    for signal_type in data[type]:
        if signal_type=="emg":
            k_fs=100
        elif signal_type=="eeg":
            k_fs=200
        total_acc_linear=0
        total_acc_rbf=0
        total_acc_neigh=0

        total_f1_linear=0
        total_f1_rbf=0
        total_f1_neigh=0

        total_pre_linear=[]
        total_pre_rbf=[]
        total_pre_neigh=[]

        total_rec_linear=[]
        total_rec_rbf=[]
        total_rec_neigh=[]

        for i in range (rep):
            f.write("############################## ")
            f.write(type+" # "+signal_type)
            f.write(" ##############################\n")

            acc_linear=0
            acc_rbf=0
            acc_neigh=0

            f1_linear=[]
            f1_rbf=[]
            f1_neigh=[]

            pre_linear=[]
            pre_rbf=[]
            pre_neigh=[]

            rec_linear=[]
            rec_rbf=[]
            rec_neigh=[]

            x=data[type][signal_type]["x"]
            y=data[type][signal_type]["y"]

            print(x.shape)
            print(y.shape)
            x = SelectKBest(chi2, k=k_fs).fit_transform(x, y)
            print(x.shape)
            print(y.shape)

            for train_index, test_index in kf.split(x):
                # Training phase linear
                x_train = x[train_index, :]
                y_train = y[train_index]
                clf_linear.fit(x_train, y_train)

                # Test phase linear
                x_test = x[test_index, :]
                y_test = y[test_index]    
                y_pred = clf_linear.predict(x_test)
                #print(x_test)
                #print(y_test)

                # Calculate confusion matrix and model performance linear
                cm = confusion_matrix(y_test, y_pred)
                acc_i = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)      
                #print('acc_i linear = ', acc_i)
                recall_i=get_recall(cm)
                precision_i=get_precision(cm)
                f1_i=get_f1_score(recall_i,precision_i)
                
                rec_linear.append(recall_i)
                pre_linear.append(precision_i)
                f1_linear.append(f1_i)
                acc_linear += acc_i 

                # Training phase rbf
                x_train = x[train_index, :]
                y_train = y[train_index]
                clf_rbf.fit(x_train, y_train)

                # Test phase rbf
                x_test = x[test_index, :]
                y_test = y[test_index]    
                y_pred = clf_rbf.predict(x_test)

                # Calculate confusion matrix and model performance linear
                cm = confusion_matrix(y_test, y_pred)
                #print("ytest: "+str(len(y_test)))
                acc_i = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)  
                #print('acc_i rbf = ', acc_i)
                recall_i=get_recall(cm)
                precision_i=get_precision(cm)
                f1_i=get_f1_score(recall_i,precision_i)
                
                rec_rbf.append(recall_i)
                pre_rbf.append(precision_i)
                f1_rbf.append(f1_i)
                acc_rbf += acc_i 

                # Training phase k-NN
                x_train = x[train_index, :]
                y_train = y[train_index]
                neigh.fit(x_train, y_train)

                # Test phase k-NN
                x_test = x[test_index, :]
                y_test = y[test_index]    
                y_pred = neigh.predict(x_test)

                # Calculate confusion matrix and model performance linear
                cm = confusion_matrix(y_test, y_pred)
                acc_i = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)        
                #print('acc_i k-NN = ', acc_i)
                recall_i=get_recall(cm)
                precision_i=get_precision(cm)
                f1_i=get_f1_score(recall_i,precision_i)

                rec_neigh.append(recall_i)
                pre_neigh.append(precision_i)
                f1_neigh.append(f1_i)
                acc_neigh += acc_i 
            
            acc_linear = acc_linear/splits
            acc_rbf = acc_rbf/splits
            acc_neigh = acc_neigh/splits
            #f1_linear = f1_linear/splits
            #f1_rbf = f1_rbf/splits
            #f1_neigh = f1_neigh/splits
            f.write("\n-----------------------------")
            f.write(' ACC linear = '+ str(acc_linear))
            f.write(' ACC rbf = '+ str(acc_rbf))
            f.write(' ACC k-NN = '+ str(acc_neigh))
            f.write(" -----------------------------\n")
            total_acc_linear+=acc_linear
            total_acc_rbf+=acc_rbf
            total_acc_neigh+=acc_neigh
            #total_f1_linear+=f1_linear
            #total_f1_rbf+=f1_rbf
            #total_f1_neigh+=f1_neigh

        total_acc_linear = total_acc_linear/rep
        total_acc_rbf = total_acc_rbf/rep
        total_acc_neigh = total_acc_neigh/rep
        total_f1_linear = get_mean(f1_linear)
        total_f1_rbf = get_mean(f1_rbf)
        total_f1_neigh = get_mean(f1_neigh)
        total_rec_linear = get_mean(rec_linear)
        total_rec_rbf = get_mean(rec_rbf)
        total_rec_neigh = get_mean(rec_neigh)
        total_pre_linear = get_mean(pre_linear)
        total_pre_rbf = get_mean(pre_rbf)
        total_pre_neigh = get_mean(pre_neigh)
        f.write("\n\nFINAL REPORT "+type+" "+signal_type+"\n")
        f.write("\n+++++++++++++++++++++++++++++")
        f.write('\nACC linear = '+ str(total_acc_linear))
        f.write('\nACC rbf = '+ str(total_acc_rbf))
        f.write('\nACC k-NN = '+ str(total_acc_neigh))
        f.write("\n+++++++++++++++++++++++++++++\n")
        f.write('\nf1 linear = '+ str(total_f1_linear))
        f.write('\nf1 rbf = '+ str(total_f1_rbf))
        f.write('\nf1 k-NN = '+ str(total_f1_neigh))
        f.write("\n+++++++++++++++++++++++++++++\n\n")
        f.write('\nrecall linear = '+ str(total_rec_linear))
        f.write('\nrecall rbf = '+ str(total_rec_rbf))
        f.write('\nrecall k-NN = '+ str(total_rec_neigh))
        f.write("\n+++++++++++++++++++++++++++++\n")
        f.write('\nprecision linear = '+ str(total_pre_linear))
        f.write('\nprecision rbf = '+ str(total_pre_rbf))
        f.write('\nprecision k-NN = '+ str(total_pre_neigh))
        f.write("\n+++++++++++++++++++++++++++++\n\n")

        print(("\n\nFINAL REPORT "+type+" "+signal_type+"\n"))
        print("\n+++++++++++++++++++++++++++++")
        print('ACC linear = ', total_acc_linear)
        print('ACC rbf = ', total_acc_rbf)
        print('ACC k-NN = ', total_acc_neigh)
        print("+++++++++++++++++++++++++++++\n")
        print('\nf1 linear = '+ str(total_f1_linear))
        print('\nf1 rbf = '+ str(total_f1_rbf))
        print('\nf1 k-NN = '+ str(total_f1_neigh))
        print("\n+++++++++++++++++++++++++++++\n\n")
        print('\nrecall linear = '+ str(total_rec_linear))
        print('\nrecall rbf = '+ str(total_rec_rbf))
        print('\nrecall k-NN = '+ str(total_rec_neigh))
        print("\n+++++++++++++++++++++++++++++\n")
        print('\nprecision linear = '+ str(total_pre_linear))
        print('\nprecision rbf = '+ str(total_pre_rbf))
        print('\nprecision k-NN = '+ str(total_pre_neigh))
        print("\n+++++++++++++++++++++++++++++\n\n")

f.close()