import matplotlib.pyplot as plt
import numpy as np
import json
import csv

from sklearn import svm
from sklearn import tree

# Set features to use
opt_artistscore = True
opt_danceability = True
opt_energy = False
opt_key = False
opt_loudness = True
opt_mode = False
opt_speechiness = True
opt_acousticness = True
opt_instrumentalness = True
opt_liveness = False
opt_valence = True
opt_tempo = True

# Set information
m_train = 1787    # Number of training examples

# Set features to consider
n = 0
if opt_artistscore:
    n +=1
if opt_danceability:
    n +=1
if opt_energy:
    n +=1
if opt_key:
    n +=1
if opt_loudness:
    n +=1
if opt_mode:
    n +=1
if opt_speechiness:
    n +=1
if opt_acousticness:
    n +=1
if opt_instrumentalness:
    n +=1
if opt_liveness:
    n +=1
if opt_valence:
    n +=1
if opt_tempo:
    n +=1

# Load dataset
artist_l = []
track_l = []
year_l = []
month_l = []
x = np.zeros([m_train,n])
y = np.zeros([m_train])
with open('complete_project_data_02_10.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            artist_l.append(row[0])
            track_l.append(row[1])
            year_l.append(row[2])
            month_l.append(row[3])
            j = 0
            if opt_artistscore:
                x[line_count-1,j] = row[4]
                j += 1
            if opt_danceability:
                x[line_count-1,j] = row[5]
                j += 1
            if opt_energy:
                x[line_count-1,j] = row[6]
                j += 1
            if opt_key:
                x[line_count-1,j] = row[7]
                j += 1
            if opt_loudness:
                x[line_count-1,j] = row[8]
                j += 1
            if opt_mode:
                x[line_count-1,j] = row[9]
                j += 1
            if opt_speechiness:
                x[line_count-1,j] = row[10]
                j += 1
            if opt_acousticness:
                x[line_count-1,j] = row[11]
                j += 1
            if opt_instrumentalness:
                x[line_count-1,j] = row[12]
                j += 1
            if opt_liveness:
                x[line_count-1,j] = row[13]
                j += 1
            if opt_valence:
                x[line_count-1,j] = row[14]
                j += 1
            if opt_tempo:
                x[line_count-1,j] = row[15]
                j += 1
            y[line_count-1] = row[16]
        line_count += 1

x_train = x
y_train = y.ravel()

m_valid = 1435    # Number of training examples

# Load dataset
artist_l = []
track_l = []
year_l = []
month_l = []
x = np.zeros([m_valid,n])
y = np.zeros([m_valid])
with open('complete_project_data_11_01.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            artist_l.append(row[0])
            track_l.append(row[1])
            year_l.append(row[2])
            month_l.append(row[3])
            j = 0
            if opt_artistscore:
                x[line_count-1,j] = row[4]
                j += 1
            if opt_danceability:
                x[line_count-1,j] = row[5]
                j += 1
            if opt_energy:
                x[line_count-1,j] = row[6]
                j += 1
            if opt_key:
                x[line_count-1,j] = row[7]
                j += 1
            if opt_loudness:
                x[line_count-1,j] = row[8]
                j += 1
            if opt_mode:
                x[line_count-1,j] = row[9]
                j += 1
            if opt_speechiness:
                x[line_count-1,j] = row[10]
                j += 1
            if opt_acousticness:
                x[line_count-1,j] = row[11]
                j += 1
            if opt_instrumentalness:
                x[line_count-1,j] = row[12]
                j += 1
            if opt_liveness:
                x[line_count-1,j] = row[13]
                j += 1
            if opt_valence:
                x[line_count-1,j] = row[14]
                j += 1
            if opt_tempo:
                x[line_count-1,j] = row[15]
                j += 1
            y[line_count-1] = row[16]
        line_count += 1

x_valid = x
y_valid = y.ravel()

clf_lin = svm.SVC(gamma='scale',kernel='linear')
theta = clf_lin.fit(x_train,y_train)
h_lin = clf_lin.predict(x_train)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    h_split[i] = h_lin[i]
svm_accuracy_lin_train = np.mean(h_split == y_train)

h_lin = clf_lin.predict(x_valid)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split[i] = h_lin[i]
svm_accuracy_lin_valid = np.mean(h_split == y_valid)

print('theta_lin')
print(theta)
print('accuracy_lin_train')
print(svm_accuracy_lin_train)
print('accuracy_lin_valid')
print(svm_accuracy_lin_valid)

clf_rbf = svm.SVC(gamma='scale',kernel='rbf')
theta = clf_rbf.fit(x_train,y_train)
h_rbf = clf_rbf.predict(x_train)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    h_split[i] = h_rbf[i]
svm_accuracy_rbf_train = np.mean(h_split == y_train)

h_rbf = clf_rbf.predict(x_valid)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split[i] = h_rbf[i]
svm_accuracy_rbf_valid = np.mean(h_split == y_valid)

print('theta_rbf')
print(theta)
print('accuracy_rbf_train')
print(svm_accuracy_rbf_train)
print('accuracy_rbf_valid')
print(svm_accuracy_rbf_valid)

clf_poly = svm.SVC(gamma='scale',kernel='poly')
theta = clf_poly.fit(x_train,y_train)
h_poly = clf_poly.predict(x_train)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    h_split[i] = h_poly[i]
svm_accuracy_poly_train = np.mean(h_split == y_train)

h_poly = clf_poly.predict(x_valid)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split[i] = h_poly[i]
svm_accuracy_poly_valid = np.mean(h_split == y_valid)

print('theta_poly')
print(theta)
print('accuracy_poly_train')
print(svm_accuracy_poly_train)
print('accuracy_poly_valid')
print(svm_accuracy_poly_valid)

clf_sig = svm.SVC(gamma='scale',kernel='sigmoid')
theta = clf_sig.fit(x_train,y_train)
h_sig = clf_sig.predict(x_train)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    h_split[i] = h_sig[i]
svm_accuracy_sig_train = np.mean(h_split == y_train)

h_sig = clf_sig.predict(x_valid)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split[i] = h_sig[i]
svm_accuracy_sig_valid = np.mean(h_split == y_valid)

print('theta_sig')
print(theta)
print('accuracy_sig_train')
print(svm_accuracy_sig_train)
print('accuracy_sig_valid')
print(svm_accuracy_sig_valid)

clf_tree = tree.DecisionTreeClassifier()
theta = clf_tree.fit(x_train,y_train)
h_tree = clf_tree.predict(x_train)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    h_split[i] = h_tree[i]
svm_accuracy_tree_train = np.mean(h_split == y_train)

h_sig = clf_tree.predict(x_valid)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split[i] = h_tree[i]
svm_accuracy_tree_valid = np.mean(h_split == y_valid)

print('theta_tree')
print(theta)
print('accuracy_tree_train')
print(svm_accuracy_tree_train)
print('accuracy_tree_valid')
print(svm_accuracy_tree_valid)
