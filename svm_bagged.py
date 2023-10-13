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
m = 3221    # Number of examples

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
x = np.zeros([m,n])
y = np.zeros([m])
with open('complete_project_data.csv') as csv_file:
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

# Split into training and validation
m_train = 2416
m_valid = m - m_train
x_train = np.zeros([m_train,n])
y_train = np.zeros([m_train])
x_valid = np.zeros([m_valid,n])
y_valid = np.zeros([m_valid])

for i in range(0,m_train):
    x_train[i,:] = x[i,:]
    y_train[i] = y[i]
for i in range(0,m_valid):
    x_valid[i,:] = x[i + m_train,:]
    y_valid[i] = y[i + m_train]

y_train = y_train.ravel()

# Obtain random sample
m_rand = 200
trials = 10

# Run algorithm with bagging
h_lin_t = np.zeros([m_train])
h_lin_v = np.zeros([m_valid])
x_rand = np.zeros([m_rand,n])
y_rand = np.zeros([m_rand])
for i in range(0,trials):
    print('Trial: ',i)
    rand_index = []
    for j in range(0,m_rand):
        while True:
            rand = np.random.randint(0,m_train)
            if not rand in rand_index:
                rand_index.append(rand)
                break
        x_rand[j,:] = x_train[rand,:]
        y_rand[j] = y_train[rand]
    clf_lin = svm.SVC(gamma='scale',kernel='linear')
    theta = clf_lin.fit(x_rand,y_rand)
    h_rand_t = clf_lin.predict(x_train)
    h_rand_v = clf_lin.predict(x_valid)
    h_lin_t = h_lin_t + h_rand_t
    h_lin_v = h_lin_v + h_rand_v
h_lin_t = h_lin_t/trials
h_lin_v = h_lin_v/trials

h_split_lin = np.zeros([m_train])
for i in range(0,m_train):
    h_split_lin[i] = h_lin_t[i]
svm_accuracy_lin_train = np.mean(h_split_lin == y_train)

h_split_lin = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split_lin[i] = h_lin_v[i]
svm_accuracy_lin_valid = np.mean(h_split_lin == y_valid)

print('accuracy_lin_train')
print(svm_accuracy_lin_train)
print('accuracy_lin_valid')
print(svm_accuracy_lin_valid)

# Run algorithm with bagging
h_rbf_t = np.zeros([m_train])
h_rbf_v = np.zeros([m_valid])
x_rand = np.zeros([m_rand,n])
y_rand = np.zeros([m_rand])
for i in range(0,trials):
    print('Trial: ',i)
    rand_index = []
    for j in range(0,m_rand):
        while True:
            rand = np.random.randint(0,m_train)
            if not rand in rand_index:
                rand_index.append(rand)
                break
        x_rand[j,:] = x_train[rand,:]
        y_rand[j] = y_train[rand]
    clf_rbf = svm.SVC(gamma='scale',kernel='rbf')
    theta = clf_rbf.fit(x_rand,y_rand)
    h_rand_t = clf_rbf.predict(x_train)
    h_rand_v = clf_rbf.predict(x_valid)
    h_rbf_t = h_rbf_t + h_rand_t
    h_rbf_v = h_rbf_v + h_rand_v
h_rbf_t = h_rbf_t/trials
h_rbf_v = h_rbf_v/trials

h_split_rbf = np.zeros([m_train])
for i in range(0,m_train):
    h_split_rbf[i] = h_rbf_t[i]
svm_accuracy_rbf_train = np.mean(h_split_rbf == y_train)

h_split_rbf = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split_rbf[i] = h_rbf_v[i]
svm_accuracy_rbf_valid = np.mean(h_split_rbf == y_valid)

print('accuracy_rbf_train')
print(svm_accuracy_rbf_train)
print('accuracy_rbf_valid')
print(svm_accuracy_rbf_valid)

# Run algorithm with bagging
h_poly_t = np.zeros([m_train])
h_poly_v = np.zeros([m_valid])
x_rand = np.zeros([m_rand,n])
y_rand = np.zeros([m_rand])
for i in range(0,trials):
    print('Trial: ',i)
    rand_index = []
    for j in range(0,m_rand):
        while True:
            rand = np.random.randint(0,m_train)
            if not rand in rand_index:
                rand_index.append(rand)
                break
        x_rand[j,:] = x_train[rand,:]
        y_rand[j] = y_train[rand]
    clf_poly = svm.SVC(gamma='scale',kernel='poly')
    theta = clf_poly.fit(x_rand,y_rand)
    h_rand_t = clf_poly.predict(x_train)
    h_rand_v = clf_poly.predict(x_valid)
    h_poly_t = h_poly_t + h_rand_t
    h_poly_v = h_poly_v + h_rand_v
h_poly_t = h_poly_t/trials
h_poly_v = h_poly_v/trials

h_split_poly = np.zeros([m_train])
for i in range(0,m_train):
    h_split_poly[i] = h_poly_t[i]
svm_accuracy_poly_train = np.mean(h_split_poly == y_train)

h_split_poly = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split_poly[i] = h_poly_v[i]
svm_accuracy_poly_valid = np.mean(h_split_poly == y_valid)

print('accuracy_poly_train')
print(svm_accuracy_poly_train)
print('accuracy_poly_valid')
print(svm_accuracy_poly_valid)

# Run algorithm with bagging
h_sig_t = np.zeros([m_train])
h_sig_v = np.zeros([m_valid])
x_rand = np.zeros([m_rand,n])
y_rand = np.zeros([m_rand])
for i in range(0,trials):
    print('Trial: ',i)
    rand_index = []
    for j in range(0,m_rand):
        while True:
            rand = np.random.randint(0,m_train)
            if not rand in rand_index:
                rand_index.append(rand)
                break
        x_rand[j,:] = x_train[rand,:]
        y_rand[j] = y_train[rand]
    clf_sig = svm.SVC(gamma='scale',kernel='sigmoid')
    theta = clf_sig.fit(x_rand,y_rand)
    h_rand_t = clf_sig.predict(x_train)
    h_rand_v = clf_sig.predict(x_valid)
    h_sig_t = h_sig_t + h_rand_t
    h_sig_v = h_sig_v + h_rand_v
h_sig_t = h_sig_t/trials
h_sig_v = h_sig_v/trials

h_split_sig = np.zeros([m_train])
for i in range(0,m_train):
    h_split_sig[i] = h_sig_t[i]
svm_accuracy_sig_train = np.mean(h_split_sig == y_train)

h_split_sig = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split_sig[i] = h_sig_v[i]
svm_accuracy_sig_valid = np.mean(h_split_sig == y_valid)

print('accuracy_sig_train')
print(svm_accuracy_sig_train)
print('accuracy_sig_valid')
print(svm_accuracy_sig_valid)

# Run algorithm with bagging
h_tree_t = np.zeros([m_train])
h_tree_v = np.zeros([m_valid])
x_rand = np.zeros([m_rand,n])
y_rand = np.zeros([m_rand])
for i in range(0,trials):
    print('Trial: ',i)
    rand_index = []
    for j in range(0,m_rand):
        while True:
            rand = np.random.randint(0,m_train)
            if not rand in rand_index:
                rand_index.append(rand)
                break
        x_rand[j,:] = x_train[rand,:]
        y_rand[j] = y_train[rand]
    clf_tree = tree.DecisionTreeClassifier()
    theta = clf_tree.fit(x_rand,y_rand)
    h_rand_t = clf_tree.predict(x_train)
    h_rand_v = clf_tree.predict(x_valid)
    h_tree_t = h_tree_t + h_rand_t
    h_tree_v = h_tree_v + h_rand_v
h_tree_t = h_tree_t/trials
h_tree_v = h_tree_v/trials

h_split_tree = np.zeros([m_train])
for i in range(0,m_train):
    h_split_tree[i] = h_tree_t[i]
svm_accuracy_tree_train = np.mean(h_split_tree == y_train)

h_split_tree = np.zeros([m_valid])
for i in range(0,m_valid):
    h_split_tree[i] = h_tree_v[i]
svm_accuracy_tree_valid = np.mean(h_split_tree == y_valid)

print('accuracy_tree_train')
print(svm_accuracy_tree_train)
print('accuracy_tree_valid')
print(svm_accuracy_tree_valid)
