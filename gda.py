import matplotlib.pyplot as plt
import numpy as np
import json
import csv

def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def plot(x, y, theta, correction=1.0):
    plt.figure()
    x_true_1 = []
    x_true_2 = []
    x_false_1 = []
    x_false_2 = []
    for i in range(0,m):
        if y[i] == 0:
            x_false_1.append(x[i,0])
            x_false_2.append(x[i,1])
        else:
            x_true_1.append(x[i,0])
            x_true_2.append(x[i,1])
    plt.plot(x_true_1, x_true_2, 'go', linewidth=2)
    plt.plot(x_false_1, x_false_2, 'bx', linewidth=2)
    x1 = np.arange(min(x[:,0]), max(x[:,0]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.savefig('graph_gda.png')

def fit(x, y):
    m = x.shape[0]
    n = x.shape[1]
        
    sum_phi = 0.0
    for i in range(0,m):
        if y[i] == 1:
            sum_phi += 1
    phi = (1/m)*sum_phi
    
    sum_mu0_n = np.zeros([n,1])
    sum_mu0_d = 0.0
    for i in range(0,m):
        if y[i] == 0:
            for j in range(0,n):
                sum_mu0_n[j] += x[i,j]
            sum_mu0_d += 1
    mu0 = sum_mu0_n/sum_mu0_d
    
    sum_mu1_n = np.zeros([n,1])
    sum_mu1_d = 0.0
    for i in range(0,m):
        if y[i] == 1:
            for j in range(0,n):
                sum_mu1_n[j] += x[i,j]
            sum_mu1_d += 1
    mu1 = sum_mu1_n/sum_mu1_d
    
    sigma = np.zeros([n,n])
    tra = np.zeros([n,1])
    for i in range(0,m):
        if y[i] == 0:
            for j in range(0,n):
                tra[j] = x[i,j] - mu0[j]
            mat_add = np.matmul(tra,np.transpose(tra))
        elif y[i] == 1:
            for j in range(0,n):
                tra[j] = x[i,j] - mu1[j]
            mat_add = np.matmul(tra,np.transpose(tra))
        sigma += mat_add
    sigma = (1/m)*sigma
    
    theta = np.zeros([n+1,1])
    theta[0] = (1.0/2.0)*(np.matmul(np.matmul(np.transpose(mu0),np.linalg.inv(sigma)),mu0)) - (1.0/2.0)*(np.matmul(np.matmul(np.transpose(mu1),np.linalg.inv(sigma)),mu1)) - np.log((1 - phi)/phi)
    theta_1 = np.zeros([n])
    theta_1 = -(np.matmul(np.linalg.inv(sigma),(mu0 - mu1)))
    
    for i in range(1,n+1):
        theta[i] = theta_1[i-1]
    
    return theta

def predict(x, theta):
    m = x.shape[0]
    h_theta = np.zeros([m,1])
    for i in range(0,m):
        h_theta[i] = sigmoid((np.matmul(np.transpose(theta[1:]),np.transpose(x[i,:]))) + theta[0])
    return h_theta

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

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
y = np.zeros([m,1])
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
y_train = np.zeros([m_train,1])
x_valid = np.zeros([m_valid,n])
y_valid = np.zeros([m_valid,1])

for i in range(0,m_train):
    x_train[i,:] = x[i,:]
    y_train[i] = y[i]
for i in range(0,m_valid):
    x_valid[i,:] = x[i + m_train,:]
    y_valid[i] = y[i + m_train]

theta = fit(x_train, y_train)
h = predict(x_train, theta)
h_split_t = np.zeros([m_train,1])
for i in range(0,m_train):
    if h[i] <= 0.5:
        h_split_t[i] = 0
    else:
        h_split_t[i] = 1
gda_accuracy_train = np.mean(h_split_t == y_train)

h = predict(x_valid, theta)
h_split_v = np.zeros([m_valid,1])
for i in range(0,m_valid):
    if h[i] <= 0.5:
        h_split_v[i] = 0
    else:
        h_split_v[i] = 1
gda_accuracy_valid = np.mean(h_split_v == y_valid)

print('theta')
print(theta)
print('training accuracy')
print(gda_accuracy_train)
print('validation accuracy')
print(gda_accuracy_valid)

tn_t = 0
fn_t = 0
tp_t = 0
fp_t = 0
for i in range(0,m_train):
    if h_split_t[i] == 0 and y_train[i] == 0:
        tn_t += 1
    if h_split_t[i] == 0 and y_train[i] == 1:
        fn_t += 1
    if h_split_t[i] == 1 and y_train[i] == 0:
        fp_t += 1
    if h_split_t[i] == 1 and y_train[i] == 1:
        tp_t += 1
gda_precision_train = tp_t/(tp_t + fp_t)
gda_recall_train = tp_t/(tp_t + fn_t)

print('tn_t = ',tn_t)
print('fn_t = ',fn_t)
print('fp_t = ',fp_t)
print('tp_t = ',tp_t)

print('training precision')
print(gda_precision_train)
print('training recall')
print(gda_recall_train)

tn_v = 0
fn_v = 0
tp_v = 0
fp_v = 0
for i in range(0,m_valid):
    if h_split_v[i] == 0 and y_valid[i] == 0:
        tn_v += 1
    if h_split_v[i] == 0 and y_valid[i] == 1:
        fn_v += 1
    if h_split_v[i] == 1 and y_valid[i] == 0:
        fp_v += 1
    if h_split_v[i] == 1 and y_valid[i] == 1:
        tp_v += 1
gda_precision_valid = tp_v/(tp_v + fp_v)
gda_recall_valid = tp_v/(tp_v + fn_v)

print('tn_v = ',tn_v)
print('fn_v = ',fn_v)
print('fp_v = ',fp_v)
print('tp_v = ',tp_v)

print('validation precision')
print(gda_precision_valid)
print('validation recall')
print(gda_recall_valid)

# Plot
if n == 2:
    correction = 1.0
    plot(x, y, theta, correction=1.0)