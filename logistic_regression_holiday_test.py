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
            x_false_1.append(x[i,1])
            x_false_2.append(x[i,2])
        else:
            x_true_1.append(x[i,1])
            x_true_2.append(x[i,2])
    plt.plot(x_true_1, x_true_2, 'go', linewidth=2)
    plt.plot(x_false_1, x_false_2, 'bx', linewidth=2)
    x1 = np.arange(min(x[:,1]), max(x[:,1]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.savefig('graph_lr.png')

def fit(x, y):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros([n, 1])
    normct = 10 ** 6
    it = 0

    while normct > 10 ** -5:
        H = np.zeros([n, n])
        for a in range(0, n):
            for b in range(0, n):
                sum = 0.0
                for i in range(0, m):
                    p1 = x[i, a]*x[i, b]
                    p2 = sigmoid(np.matmul(np.transpose(theta), np.transpose(x[i, :])))
                    p3 = (1 - sigmoid(np.matmul(np.transpose(theta), np.transpose(x[i, :]))))
                    sum += p1*p2*p3
                H[a, b] = (1 / m)*sum
        Hinv = np.linalg.inv(H)

        grad = np.zeros([n,1])
        for a in range(0, n):
            sum = 0.0
            for i in range(0, m):
                sum += x[i, a]*y[i] - x[i, a]*sigmoid(np.matmul(np.transpose(theta), np.transpose(x[i, :])))
            grad[a] = (-1 / m)*sum

        theta_new = theta - np.matmul(Hinv, grad)
        theta_check = theta - theta_new
        normc = 0.0
        for j in range(0, n):
            normc += abs(theta_check[j])
        normct = normc.copy()
        theta = theta_new.copy()
        print('normct is', normct)
        it += 1
        print('it',it)

    return theta

def predict(x, theta):
    m = x.shape[0]
    h_theta = np.zeros([m,1])
    for i in range(0, m):
        h_theta[i] = sigmoid(np.matmul(np.transpose(theta), np.transpose(x[i, :])))
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
opt_valence = False
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

x = add_intercept(x)

x_train = x
y_train = y

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

x = add_intercept(x)
n = n + 1

x_valid = x
y_valid = y

theta = fit(x_train, y_train)
h = predict(x_train, theta)
h_split = np.zeros([m_train])
for i in range(0,m_train):
    if h[i] <= 0.5:
        h_split[i] = 0
    else:
        h_split[i] = 1
logreg_accuracy_train = np.mean(h_split == y_train)

h = predict(x_valid, theta)
h_split = np.zeros([m_valid])
for i in range(0,m_valid):
    if h[i] <= 0.5:
        h_split[i] = 0
    else:
        h_split[i] = 1
logreg_accuracy_valid = np.mean(h_split == y_valid)

print('theta')
print(theta)
print('training accuracy')
print(logreg_accuracy_train)
print('validation accuracy')
print(logreg_accuracy_valid)

# Plot
if n == 3:
    plot(x, y, theta, correction=1.0)
