import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import random

# Set features to use
opt_artistscore = True
opt_danceability = True
opt_energy = True
opt_key = False
opt_loudness = True
opt_mode = False
opt_speechiness = True
opt_acousticness = True
opt_instrumentalness = True
opt_liveness = True
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

portion_supervised = 0.05
skip = 1/portion_supervised
num_supervised = m*0.05
labeled_idxs = []
for i in range(0,m):
    if int(portion_supervised*i) == portion_supervised*i:
        labeled_idxs.append(True)
    else:
        labeled_idxs.append(False)

x_tilde = x[np.asarray(labeled_idxs), :]   # Labeled examples
z = y[np.asarray(labeled_idxs), :]         # Corresponding labels
x = x[~np.asarray(labeled_idxs), :]        # Unlabeled examples

m = x.shape[0]
m_tilde = x_tilde.shape[0]

# Initialise unsupervised expectation maximisation algorithm
K = 2   # Number of groups
init_group = np.zeros([m,1])
for i in range(0,m):
    init_group[i] = random.randint(0,K-1)

mu = []
sigma = []
for g in range(0,K):
    mu_n = np.zeros([n,1])
    mu_d = 0
    for i in range(0,m):
        if init_group[i] == g:
            for j in range(0,n):
                mu_n[j] += x[i,j]
            mu_d += 1
    mu.append(mu_n/mu_d)
    sigma_n = np.zeros([n,n])
    for i in range(0,m):
        if init_group[i] == g:
            temp_xg = np.zeros([n,1])
            for o in range(0,n):
                temp_xg[o] = x[i,o] - mu[g][o]
            sigma_n += np.matmul(temp_xg,np.transpose(temp_xg))
    sigma.append(sigma_n/mu_d)
    phi = np.zeros([K,1])
for g in range(0,K):
    phi_n = 0
    for i in range(0,m):
        if init_group[i] == g:
            phi_n += 1
    phi[g] = phi_n/m

w = np.zeros([m,K])
for i in range(0,m):
    for j in range(0,K):
        w[i,j] = 1/K

# Run expectation maximisation algorithm
eps = 1e-3          # Convergence threshold
max_iter = 1000     # Maximum interations
alpha = 20          # Weight of supervised subset

it = 0
ll = prev_ll = None
while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
    prev_ll = ll

    for i in range(0,m):
        w_d = 0
        for l in range(0,K):
            temp_xl = np.zeros([n,1])
            for o in range(0,n):
                temp_xl[o] = x[i,o] - mu[l][o]
            w_d_1 = 1/(((2*3.1415)**(n/2))*((np.linalg.det(sigma[l]))**0.5))
            w_d_2 = np.exp(-0.5*np.matmul(np.matmul(np.transpose(temp_xl),np.linalg.inv(sigma[l])),temp_xl))
            w_d_3 = phi[l]
            w_d += w_d_1*w_d_2*w_d_3
        for j in range(0,K):
            temp_xj = np.zeros([n,1])
            for o in range(0,n):
                temp_xj[o] = x[i,o] - mu[j][o]
            w_n_1 = 1/(((2*3.1415)**(n/2))*((np.linalg.det(sigma[j]))**0.5))
            w_n_2 = np.exp(-0.5*np.matmul(np.matmul(np.transpose(temp_xj),np.linalg.inv(sigma[j])),temp_xj))
            w_n_3 = phi[j]
            w_n = w_n_1*w_n_2*w_n_3
            w[i,j] = (w_n/w_d)

    mu = []
    sigma = []
    for j in range(0,K):
        phi[j] = 0
        for i in range(0,m):
            phi[j] += w[i,j]
        for i in range(0,m_tilde):
            if z[i] == j:
                phi[j] += alpha
        phi[j] = phi[j]/(m + alpha*m_tilde)
        mu_n = np.zeros([n,1])
        mu_d = 0
        for i in range(0,m):
            for o in range(0,n):
                mu_n[o] += w[i,j]*x[i,o]
            mu_d += w[i,j]
        for i in range(0,m_tilde):
            if z[i] == j:
                for o in range(0,n):
                    mu_n[o] += alpha*x_tilde[i,o]
                mu_d += alpha
        mu.append(mu_n/mu_d)
        sigma_n = np.zeros([n,n])
        for i in range(0,m):
            temp_xj = np.zeros([n,1])
            for o in range(0,n):
                temp_xj[o] = x[i,o] - mu[j][o]
            sigma_n += w[i,j]*np.matmul(temp_xj,np.transpose(temp_xj))
        for i in range(0,m_tilde):
            if z[i] == j:
                temp_xtj = np.zeros([n,1])
                for o in range(0,n):
                    temp_xtj[o] = x_tilde[i,o] - mu[j][o]
                sigma_n += alpha*np.matmul(temp_xtj,np.transpose(temp_xtj))
        sigma.append(sigma_n/mu_d)

    ll = 0
    for i in range(0,m):
        ll_j = 0
        for j in range(0,K):
            ll_1a = (1/(((2*3.1415)**(n/2))*((np.linalg.det(sigma[j]))**0.5)))
            temp_xj = np.zeros([n,1])
            for o in range(0,n):
                temp_xj[o] = x[i,o] - mu[j][o]
            ll_1b = np.exp(-0.5*np.matmul(np.matmul(np.transpose(temp_xj),np.linalg.inv(sigma[j])),(temp_xj)))
            ll_j += ll_1a*ll_1b*phi[j]
        ll += np.log(ll_j)
    for i in range(0,m_tilde):
        ind = int(z[i])
        ll_1a = (1/(((2*3.1415)**(n/2))*((np.linalg.det(sigma[ind]))**0.5)))
        temp_xj = np.zeros([n,1])
        for o in range(0,n):
            temp_xj[o] = x[i,o] - mu[ind][o]
        ll_1b = np.exp(-0.5*np.matmul(np.matmul(np.transpose(temp_xj),np.linalg.inv(sigma[ind])),(temp_xj)))
        ll_1 = np.log(ll_1a*ll_1b)
        ll += alpha*(ll_1)

    it += 1

h = np.zeros(m)
for i in range(m):
    h[i] = np.argmax(w[i])

ss_em_accuracy = np.mean(h == y)
print('accuracy')
print(ss_em_accuracy)
