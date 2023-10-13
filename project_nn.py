import matplotlib.pyplot as plt
import numpy as np
import math
import util


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1/(1+np.exp(-x))


def neural_network(file_path, lr_0, lr_1, p_train_1, p_train_2, idx_del):
    # Data
    x_train, y_train, x_dev, y_dev = data_pro(file_path, p_train_1, p_train_2, idx_del)
    # Seed
    np.random.seed(1)

    syn0 = 2 * np.random.random((x_train.shape[1], 6)) - 1
    syn1 = 2 * np.random.random((6, 1)) - 1

    plot_axis = []
    plot_at = []
    plot_ad = []
    # Training
    for j in range(18000):
        # Layers
        l0 = x_train
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        # Back propagation
        l2_error_reg = y_train - l2
        l2_error = l2_error_reg + lr_0 * np.linalg.norm(syn0) + lr_1 * np.linalg.norm(syn1)
        l2_delta = l2_error * sigmoid(l2, deriv=True)

        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        # Update weights (synapses)
        syn0 += lr_0 * l0.T.dot(l1_delta)
        syn1 += lr_1 * l1.T.dot(l2_delta)

        if (j % 100) == 0:
            plot_axis.append(j)
            # print('Error-', j, ' is: ', str(np.mean(np.abs(l2_error))))

            predict_ytrain = np.zeros_like(l2)
            for i in range(len(l2)):
                if l2[i] >= 0.5:
                    predict_ytrain[i] = 1
                else:
                    predict_ytrain[i] = 0
            accuracy_train = np.mean(y_train == predict_ytrain)

            # Check dev set
            c0 = x_dev
            c1 = sigmoid(np.dot(c0, syn0))
            c2 = sigmoid(np.dot(c1, syn1))

            predict_ydev = np.zeros_like(c2)
            for i in range(len(c2)):
                if c2[i] >= 0.5:
                    predict_ydev[i] = 1
                else:
                    predict_ydev[i] = 0
            accuracy_dev = np.mean(y_dev == predict_ydev)

            plot_at.append(accuracy_train)
            plot_ad.append(accuracy_dev)
            print('Accuracy iteration-', j, ' is: ', accuracy_train, '-', accuracy_dev)

    max_accuracy = np.max(plot_ad)
    print(np.max(plot_at))

    return max_accuracy, plot_axis, plot_at, plot_ad, predict_ytrain, predict_ydev, y_train, y_dev


def data_pro(file_path, p_train_1, p_train_2, idx_del):
    # Data
    train_path = file_path
    x_input, y_input = util.load_dataset(train_path, add_intercept=True)

    x = x_input
    y = np.vstack(y_input)

    # Normalization
    x[:, 1] *= 1
    x[:, 7] *= 10
    x[:, 8] *= 10
    x[:, 9] *= 5
    x[:, 12] *= 0.015
    print(x_input.shape)

    # Separate into train set and dev set
    n_data = x.shape[0]
    n_train_1 = math.ceil(p_train_1 * n_data)
    n_dev_1 = n_data - n_train_1

    n_train_2 = math.ceil(p_train_2 * n_data)
    n_dev_2 = n_data - n_train_2

    # Choose features
    # 0: intercept
    # 1: x_ArtistScore
    # 2: x_Danceability
    # 3: x_Energy
    # 4: x_Key
    # 5: x_Loudness
    # 6: x_Mode
    # 7: x_Speechiness
    # 8: x_Acousticness
    # 9: x_Instrumentalness
    # 10: x_Liveness
    # 11: x_Valence
    # 12: x_Tempo

    x_f = np.delete(x, idx_del, axis=1)

    x_train_1 = np.delete(x_f, np.s_[n_train_1:n_data], axis=0)
    y_train_1 = np.delete(y, np.s_[n_train_1:n_data], axis=0)
    x_dev_1 = np.delete(x_f, np.s_[0:n_train_1], axis=0)
    y_dev_1 = np.delete(y, np.s_[0:n_train_1], axis=0)

    x_train_2 = np.delete(x_f, np.s_[n_train_2:n_data], axis=0)
    y_train_2 = np.delete(y, np.s_[n_train_2:n_data], axis=0)
    x_dev_2 = np.delete(x_f, np.s_[0:n_train_2], axis=0)
    y_dev_2 = np.delete(y, np.s_[0:n_train_2], axis=0)

    x_train = x_train_1
    y_train = y_train_1
    x_dev = x_dev_2
    y_dev = y_dev_2

    return x_train, y_train, x_dev, y_dev


all_path = ['/Users/marcellasuta/Downloads/ps1/data/complete_project_data_no_date.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_1990_1994.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_1995_1999.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_2000_2004.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_2005_2009.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_2010_2014.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_2015_2018.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_06_08.csv',
            '/Users/marcellasuta/Downloads/ps1/data/complete_project_data_11_01.csv']

lr0 = 0.005
lr1 = 0.003

"""idx = [1, 3, 4, 5, 6, 9, 10, 12]
idx_delete = []
for i in range(len(idx)):
    idx_d = np.delete(idx, i, axis=0)
    max_acc,_,_,_ = neural_network(all_path[0], lr0, lr1, 0.8, 0.8, idx_d)
    print('Accuracy for feature', idx[i], 'is:', max_acc)
    idx_delete.append(idx_d)"""

# print(idx_delete)
# max_acc = neural_network(lr0, lr1, 0.8, 0.8, idx_delete)
# print('Accuracy for lr =', lr0, 'and', lr1, 'is:', max_acc)

idx_1 = np.array([4, 5, 6, 10])
idx_2 = np.array([4, 5, 9, 12])

"""Accuracy = []
for f_path in all_path:
    print(f_path)
    max_acc,_,_,_ = neural_network(f_path, lr0, lr1, 0.8, 0.8, idx_1)
    print('Accuracy:', max_acc)
    Accuracy.append(max_acc)"""

max_acc, plot_axis, plot_at, plot_ad, pred_train, pred_dev, y_t, y_d = neural_network(all_path[0], lr0, lr1, 0.8, 0.8, idx_1)

# check for training set
true_pos_t = 0
true_neg_t = 0
false_pos_t = 0
false_neg_t = 0
for i in range(len(pred_train)):
    if y_t[i] == 1:
        if pred_train[i] == y_t[i]:
            true_pos_t += 1
        else:
            false_pos_t += 1
    if y_t[i] == 0:
        if pred_train[i] == y_t[i]:
            true_neg_t += 1
        else:
            false_neg_t += 1

precision_t = true_pos_t / (true_pos_t + false_pos_t)
recall_t = true_pos_t / (true_pos_t + false_neg_t)

# check for dev set
true_pos_d = 0
true_neg_d = 0
false_pos_d = 0
false_neg_d = 0
for i in range(len(pred_dev)):
    if y_t[i] == 1:
        if pred_dev[i] == y_d[i]:
            true_pos_d += 1
        else:
            false_pos_d += 1
    if y_t[i] == 0:
        if pred_dev[i] == y_d[i]:
            true_neg_d += 1
        else:
            false_neg_d += 1

precision_d = true_pos_d / (true_pos_d + false_pos_d)
recall_d = true_pos_d / (true_pos_d + false_neg_d)
print('Positive example = ', true_pos_t + false_pos_t)

print('Sanity check = ', true_pos_t + true_neg_t + false_pos_t + false_neg_t)
print('Sanity check = ', true_pos_d + true_neg_d + false_pos_d + false_neg_d)
print(precision_t, recall_t, precision_d, recall_d)

print(true_pos_d, false_pos_d, true_neg_d, false_neg_d)

"""plt.figure()
plt.title('Iteration Accuracy')
plt.plot(plot_axis, plot_at)
plt.plot(plot_axis, plot_ad)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('/Users/marcellasuta/Downloads/ps1/data/test.png')"""

# np.savetxt('/Users/marcellasuta/Downloads/ps1/data/axis_noreg.txt', Accuracy)