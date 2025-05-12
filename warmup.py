import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, e, sqrt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import random


def model(w0,w1,mu,sigma_2,alpha,training_size,sample_sizes,test_size,i):

    beta = 1 / sigma_2
    cov = (1 / alpha) * np.eye(2)
    sigma = np.sqrt(sigma_2)

    # -------- Prior ---------------

    w0list = np.linspace(-2,2,100)
    w1list = np.linspace(-2,2,100)

    W0list,W1list = np.meshgrid(w0list,w1list)
    pos = np.dstack((W0list,W1list))

    prior = multivariate_normal([mu,mu],cov)
    prior_pdf = prior.pdf(pos)
    

    plt.subplot(5, 4, i)
    plt.contour(W0list, W1list, prior_pdf, levels = 10, cmap='viridis')
    plt.grid(True)
    plt.title('Prior')


    # --------- Training Data ----------------
    t_list_training = np.linspace(-1, 1, training_size)
    y_list_training = w0 + w1 * t_list_training
    D_training = y_list_training + np.random.normal(mu, sigma_2, training_size)

    t_list_test = np.linspace(-1.5, 1.5, test_size)
    y_list_test = w0 + w1 * t_list_test
    D_test = y_list_test + np.random.normal(mu, sigma_2, test_size)

    training_data = list(zip(t_list_training, D_training))
    # random.seed(42)

    for index, sample_size in enumerate(sample_sizes, start=1):
        sample = random.sample(training_data, sample_size)
        likelihood = np.zeros(W0list.shape)

        for j in range(W0list.shape[0]):
            for k in range(W0list.shape[1]):
                w0_temp = W0list[j, k]
                w1_temp = W1list[j, k]

                product = 1.0
                for x_i, t_i in sample:
                    WTx = w0_temp + w1_temp * x_i
                    probability = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t_i - WTx)/sigma)**2)
                    product *= probability
                
                likelihood[j, k] = product
        

        plt.subplot(5,4,4+i+(index-1))
        plt.contour(W0list,W1list,likelihood, levels=10,cmap='viridis')

        x_sample = np.array([p[0] for p in sample])
        t_sample = np.array([p[1] for p in sample])


        Xext = np.column_stack((np.ones_like(x_sample), x_sample))

        SN_inv = alpha * np.eye(2) + np.matmul((beta * Xext.T), Xext)

        SN = np.linalg.inv(SN_inv)

        mN = np.matmul(np.matmul((beta * SN), Xext.T), t_sample)


        # print("Posterior mN = ", mN)
        # print("Posterior SN = ", SN)

        posterior = multivariate_normal(mN,SN)
        posterior_pdf = posterior.pdf(pos)

        plt.subplot(5,4,8+i+(index-1))
        plt.contour(W0list, W1list, posterior_pdf, levels = 10, cmap='viridis')

        weighted_samples = np.random.multivariate_normal(mN,SN,size=5)
        x_gird = np.linspace(-1.5,1.5,200)
        plt.subplot(5,4,12+i+(index-1))

        plt.scatter(x_sample, t_sample, c='blue')
        plt.scatter(t_list_test, D_test, c='red')

        plt.plot(x_gird, w0 + w1 * x_gird, c='black', ls='--')

        for w in weighted_samples:
            y_grid = w[0] + w[1] * x_gird
            plt.plot(x_gird, y_grid, linestyle='dashdot')


        Xext_test = np.column_stack((np.ones_like(t_list_test), t_list_test))

        BmN = np.matmul(Xext_test, mN)
        Bvar = np.array([1/beta + np.matmul(np.matmul(ext, SN), ext) for ext in Xext_test])
        std_pred = np.sqrt(Bvar)

        plt.subplot(5,4,16+i+(index-1))
        plt.errorbar(t_list_test, BmN, yerr=std_pred, fmt='o', ms=4, c='green', ecolor='green', capsize=2)

        plt.scatter(t_list_test, D_test, c='orange')


        


            





mu = 0
sigma_2 = 0.2
alpha = 2

w0 = -1.2
w1 = 0.9

sample_sizes = [3,10,20,100]

model(w0,w1,mu,sigma_2,alpha,200,sample_sizes,30,1)

plt.show()