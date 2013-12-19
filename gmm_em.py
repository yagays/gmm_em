
import math
import numpy as np

def mv_norm(x, mu, sigma):
	norm1 = 1 / (math.pow(2 * math.pi, len(x)/2.0) * math.pow(np.linalg.det(sigma), 1.0/2.0))
	x_mu = np.matrix(x-mu)
	norm2 = np.exp(-0.5 * x_mu * sigma.I * x_mu.T)
	return float(norm1 * norm2)


data = np.array([[-1,-1],[-1,0],[0,1],[1,1],[1,2]])

K = 2
N = len(data)

mu = [np.array([0, 0]), np.array([1, 0])]
sigma = [np.eye(2), np.eye(2)]
pi_k = [0.5, 0.5]

L = []
mu_iter = []
sigma_iter = []
pi_k_iter = []
diff = 1

while diff > 0.1 :

	# E-step
	likelihood = np.zeros((N,K))
	gamma_nk = np.zeros((N,K))
	for k in range(K):
		likelihood[:,k] = [mv_norm(d, mu[k], np.array(sigma[k]))*pi_k[k] for d in data]

	for n in range(N):
		gamma_nk[n,:] = likelihood[n,:] / sum(likelihood[n,:])

	# M-step
	N_k = np.array([sum(gamma_nk[:,k]) for k in range(K)])

	pi_k = N_k/sum(N_k)

	mu = np.dot(gamma_nk.T, data)/N_k

	for k in range(K):
		sig = 0
		for n in range(N):
			x_mu = data[n,:] - mu[:,k]
			sig += gamma_nk[n,k] * np.outer(x_mu, x_mu.T)
		sigma[k] = np.array(sig/N_k[k])

	# iter
	mu_iter.append(mu)
	sigma_iter.append(sigma)
	pi_k_iter.append(pi_k)

	l = sum(map(np.log,[sum(likelihood[n,:]) for n in range(N)]))/N
	print l
	if L:
		diff = math.fabs(L[-1] - l)
		L.append(l)
	else:
		L.append(l)
