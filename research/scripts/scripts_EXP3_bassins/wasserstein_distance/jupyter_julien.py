import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import sklearn.manifold as sm

a = 1
b = 10
epsilon = 0.01
Ns = 100
d = 1000

data = np.zeros((d,Ns))


for s in range(Ns):
    mu_s = epsilon*np.random.randn(1,1)
    if s < Ns/2:
        sigma_s = a * np.log(s+1) + b
        #sigma_s = a * (s+1) + b
    else:
        sigma_s = a * np.log(Ns/2+1)+b
        #sigma_s = a * (Ns/2+1) +b
    data[:,s] = mu_s + sigma_s*np.random.randn(d,)

#plt.boxplot(data)


Distances = np.zeros((Ns,Ns))
for s1 in range(Ns):
    for s2 in range(Ns):
        Distances[s1,s2] = ss.wasserstein_distance(data[:,s1],data[:,s2])

plt.imshow(Distances)
plt.title("Pairwise Wasserstein distances ")