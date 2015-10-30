import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import ProjectedGradientNMF
model = ProjectedGradientNMF(n_components=10, init='random', random_state=0)
model.fit(X)

print model.components_
U = X.dot(model.components_.T)
print U
print U.dot(model.components_)
model.reconstruction_err_

model = ProjectedGradientNMF(
    n_components=2, sparseness='components', init='random', random_state=0)
model.fit(X)
ProjectedGradientNMF(beta=1, eta=0.1, init='random', max_iter=200,
                     n_components=2, nls_max_iter=2000, random_state=0,
                     sparseness='components', tol=0.0001)
model.components_
model.reconstruction_err_
