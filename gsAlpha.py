import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

data = np.loadtxt( 'spiral3.csv', delimiter = ',' )
data_train, data_test = train_test_split(data, test_size=0.25, random_state=0)
y_train = data_train[:, 2].astype(int)
y_test = data_test[:, 2].astype(int)
sc = preprocessing.StandardScaler().fit(data_train[:, :2])
X_train = sc.transform(data_train[:, :2])
X_test = sc.transform(data_test[:, :2])

params = {
    'hidden_layer_sizes': [(10,10)],
    'alpha': np.logspace(-8,0,10)
}
model = MLPClassifier(max_iter=1000)
clf = GridSearchCV(model,params,cv=5,verbose=3,scoring='accuracy',n_jobs=-1)

clf.fit( X_train, y_train )
y_hat = clf.predict(X_test)

print('\nResult:')
print('\tBest Score: ', clf.best_score_)
print('\tBest Params: ', clf.best_params_)
print('\tAccuracy (test): ', accuracy_score(y_test, y_hat))

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(X_test[:,0],X_test[:,1],c=y_hat,s=1,alpha=0.5)
plt.show()


