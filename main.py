import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


data = np.loadtxt( 'sample2.csv', delimiter = ',' )
data_train, data_test = train_test_split(data, test_size=0.25, random_state=0)
y_train = data_train[:, 2].astype(int)
y_test = data_test[:, 2].astype(int)
sc = preprocessing.StandardScaler().fit(data_train[:, :2])
X_train = sc.transform(data_train[:, :2])
X_test = sc.transform(data_test[:, :2])

params = {
    'hidden_layer_sizes': [(10,10,10,10,10,10,10,10,10,10,10)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0000001],
    'learning_rate': ['adaptive'],
}

#model = MLPClassifier(max_iter=1000)
#gscv = GridSearchCV(model,params,cv=1,verbose=3,scoring='accuracy',n_jobs=-1)
#gscv = MLPClassifier(max_iter=1000,hidden_layer_sizes=(10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10),alpha=0.0000001,verbose=True)
#gscv = MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100),alpha=0.0000001,verbose=True)#0.99312
#gscv = MLPClassifier(max_iter=1000,hidden_layer_sizes=(200,200,200,200,200,200,200,200,200,200),alpha=0.0000001,verbose=True,learning_rate='adaptive')#0.98404
#gscv = MLPClassifier(max_iter=1000,hidden_layer_sizes=(50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50),alpha=0.0000001,verbose=True,learning_rate='adaptive')#0.98716
gscv = MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100),alpha=0.0000001,verbose=True,learning_rate='adaptive')#0.98716
gscv.fit( X_train, y_train )

print(gscv.score(X_train, y_train))
print(gscv.score(X_test, y_test))
y_hat = gscv.predict(X_test)
print(y_hat)

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(X_test[:,0],X_test[:,1],c=y_hat,s=1)
plt.show()


