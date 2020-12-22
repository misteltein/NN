import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

data = np.loadtxt( 'spiral3.csv', delimiter = ',' )
data_train, data_test = train_test_split(data, test_size=0.25, random_state=0)
y_train = data_train[:, 2].astype(int)
y_test = data_test[:, 2].astype(int)
sc = preprocessing.StandardScaler().fit(data_train[:, :2])
X_train = sc.transform(data_train[:, :2])
X_test = sc.transform(data_test[:, :2])

def tpl(length,num):
    return tuple([length]*num)

for size in range (10,210,10):
    for layer in range (1,11,1):
        clf = MLPClassifier(hidden_layer_sizes=tpl(size,layer),alpha=1.0e-8,max_iter=1000)
        clf.fit( X_train, y_train )
        y_hat = clf.predict(X_train)
        print(size,layer,clf.score(X_train, y_train))
        
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(X_train[:,0],X_train[:,1],c=y_hat,s=1,alpha=0.5)
        plt.savefig('NN_'+str(size)+'_'+str(layer)+'.png',dpi=300)
        plt.clf()
        plt.close()

