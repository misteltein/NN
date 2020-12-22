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

best_score = 0.0
best_size  = 0
best_layer = 0

print('size','layer','score')
for size in range (10,110,10):
    for layer in range (1,11,1):
        clf = MLPClassifier(hidden_layer_sizes=tpl(size,layer),alpha=1.0e-8,max_iter=1000)
        clf.fit( X_train, y_train )
        y_hat = clf.predict(X_train)
        score = clf.score(X_train, y_train)
        if best_score< score:
            best_score = score
            best_size = size
            best_layer = layer
        print(size,layer,score,flush=True)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(X_train[:,0],X_train[:,1],c=y_hat,s=1,alpha=0.5)
        plt.savefig('NN_'+str(size)+'_'+str(layer)+'.png',dpi=300)
        plt.clf()
        plt.close()

print('best_size','best_layer','best_score')
print(best_size,best_layer,best_score)

