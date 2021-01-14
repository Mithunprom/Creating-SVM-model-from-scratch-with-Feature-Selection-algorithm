# svm.py
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score ,recall_score
from sklearn.utils import shuffle
import os
os.chdir('C:/Users/mithu/OneDrive/ML/SVM')
df=pd.read_csv('data.txt',header=0)
mapping={'M':1,'B':-1}
df.diagnosis=df.diagnosis.map(mapping)
Y=df.diagnosis
Y=Y.astype('float64')############
df = df.drop(df.columns[[0,1, -1]], axis=1)  # df.columns is zero-based pd.Index
scaleX=MinMaxScaler().fit_transform(df)
X=pd.DataFrame(scaleX)
def correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr().values
    deleted_column=[]
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            if corr[i,j]>=corr_threshold:
                deleted_column.append(j)
    X=X.drop(X.columns[deleted_column], axis=1,inplace=True)
    return X
#backward elemination from p-values
def less_significant_features(X,y):
    alpha=0.05
    col_drop=[]
    for i in range(len(X.columns)):
        reg=sm.OLS(y,X).fit()
        maxcol=reg.pvalues.idxmax()
        maxval=reg.pvalues.max()
        if maxval>=alpha:
            col_drop.append(maxcol)
            X.drop(maxcol, axis='columns', inplace=True)
        else:
            break
        reg.summary()
    return col_drop
drop_col=less_significant_features(X,Y)
X['0']=np.repeat(1,len(X.index))
cols = X.columns.tolist()
cols=list(cols[-1])+cols[0:-1]
X=X[cols]
correlated_features(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
class Svm():
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.w=np.zeros(X.shape[1])
        self.N=self.y.shape[0]
    def predict(self,x,w):
        n=x.shape[0]
        f=np.dot(x,w)
        return np.sign(f)
    def cost(self,C,X,Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, self.w.reshape(-1,1)))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = C * (np.sum(distances) / N)        
        # calculate cost
        cost = 0.5* np.dot(self.w, self.w.T) + hinge_loss
        return cost

    def gradient(self,X,y,C):
        # if only one example is passed (eg. in case of SGD)
        if type(y) == np.float64:
            y = np.array([y])
            X = np.array([X])
        distance=1-y*np.dot(X,self.w.reshape(-1,1))
        dw = np.zeros(self.w.shape[0])
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = self.w
            else:
                di = self.w - (C * y[ind] *X[ind])
            dw += di.reshape(-1,)
            dw = dw/(self.N)  # average
        return dw
    
    def sgd(self,maxepoachs=8000,learning_rate=0.000001,C=10000):
        # stochastic gradient descent
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.0001  # in percent
        for epoch in range(1,maxepoachs):
            X,Y=shuffle(self.X,self.y)
            for ind,x in enumerate(X):
                ascent=self.gradient(x,Y[ind],C)
                self.w = self.w - (learning_rate * ascent)
            # convergence check on 2^nth epoch
            if (epoch == 2 ** nth) or (epoch == maxepoachs - 1):
                cost = self.cost(C,X,Y)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return self.w
                prev_cost = cost
                nth += 1        
        return self.w
# inside init()
# train the model
print("training started...")
model=Svm(X_train.to_numpy(), y_train.to_numpy())
W = model.sgd()
print("training finished.")
print("weights are: {}".format(W))

# inside init()
# testing the model on test set
#y_test_predicted = np.array([])
y_test_predicted=model.predict(X_test, W)

print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
