
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pickle

data = np.array(pd.read_csv('data.csv'))
X = data[:,:-1]
y = data[:,-1]

# In[69]:
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(100,30,30,30,30),max_iter=1000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
plt.scatter(X_test[:,1],y_test,label='Original Data')
plt.scatter(X_test[:,1],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Dose')
plt.ylabel('Output')
plt.savefig('Plot 1.png')


# In[72]:


plt.scatter(X_test[:,2],y_test,label='Original Data')
plt.scatter(X_test[:,2],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Output')
plt.savefig('Plot 2.png')


# In[73]:


plt.scatter(X_test[:,3],y_test,label='Original Data')
plt.scatter(X_test[:,3],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Angle')
plt.ylabel('Output')
plt.savefig('Plot 3.png')

# In[74]:
# In[75]:
print('r2 Score for model is',r2_score(y_test,y_pred))


# In[63]:


print('Mean squared error is',mean_squared_error(y_test,y_pred))
