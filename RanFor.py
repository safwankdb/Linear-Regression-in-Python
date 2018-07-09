import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

# In[68]:


data = np.array(pd.read_csv('data.csv'))
X = data[:, :-1]
y = data[:, -1]

# In[69]:

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

plt.scatter(X_test[:, 1], y_test, label='Original Data')
plt.scatter(X_test[:, 1], y_pred, label='Predicted Data')
plt.legend()
plt.xlabel('Dose')
plt.ylabel('Output')
plt.savefig('Plot 1 Random Forest.png')
plt.clf()

# In[72]:


plt.scatter(X_test[:, 2], y_test, label='Original Data')
plt.scatter(X_test[:, 2], y_pred, label='Predicted Data')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Output')
plt.savefig('Plot 2 Random Forest.png')
plt.clf()

# In[73]:


plt.scatter(X_test[:, 3], y_test, label='Original Data')
plt.scatter(X_test[:, 3], y_pred, label='Predicted Data')
plt.legend()
plt.xlabel('Angle')
plt.ylabel('Output')
plt.savefig('Plot 3 Random Forest.png')

# In[74]:
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[75]:


print('r2 Score for model is', r2_score(y_test, y_pred))


# In[63]:


print('Mean squared error is', mean_squared_error(y_test, y_pred))
