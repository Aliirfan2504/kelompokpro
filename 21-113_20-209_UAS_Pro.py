#Data ini diambil dari yahoo financial,type data dari dataset ini ada number, kontinu, data in tentang keuangan dalam sebuah perusahaan PT,KAI
import numpy as np
import pandas as pd
df_data = pd.read_csv('KAI.csv')
df_data.head(7)
df_data.shape
df_high= df_data['High']
n_steps = 2
X, y = split_sequence(df_high, n_steps)
print(X.shape, y.shape)
# column names to X and y data frames
df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1,-1)])
df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

# concat df_X and df_y
df = pd.concat([df_X, df_y], axis=1)
# df_X.head()
# df_y.head()
df.head(3)
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
X_norm= scaler.fit_transform(df_X)
y_norm= scaler.fit_transform(df_y)
X_norm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
model_knn = KNeighborsRegressor(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred=model_knn.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test, y_pred)
y_test.shape
y_pred.shape
df_y_test = pd.DataFrame(y_test,columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

df_hasil = pd.concat([df_y_test, df_y_pred], axis=1)
df_hasil