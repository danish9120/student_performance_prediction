from sklearn.neighbors import KNeighborsRegressor
import numpy as np
#features:(study, sleep, attendance)
X = np.array([[2, 6, 60], [4, 7, 70], [6, 6, 70], [8, 8, 90]])
y = np.array([40, 55, 70, 90])
model= KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)
y_pred=model.predict([[5, 7, 80]])
print("Predicted Marks:", y_pred[0])    