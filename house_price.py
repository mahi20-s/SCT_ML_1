import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y_train = train["SalePrice"]

X_test = test[["GrLivArea", "BedroomAbvGr", "FullBath"]]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({"Id": test["Id"],"SalePrice": predictions})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")