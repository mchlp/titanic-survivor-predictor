import pandas as pd
import utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "SibSp", "Parch"]].values

classifer = linear_model.LogisticRegression(max_iter=50000)

classifier_ = classifer.fit(features, target)
print(classifier_.score(features, target))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifier_ = classifer.fit(poly_features, target)
print(classifier_.score(poly_features, target))
