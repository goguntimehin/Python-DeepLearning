# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

dataset = pd.read_csv(r'C:\Users\Josh\Downloads\iris.csv')
print(dataset["Species"].value_counts())
X_train = dataset.drop("Species", axis=1)
y_train = dataset["Species"]
X_train, X_test, y_train, y_test= train_test_split(X_train, y_train, test_size=0.4, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
Y_pred = gnb.predict(X_test)
acc_gnb = round(gnb.score(X_test, y_test) * 100)
print("gnb accuracy is:", acc_gnb)
print(dataset.plot(kind="scatter", x="Sepal.Length", y="Sepal.Width"))
print(sns.pairplot(dataset, hue="Species", size=3))
plt.show()

