import pandas as pd
import numpy as np



def prediction1():
  file = pd.read_csv('startup_fail_success.csv')

  file.head()

  y = file.Result

  x = file.drop('Result', axis='columns')

  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()

  dummies = pd.get_dummies(file['State'])

  x = x.drop('State', axis='columns')

  x = pd.concat([x, dummies], axis='columns')

  x.head()

  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

  model.fit(x_train, y_train)

  model.score(x_test, y_test)

  g = model.predict([[350, 146897, 443898, 2269, 0, 0, 1]])

  def get(g):
    if (g == 0):
      import random
      a = random.random()
      return a
    else:
      return 1

  print("The probability of the success of your startup is:", get(g) * 100)

