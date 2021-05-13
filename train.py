import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


def main():
    """
    This code reads a csv file, split it into a train and test set,
    and save them in a csv file.
    It runs a classifier model on the dataset and saves it to a file on disk named churn_model.pkl
    :return: the string decrypted
    Author:  Elie Ghanassia
    """
    df = pd.read_csv("cellular_churn_greece.csv")
    features = df.columns.tolist()[:-1]
    target = 'churned'

    X = df[features]
    y = df[target]

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        train_size=0.8, stratify=df[target])

    # We choose RandomForestClassifier and found the parameters with a grid search
    clf = RandomForestClassifier(criterion='gini', max_depth=30, min_samples_split=7)

    # fits the model with data
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    y_train.to_csv('y_train.csv', index=False)
    X_train.to_csv('train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    np.savetxt('preds.csv', y_pred)
    pickle.dump(clf, open('churn_model.pkl', 'wb'))


if __name__ == '__main__':
    main()