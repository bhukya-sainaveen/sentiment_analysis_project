# classification_algorithms.py

from sklearn.model_selection import train_test_split
from sklearn import tree, svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def split_data(df_input, target, test_size=0.2, random_state=40):
    X_train, X_test, y_train, y_test = train_test_split(df_input, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train.values.ravel(), y_test

def decision_tree_classifier(X_train, y_train, X_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def svm_classifier(X_train, y_train, X_test):
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def print_result(y_test, predictions):
    print('Confusion Matrix:', confusion_matrix(y_test, predictions))
    print('Accuracy Score:', accuracy_score(y_test, predictions))
    print('Classification Report:', classification_report(y_test, predictions, zero_division=0))