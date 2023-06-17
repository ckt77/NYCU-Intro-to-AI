import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class CarClassifier:
    def __init__(self, model_name, train_data, test_data):
        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        self.x_train = np.array([train_data[0][0].reshape(-1)])
        self.y_train = np.array([train_data[0][1]])
        self.x_test = np.array([test_data[0][0].reshape(-1)])
        self.y_test = np.array([test_data[0][1]])

        for i in range(1, len(train_data)):
            self.x_train = np.concatenate(
                (self.x_train, np.array([train_data[i][0].reshape(-1)])))
            self.y_train = np.concatenate(
                (self.y_train, np.array([train_data[i][1]])))

        for i in range(1, len(test_data)):
            self.x_test = np.concatenate(
                (self.x_test, np.array([test_data[i][0].reshape(-1)])))
            self.y_test = np.concatenate(
                (self.y_test, np.array([test_data[i][1]])))
        # End your code (Part 2-1)

        self.model = self.build_model(model_name)

    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        model = None

        if model_name == 'RF':
            n_estimators = 300
            criterion = 'entropy'
            max_features = 'log2'
            max_samples = 0.8

            model = RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_features=max_features, max_samples=max_samples)

            print("n_estimators =", n_estimators)
            print('criterion =', criterion)
            print('max_features =', max_features)
            print('max_samples =', max_samples)
            print()

        elif model_name == 'KNN':
            n_neighbors = 1
            weights = 'distance'
            p = 1

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, p=p)

            print("n_neighbors =", n_neighbors)
            print('weights =', weights)
            print('p =', p)
            print()

        else:  # model_name == 'AB'
            estimator = RandomForestClassifier(
                n_estimators=10, criterion='entropy')
            n_estimators = 500
            learning_rate = 0.4

            model = AdaBoostClassifier(estimator=estimator,
                                       n_estimators=n_estimators, learning_rate=learning_rate)

            print('n_estimators =', n_estimators)
            print('learning_rate =', learning_rate)
            print()

        return model
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        trained_model = self.model.fit(self.x_train, self.y_train)
        return trained_model
        # End your code (Part 2-3)

    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(self.y_test, y_pred))

    def classify(self, input):
        return self.model.predict(input)[0]
