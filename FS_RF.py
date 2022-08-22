####################### Random Forest algorithm for classification  #######################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class FeatureSelection:

    def __init__(self):
        self.df = pd.read_csv('breast-cancer.csv')

        # change the diagnosis column from M and B to 0 and 1
        self.diagnosis = {'M': 1, 'B': 0}
        self.df.diagnosis = [self.diagnosis[item] for item in self.df.diagnosis]

        del self.df['id']

        self.y = self.df.iloc[:, 0]  # Dependent variable, the diagnosis, is the first one
        self.X = self.df.iloc[:, 1:31]  # Independent variables (first and second variable: age and interest)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=42,
                                                                                shuffle=True)
        self.model = RandomForestClassifier(
            n_estimators=100
            , max_depth=2
            , criterion="gini"
            , random_state=42)
        # Fit the training data into the model and generate the score
        self.model.fit(self.X_train, self.y_train)
        self.model.score(self.X_test, self.y_test)

    def __len__(self):
        return self.X.shape[1]

    def accuracy(self, allFeatures):
        allFeaturesAccuracy = self.model.score(self.X_test, self.y_test)
        return allFeaturesAccuracy


def main():
    fs = FeatureSelection()
    allOnes = [1] * len(fs)

    print(f'The accuracy score is {round(fs.accuracy(allOnes), 5)} when using all {len(fs.df.columns)} columns')


if __name__ == "__main__":
    main()
