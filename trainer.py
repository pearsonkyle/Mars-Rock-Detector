import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay

class Ensembler():
    def __init__(self, models):
        self.models = models
        self.acc = [] # validation accuracy

    def fit(self, X, y, preprocess=True):
        self.xdim = X.shape[1]

        # preprocess data by subtracting mean and dividing by std
        if preprocess:
            Xp = (X - X.mean(axis=1, keepdims=True)) / (1+X.std(axis=1, keepdims=True))

        for i, model in enumerate(self.models):

            # split into training and validation sets
            X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.15, random_state=i)

            # train the model
            model.fit(X_train, y_train)

            # compute validation accuracy
            y_pred = model.predict(X_test)
            self.acc.append(accuracy_score(y_test, y_pred))

    def predict(self, X, prob=False):
        # predict with each model and return the average
        Xp = (X - X.mean(axis=1, keepdims=True)) / (1+X.std(axis=1, keepdims=True))
        if prob:
            predictions = np.array([model.predict_proba(Xp) for model in self.models])
        else:
            predictions = np.array([model.predict(Xp) for model in self.models])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def predict_best(self, X, prob=False):
        # predict with each model and return the average
        Xp = (X - X.mean(axis=1, keepdims=True)) / (1+X.std(axis=1, keepdims=True))
        if prob:
            predictions = self.best.predict_proba(Xp)
        else:
            predictions = self.best.predict(Xp)
        return predictions

    @property
    def best(self):
        # return the best model
        return self.models[np.argmax(self.acc)]

    def plot_roc(self, X, y):
        # preprocess data by subtracting mean and dividing by std
        Xp = (X - X.mean(axis=1, keepdims=True)) / (1+X.std(axis=1, keepdims=True))
        
        # overplot the ROC curve for each model in the ensemble
        fig,ax = plt.subplots(figsize=(6,6))
        for i in range(len(self.models)):
            
            y_score = self.models[i].predict_proba(Xp)
            RocCurveDisplay.from_predictions(
                y, y_score[:, 1],
                name=f"Model {i}",
                #color=plt.cm.tab10(i),
                color=plt.cm.jet(i/len(self.models)),
                alpha=0.7,
                ax=ax
            )

        ax.grid(True, ls='--')
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        ax.axis("square")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC curves for each model in the ensemble")
        plt.legend()
        plt.tight_layout()
        plt.show()
    

def train_ensembler(train_file, test_file, class_key='label', plot=False):
    # Load a dataset in a Pandas dataframe.
    rock_df = pd.read_csv(train_file)

    # randomize the order of the data
    rock_df = rock_df.sample(frac=1).reset_index(drop=True)

    # split into labels and data arrays
    rock_label = rock_df[class_key].values.astype(np.float32)
    rock_data = rock_df.drop(class_key, axis=1).values

    print(f"Training on {rock_data.shape[0]} images with {rock_data.shape[1]} features each...")

    # create an ensemble of models
    models = Ensembler([RandomForestClassifier(class_weight='balanced') for i in range(9)])

    # train the ensemble
    models.fit(rock_data, rock_label)

    # print accuracy of each model
    print(models.acc)

    # roc plot
    test_df = pd.read_csv(test_file)
    test_label = test_df[class_key].values.astype(np.float32)
    test_data = test_df.drop(class_key, axis=1).values

    if plot:
        models.plot_roc(test_data, test_label)

    return models


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ws", "--windowsize", type=int, default=11,
            help="size of training sample output in px") # size of training data

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    ensemble = train_ensembler(
        f'training/training_data_{args.windowsize**2}.csv',
        f'training/testing_data_{args.windowsize**2}.csv')
