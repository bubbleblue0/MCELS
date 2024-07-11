from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = "MTS_DATA"

name = "ArticularyWordRecognition"

class ArticularyWordRecognitionDataset():
    def __init__(self):
        print("Loading train data . . .")
        self.train_data, self.train_label = self.load_train_data()
        print("Loading test data . . .")
        self.test_data, self.test_label = self.load_test_data()
        self.name = name

    def load_train_data(self):
        train_data, train_label = load_from_tsfile(os.path.join(DATA_PATH, name, name + "_TRAIN.ts"),
                                            return_data_type="numpy3d")
        encoder = LabelEncoder()
        train_label = encoder.fit_transform(train_label)

        return train_data, train_label

    def load_test_data(self):
        test_data, test_label = load_from_tsfile(os.path.join(DATA_PATH, name, name + "_TEST.ts"),
                                          return_data_type="numpy3d")

        encoder = LabelEncoder()
        test_label = encoder.fit_transform(test_label)

        return test_data, test_label

