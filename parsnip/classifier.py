from sklearn.model_selection import StratifiedKFold
import lightgbm
import numpy as np
import os
import pandas as pd
import pickle


class Classifier():
    def __init__(self):
        # Keys to use
        # TODO: make this something that can be changed.
        self.keys = [
            'color',
            'color_error',
            's1',
            's1_error',
            's2',
            's2_error',
            's3',
            's3_error',
            'luminosity',
            'luminosity_error',
            'reference_time_error',
        ]

    def train(self, predictions, num_folds=10, labels=None, target_label=None):
        """Train a classifier on the predictions from a VAE model."""
        print("Training classifier with keys:")
        for key in self.keys:
            print(f"    {key}")

        if labels is None:
            # Use default labels
            labels = predictions['label']

        if target_label is not None:
            # Single class classification
            labels = labels == target_label
            class_names = np.array(['Other', target_label])
        else:
            # Multi-class classification
            class_names = np.unique(labels)

        # Assign numbers to the labels so that we can guarantee a consistent ordering.
        label_map = {j: i for i, j in enumerate(class_names)}
        numeric_labels = labels.replace(label_map)

        # Assign folds while making sure that we keep all augmentations of the same
        # object in the same fold.
        if num_folds > 1:
            if 'augmented' in predictions:
                original_mask = ~predictions['augmented']
            else:
                original_mask = np.ones(len(predictions), dtype=bool)

            object_ids = predictions['original_object_id'][original_mask]
            original_labels = numeric_labels[original_mask]

            kf = StratifiedKFold(num_folds, random_state=1, shuffle=True)

            fold_map = {}

            for fold_idx, (train_index, test_index) in enumerate(
                    kf.split(object_ids, original_labels)):
                test_ids = object_ids[test_index]
                for tid in test_ids:
                    fold_map[tid] = fold_idx

            predictions['fold'] = [fold_map[i] for i in
                                   predictions['original_object_id']]
        else:
            predictions['fold'] = -1

        classifiers = []

        features = predictions[self.keys]

        # Normalize by the class counts. We normalize so that the average weight is 1
        # across all objects, and so that the sum of weights for each class is the same.
        class_counts = numeric_labels.value_counts()
        class_weights = np.mean(class_counts) / class_counts
        weights = np.array([class_weights[i] for i in numeric_labels])

        # Calculate out-of-sample classifications with K-fold cross-validation if we
        # are doing that.
        if num_folds > 1:
            classifications = np.zeros((len(numeric_labels), len(class_names)))

        for fold in range(num_folds):
            if target_label is not None:
                # Single class classification
                lightgbm_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "min_child_weight": 1000.,
                }
            else:
                lightgbm_params = {
                    "objective": "multiclass",
                    "num_class": len(class_names),
                    "metric": "multi_logloss",
                    "min_child_weight": 1000.,
                }

            train_index = predictions['fold'] != fold
            test_index = ~train_index

            fit_params = {"verbose": 100, "sample_weight": weights[train_index]}

            if num_folds > 1:
                fit_params["eval_set"] = [(features[test_index].values,
                                           numeric_labels[test_index])]
                fit_params["eval_sample_weight"] = [weights[test_index]]

            classifier = lightgbm.LGBMClassifier(**lightgbm_params)
            classifier.fit(features[train_index], numeric_labels[train_index],
                           **fit_params)

            classifiers.append(classifier)

            if num_folds > 1:
                # Out of sample predictions
                classifications[test_index] = classifier.predict_proba(
                    features[test_index]
                )

        # Keep the trained classifiers
        self.classifiers = classifiers
        self.class_names = class_names

        if num_folds == 1:
            # Only had a single fold, so do in sample predictions
            classifications = classifiers[0].predict_proba(features)

        classifications = pd.DataFrame(
            classifications,
            index=predictions.index,
            columns=class_names
        )

        return classifications

    def classify(self, predictions):
        """Classify objects using predictions from a VAE model.

        If the classifier was trained with K-folding, we average the classification
        probabilities over all folds.
        """
        features = predictions[self.keys]

        classifications = 0.

        for classifier in self.classifiers:
            classifications += classifier.predict_proba(features)

        classifications /= len(self.classifiers)

        classifications = pd.DataFrame(
            classifications,
            index=predictions.index,
            columns=self.class_names
        )

        return classifications

    def write(self, name):
        path = f'./classifiers/classifier_{name}.pkl'

        os.makedirs('./classifiers', exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name):
        path = f'./classifiers/classifier_{name}.pkl'

        with open(path, 'rb') as f:
            classifier = pickle.load(f)

        return classifier
