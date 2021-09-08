from sklearn.model_selection import StratifiedKFold
import astropy.table
import lightgbm
import numpy as np
import os
import pickle


def extract_top_classifications(classifications):
    """Extract the top classification for each row a classifications Table.

    This is a bit complicated when working with astropy Tables.

    Parameters
    ----------
    classifications : `~astropy.table.Table`
        Classifications table output from a `Classifier`

    Returns
    -------
    `numpy.array`
        numpy array with the top type for each light curve
    """
    types = classifications.colnames[1:]
    dtype = classifications[types[0]].dtype
    probabilities = classifications[types].as_array().view((dtype, len(types)))
    top_types = np.array(types)[probabilities.argmax(axis=1)]

    return top_types


def weighted_multi_logloss(true_types, classifications):
    """Calculate a weighted log loss metric.

    This is the metric used for the PLAsTiCC challenge (with class weights set to 1)
    as described in Malz et al. 2019

    Parameters
    ----------
    true_types : `~numpy.ndarray`
        True types for each object
    classifications : `~astropy.table.Table`
        Classifications table output from a `~Classifier`

    Returns
    -------
    [type]
        [description]
    """
    total_logloss = 0.
    unique_types = np.unique(true_types)
    for type_name in unique_types:
        type_mask = true_types == type_name
        type_predictions = classifications[type_name][type_mask]
        type_loglosses = (
            -np.log(type_predictions)
            / len(unique_types)
            / len(type_predictions)
        )
        total_logloss += np.sum(type_loglosses)
    return total_logloss


class Classifier():
    """LightGBM classifier that operates on ParSNIP predictions"""
    def __init__(self):
        # Keys to use
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

    def extract_features(self, predictions):
        """Extract features used for classification

        The features to use are specified by the `keys` attribute.

        Parameters
        ----------
        predictions : `~astropy.table.Table`
            Predictions output from `ParsnipModel.predict_dataset`

        Returns
        -------
        `~numpy.ndarray`
            Extracted features that will be used for classification
        """
        return np.array([predictions[i].data for i in self.keys]).T

    def train(self, predictions, num_folds=10, labels=None, target_label=None,
              reweight=True, min_child_weight=1000.):
        """Train a classifier on the predictions from a ParSNIP model

        Parameters
        ----------
        predictions : `~astropy.table.Table`
            Predictions output from `ParsnipModel.predict_dataset`
        num_folds : int, optional
            Number of K-folds to use, by default 10
        labels : List[str], optional
            True labels for each light curve, by default None
        target_label : str, optional
            If specified, do one-vs-all classification for the given label, by default
            None
        reweight : bool, optional
            If true, weight all light curves so that each type has the same total
            weight, by default True
        min_child_weight : float, optional
            `min_child_weight` parameter for LightGBM, by default 1000

        Returns
        -------
        `~astropy.table.Table`
            K-folding out-of-sample predictions for each light curve
        """
        print("Training classifier with keys:")
        for key in self.keys:
            print(f"    {key}")

        if labels is None:
            # Use default labels
            labels = predictions['type']

        if target_label is not None:
            # Single class classification
            labels = labels == target_label
            class_names = np.array([target_label, 'Other'])

            numeric_labels = (~labels).astype(int)
        else:
            # Multi-class classification
            class_names = np.unique(labels)

            # Assign numbers to the labels so that we can guarantee a consistent
            # ordering.
            label_map = {j: i for i, j in enumerate(class_names)}
            numeric_labels = np.array([label_map[i] for i in labels])

        # Assign folds while making sure that we keep all augmentations of the same
        # object in the same fold.
        if num_folds > 1:
            if 'augmented' in predictions.colnames:
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

        features = self.extract_features(predictions)

        # Normalize by the class counts. We normalize so that the average weight is 1
        # across all objects, and so that the sum of weights for each class is the same.
        if reweight:
            count_names, class_counts = np.unique(numeric_labels, return_counts=True)
            norm = np.mean(class_counts)
            class_weights = {name: norm / count for name, count in zip(count_names,
                                                                       class_counts)}
            weights = np.array([class_weights[i] for i in numeric_labels])
        else:
            weights = np.ones_like(numeric_labels)

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
                    "min_child_weight": min_child_weight,
                }
            else:
                lightgbm_params = {
                    "objective": "multiclass",
                    "num_class": len(class_names),
                    "metric": "multi_logloss",
                    "min_child_weight": min_child_weight,
                }

            train_index = predictions['fold'] != fold
            test_index = ~train_index

            fit_params = {"verbose": 100, "sample_weight": weights[train_index]}

            if num_folds > 1:
                fit_params["eval_set"] = [(features[test_index],
                                           numeric_labels[test_index])]
                fit_params["eval_sample_weight"] = [weights[test_index]]

            classifier = lightgbm.LGBMClassifier(**lightgbm_params)
            classifier.fit(features[train_index],
                           numeric_labels[train_index], **fit_params)

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

        classifications = astropy.table.hstack([
            predictions['object_id'],
            astropy.table.Table(classifications, names=class_names)
        ])

        return classifications

    def classify(self, predictions):
        """Classify light curves using predictions from a `~ParsnipModel`

        If the classifier was trained with K-folding, we average the classification
        probabilities over all folds.

        Parameters
        ----------
        predictions : `~astropy.table.Table`
            Predictions output from `ParsnipModel.predict_dataset`

        Returns
        -------
        Returns
        -------
        `~astropy.table.Table`
            Predictions for each light curve
        """
        features = self.extract_features(predictions)

        classifications = 0.

        for classifier in self.classifiers:
            classifications += classifier.predict_proba(features)

        classifications /= len(self.classifiers)

        classifications = astropy.table.hstack([
            predictions['object_id'],
            astropy.table.Table(classifications, names=self.class_names)
        ])

        return classifications

    def write(self, path):
        """Write the classifier out to disk

        Parameters
        ----------
        path : str
            Path to write to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a classifier that was saved to disk

        Parameters
        ----------
        path : str
            Path where the classifier was saved

        Returns
        -------
        `~Classifier`
            Loaded classifier
        """
        with open(path, 'rb') as f:
            classifier = pickle.load(f)

        return classifier
