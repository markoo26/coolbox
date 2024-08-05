from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
import plotly.express as px
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from .logging import step_time_calculation, list_devices
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


class UltimateClassifier():
    """
        A comprehensive classifier that integrates multiple machine learning models and evaluation metrics.

        This class supports the following functionalities:
        - Preprocessing data with various scaling techniques.
        - Splitting data into training and testing sets.
        - Fitting multiple classifiers including AdaBoost, RandomForest, KNeighbors, etc.
        - Generating predictions and evaluating model performance using specified metrics.
        - Blending predictions from different models.

        Parameters:
        -----------
        X : array-like
            Features for training and testing the models.
        y : array-like
            Target variable for classification.
        scaling : str, optional
            Scaling method to apply ('min_max_scaler', 'max_abs_scaler', 'normalizer'). Default is 'min_max_scaler'.
        test_size : float, optional
            Fraction of data to use for testing. Default is 0.2.
        use_gpu_if_available : bool, optional
            Flag to indicate whether to use GPU if available. Default is True.

        Attributes:
        -----------
        classifier_methods : list
            List of classifier methods available in the class.
        metrics_dict : dict
            Dictionary of metrics functions used for evaluation.
        transformations : dict
            Dictionary of scaling transformations available.
        """

    # Comment out or remove any classifier attribute of the class to not include it during the training
    def __init__(self, X, y, scaling='min_max_scaler', test_size=0.2, use_gpu_if_available=True):
        self.task_type = 'GPU' if use_gpu_if_available and list_devices()['GPU'] else 'CPU'
        self.verbose_trainings = False
        self.ada_boost_classifier = AdaBoostClassifier()
        self.random_forest_classifier = RandomForestClassifier()
        self.k_neighbors_classifier = KNeighborsClassifier()
        self.support_vector_classifier = SVC()
        self.bayes_classifier = GaussianNB()
        self.mlp_classifier = MLPClassifier()
        self.logreg_classifier = LogisticRegression()
        self.catboost_classifier = CatBoostClassifier(task_type=self.task_type)
        self.xgboost_classifier = XGBClassifier()
        self.lgbm_classifier = LGBMClassifier(verbosity=-1 if not self.verbose_trainings else 1)
        self.X = X
        self.y = y

        # Dict of items to store different versions of train and test sets depending on scaling used
        self.X_train = {}
        self.y_train = {}
        self.X_test = {}
        self.y_test = {}
        self.X_transformed = {}

        # Support of metrics that don't require probabilities, just labels
        self.metrics_dict = {'accuracy': sklearn.metrics.accuracy_score,
                             'roc_auc': sklearn.metrics.roc_auc_score}

        self.test_size = test_size
        self.scaling = scaling
        self.transformations = {'min_max_scaler': MinMaxScaler(),
                                'max_abs_scaler': MaxAbsScaler(),
                                'normalizer': Normalizer()}

        self.classifier_methods = [method for method in dir(self) if 'classifier' in method]

        print(f'Utilizing {self.task_type} for all trainings')
    @step_time_calculation(step_name='transform_X')
    def transform_X(self):
        for key, value in self.transformations.items():
            scaler = self.transformations[key].fit(self.X)
            self.X_transformed[key] = scaler.transform(self.X)

    @step_time_calculation(step_name='build_train_sets')
    def build_train_test_sets(self):
        for key in self.X_transformed.keys():
            self.X_train[key], self.X_test[key], self.y_train[key], self.y_test[key] = train_test_split(
                self.X_transformed[key],
                self.y,
                test_size=self.test_size)

    @step_time_calculation(step_name='fit_models')
    def fit_models(self):
        for model_index, method in enumerate(self.classifier_methods):

            print(f"Fitting {method}, which is {model_index + 1} out of {len(self.classifier_methods)} models")

            if method in ['catboost_classifier', 'xgboost_classifier']:
                getattr(self, method).fit(X=self.X_train[self.scaling], y=self.y_train[self.scaling],
                                          verbose=self.verbose_trainings)
            else:
                getattr(self, method).fit(X=self.X_train[self.scaling], y=self.y_train[self.scaling])

    @step_time_calculation(step_name='cross_val_models')
    def cross_val_models(self):
        print('To be implemented')

    @step_time_calculation(step_name='generate_predictions')
    def generate_predictions(self):

        self.predictions = pd.DataFrame()
        for method in self.classifier_methods:
            if method == 'catboost_classifier':
                preds = getattr(self, method).predict(data=self.X_test[self.scaling])
                self.predictions[method] = preds
            else:
                preds = getattr(self, method).predict(X=self.X_test[self.scaling])
                self.predictions[method] = preds
        self.predictions['target'] = self.y_test[self.scaling].values
        self.predictions['index'] = self.predictions.index

    @step_time_calculation(step_name='generate_metrics')
    def generate_metrics(self, plot_height=1000, plot_width=1000):
        self.metrics = pd.DataFrame(columns=['method'] + list(self.metrics_dict.keys()))
        for method in self.predictions.columns:
            if method not in ['target', 'index']:
                self.metrics.loc[len(self.metrics)] = [method] + [
                    metric_fn(self.predictions['target'], self.predictions[method]) for metric_fn in
                    self.metrics_dict.values()]

        self.metrics = self.metrics.melt(id_vars=['method'],
                                         value_vars=[col for col in self.metrics.columns if
                                                     col != 'method'])
        self.metrics.columns = ['model', 'measure', 'measure_value']

        print(f'Average metrics generated using {self.scaling} transformation:\n')
        print(self.metrics[['measure', 'measure_value']].groupby(by='measure').mean())
        print('\n')

        fig = px.bar(self.metrics, x='model', y='measure_value', color='model', facet_col='measure',
                      facet_col_wrap=1,
                      title=f'Accuracy comparison across different models using {self.scaling} and test_size = {self.test_size}')

        fig.update_layout(height=plot_height, width=plot_width)
        return fig
    @step_time_calculation(step_name='extend_training_records')
    def extend_training_records(self):

        self.X_train_extended = pd.DataFrame(self.X_train[self.scaling].copy())
        self.X_test_extended = pd.DataFrame(self.X_test[self.scaling].copy())
        for method in self.classifier_methods:
            # Catboost has 'data' argument to 'predict' method instead of 'X'
            if method == 'catboost_classifier':
                self.X_train_extended[method] = getattr(self, method).predict(data=self.X_train[self.scaling]).astype(
                    'int32')
                self.X_test_extended[method] = getattr(self, method).predict(data=self.X_test[self.scaling]).astype(
                    'int32')
            else:
                self.X_train_extended[method] = getattr(self, method).predict(X=self.X_train[self.scaling]).astype(
                    'int32')
                self.X_test_extended[method] = getattr(self, method).predict(X=self.X_test[self.scaling]).astype(
                    'int32')

    @step_time_calculation(step_name='blend_predictions')
    def blend_predictions(self, models=['catboost_classifier', 'lgbm_classifier'],
                          ratios=[x / 10 for x in range(10) if x != 0]):

        for model in models:
            second_model = [second_model for second_model in models if second_model != model][0]
            for ratio in ratios:
                self.predictions[f"{model}_{round(ratio, 1)}_{second_model}_{round(1 - ratio, 1)}"] = self.predictions[
                                                                                                          model] * ratio + \
                                                                                                      self.predictions[
                                                                                                          second_model] * 1 - ratio
                self.predictions[f"{model}_{round(ratio, 1)}_{second_model}_{round(1 - ratio, 1)}"] = self.predictions[
                    f"{model}_{round(ratio, 1)}_{second_model}_{round(1 - ratio, 1)}"].astype('int32')

@step_time_calculation(step_name='Catboost Single Training')
def train_catboost(X, y, categorical_feats, params = {'verbose': 0}):
    metrics = {'recall': [], 'precision': [], 'accuracy': [],
               'roc_auc': [], 'f1_score': []}

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    train = Pool(x_train, label=y_train, cat_features=categorical_feats)
    valid = Pool(x_valid, label=y_valid, cat_features=categorical_feats)

    model = CatBoostClassifier(**params)
    model.fit(train)
    y_pred = model.predict(valid)

    metrics['accuracy'].append(accuracy_score(y_pred, y_valid))
    metrics['precision'].append(precision_score(y_pred, y_valid))
    metrics['recall'].append(recall_score(y_pred, y_valid))
    metrics['roc_auc'].append(roc_auc_score(y_pred, y_valid))
    metrics['f1_score'].append(f1_score(y_pred, y_valid))

    print('Catboost Model accuracy score: {0:0.4f}'.format(metrics['accuracy'][-1]))
    print('Catboost Model precision score: {0:0.4f}'.format(metrics['precision'][-1]))
    print('Catboost Model recall score: {0:0.4f}'.format(metrics['recall'][-1]))
    print('Catboost Model ROC-AUC score: {0:0.4f}'.format(metrics['roc_auc'][-1]))
    print('Catboost Model F1 score: {0:0.4f}'.format(metrics['f1_score'][-1]))

    for metric_name, metric_value in metrics.items():
        metrics[metric_name] = round(np.mean(metrics[metric_name]), 4)

    return model, metrics

