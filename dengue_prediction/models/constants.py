from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class ClassificationMetricAgg(Enum):
    BINARY_METRIC_AGGREGATION = "micro"
    MULTICLASS_METRIC_AGGREGATION = "micro"


class MetricComputationApproach(Enum):
    CV = 1
    TRAIN_TEST = 2


CLASSIFICATION_SCORING = [
    {"name": "Accuracy", "scoring": "accuracy"},
    {"name": "Precision", "scoring": "precision"},
    {"name": "Recall", "scoring": "recall"},
    {"name": "ROC AUC", "scoring": "roc_auc"},
]

REGRESSION_SCORING = [
    {"name": "Root Mean Squared Error", "scoring": "root_mean_squared_error"},
    {"name": "R-squared", "scoring": "r2"},
]
