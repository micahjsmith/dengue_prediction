from enum import Enum, auto


class ProblemType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()


class ClassificationMetricAggregation(Enum):
    BINARY_METRIC_AGGREGATION = "micro"
    MULTICLASS_METRIC_AGGREGATION = "micro"


class MetricComputationApproach(Enum):
    CV = auto()
    TRAIN_TEST = auto()


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
