import logging

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.build_features import (
    build_features, build_features_from_dir, build_target)
from dengue_prediction.models.modeler import create_model

logger = logging.getLogger(__name__)


def train_model(train_dir=None):
    '''Train model using training data found in input_dir'''
    logger.info('Training model...')
    X_df_tr, y_df_tr = load_data(input_dir=train_dir)
    model = _train_model(X_df_tr, y_df_tr)
    logger.info('Training model...DONE')
    return model

def _train_model(X_df_tr, y_df_tr):
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)
    model = create_model()
    model.fit(X_tr, y_df_tr)
    return model


def predict_model(predict_dir, train_dir=None):
    model = train_model(train_dir=train_dir)
    X_te, y_te = build_features_from_dir(input_dir)
    y_te_pred = _predict_model(model, X_te, y_te)
    return y_te_pred


def _predict_model(model, X_te, y_te):
    y_te_pred = model.predict(X_te)
    # TODO add inverse transform
    # y_te_pred = mapper_y.inverse_transform(y_te_pred)
    return y_te_pred
