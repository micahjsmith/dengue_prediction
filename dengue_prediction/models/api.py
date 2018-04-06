import logging
import os
import pathlib

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.build_features import (
    build_features, build_features_from_dir, build_target)
from dengue_prediction.io import write_tabular
from dengue_prediction.models.modeler import create_model

logger = logging.getLogger(__name__)


def train_model():
    logger.info('Training model...')
    X_df_tr, y_df_tr = load_data()
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)
    model = create_model()
    model.fit(X_tr, y_df_tr)
    logger.info('Training model...DONE')
    return model


def save_model(model, output_dir):
    logger.info('Saving model...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('model.pkl')
    model.dump(filepath)
    logger.info('Saving model...DONE ({})'.format(filepath))


def _predict_model(model, X_te, y_te):
    y_te_pred = model.predict(X_te)
    # TODO add inverse transform
    # y_te_pred = mapper_y.inverse_transform(y_te_pred)
    return y_te_pred


def predict_model(input_dir):
    model = train_model()
    X_te, y_te = build_features_from_dir(input_dir)
    y_te_pred = _predict_model(model, X_te, y_te)
    return y_te_pred


def save_predictions(y, output_dir):
    logger.info('Saving predictions...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('predictions.pkl')
    write_tabular(y, filepath)
    logger.info('Saving predictions...DONE ({})'.format(filepath))
