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
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)
    model = create_model()
    model.fit(X_tr, y_df_tr)
    logger.info('Training model...DONE')
    return model


def predict_model(test_dir, train_dir=None):
    model = train_model(train_dir=train_dir)

    logger.info('Making predictions...')
    X_te, y_te, mapper_X, mapper_y = build_features_from_dir(
        train_dir, return_mappers=True)
    y_te_pred = model.predict(X_te)
    # TODO forget about this for now
    # y_te_pred = mapper_y.inverse_transform(y_te_pred)
    logger.info('Making predictions...DONE')
    return y_te_pred
