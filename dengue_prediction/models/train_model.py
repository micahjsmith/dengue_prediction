import logging

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.build_features import (
    build_features, build_target)
from dengue_prediction.models.modeler import create_model

logger = logging.getLogger(__name__)


def train_model():
    logger.info('Training model...')
    X_df_tr, y_df_tr = load_data()
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)
    model = create_model()
    model.fit(X_tr, y_df_tr)
    # todo save model
    logger.info('Training model...DONE')
    return model


if __name__ == '__main__':
    train_model()
