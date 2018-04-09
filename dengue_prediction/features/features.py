import logging
import pkgutil

import funcy
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing

import dengue_prediction.features.contrib
from fhub_core.feature import Feature
from dengue_prediction.features.transformers import (
    IdentityTransformer, LagImputer, NullFiller, NullIndicator,
    SimpleFunctionTransformer, SingleLagger)

logger = logging.getLogger(__name__)


def _import_names_from_module(importer, modname, required, optional):
    mod = importer.find_module(modname).load_module(modname)

    msg = funcy.partial(
        'Required variable {varname} not found in module {modname}'
        .format, modname=modname)

    # required vars
    if required:
        required_vars = {}
        if isinstance(required, str):
            required = [required]
        for varname in required:
            if hasattr(mod, varname):
                required_vars[varname] = getattr(mod, varname)
            else:
                raise ImportError(msg(varname=varname))
    else:
        required_vars = None

    # optional vars
    if optional:
        if isinstance(optional, str):
            optional = [optional]
        optional_vars = {k: getattr(mod, k)
                         for k in optional if hasattr(mod, k)}
    else:
        optional_vars = None

    return required_vars, optional_vars


def import_contrib_feature_from_components(importer, modname):
    required = ['input', 'transformer']
    optional = ['name', 'description', 'output', 'options']
    required_vars, optional_vars = _import_names_from_module(
        importer, modname, required, optional)
    feature = Feature(
        input=required_vars['input'],
        transformer=required_vars['transformer'],
        source=modname,
        **optional_vars)
    return feature


def import_contrib_feature_from_collection(importer, modname):
    required = 'features'
    optional = None
    required_vars, _ = _import_names_from_module(
        importer, modname, required, optional)
    features = required_vars['features']
    return features


def get_contrib_features(contrib):

    def onerror(pkgname):
        logging.error(pkgname)

    for importer, modname, _ in pkgutil.walk_packages(
            path=contrib.__path__,
            prefix=contrib.__name__ + '.',
            onerror=onerror
    ):
        logging.debug(
            'Importing contributed feature from module {modname}'
            .format(modname=modname))

        # case 1: file is __init__.py
        if '__init__' in modname:
            logging.debug(
                'Skipping module {modname}: it is an __init__'
                .format(modname=modname))
            continue

        # case 2: file defines `features` variable
        try:
            features = import_contrib_feature_from_collection(
                importer, modname)
            for feature in features:
                yield feature
        except ImportError:

            # case 3: file has at last `input` and `transformer` defined
            try:
                yield import_contrib_feature_from_components(
                    importer, modname)
            except ImportError:
                logging.debug(
                    'Failed to import anything useful from module {modname}'
                    .format(modname=modname))


def get_feature_transformations():
    features = []

    features.append(
        Feature(
            input='ndvi_se',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_sw',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='precipitation_amt_mm',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                SimpleFunctionTransformer(np.log1p)
            ]
        )
    )

    features.append(
        Feature(
            input='precipitation_amt_mm',
            transformer=[
                NullIndicator(),
            ]
        )
    )

    # Same basic cleaning of time series features.
    for input_ in [
        'reanalysis_air_temp_k',
        'reanalysis_avg_temp_k',
        'reanalysis_dew_point_temp_k',
        'reanalysis_max_air_temp_k',
        'reanalysis_min_air_temp_k',
        'reanalysis_precip_amt_kg_per_m2',
        'reanalysis_relative_humidity_percent',
        'reanalysis_specific_humidity_g_per_kg',
        'reanalysis_tdtr_k',

        'station_avg_temp_c',
        'station_diur_temp_rng_c',
        'station_max_temp_c',
        'station_min_temp_c',
        'station_precip_mm',
    ]:
        features.append(
            Feature(
                input=input_,
                transformer=[
                    LagImputer(groupby_kwargs={'level': 'city'}),
                    NullFiller(replacement=0.0),
                    sklearn.preprocessing.StandardScaler(),
                ]
            )
        )

    features.append(
        Feature(
            input='reanalysis_sat_precip_amt_mm',
            transformer=[
                SingleLagger(1, groupby_kwargs={'level': 'city'}),
                LagImputer(groupby_kwargs={'level': 'city'}),
                NullFiller(replacement=0.0),
            ]
        )
    )

    features.append(
        Feature(
            input=['reanalysis_sat_precip_amt_mm',
                   'reanalysis_relative_humidity_percent',
                   'reanalysis_specific_humidity_g_per_kg',
                   'reanalysis_precip_amt_kg_per_m2',
                   'precipitation_amt_mm',
                   'station_precip_mm',
                   ],
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.decomposition.PCA(n_components=2),
            ]
        )
    )

    # add contributed features
    features.extend(get_contrib_features(dengue_prediction.features.contrib))

    return features


def get_target_transformations():
    transformations = []
    transformations.append(
        Feature(
            input='total_cases',
            transformer=IdentityTransformer(),
        )
    )
    return transformations
