import pathlib
import warnings

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

warnings.filterwarnings(
    'ignore',
    message='Conversion of the second argument of issubdtype',
    module='h5py',
    lineno=36,
)

warnings.filterwarnings(
    'ignore',
    message='Predicted variances smaller than 0.',
    module='sklearn.gaussian_process.gpr',
    lineno=335,
)
