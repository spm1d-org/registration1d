'''
Make the package importable from a source checkout without installation
(src layout): tests import registration1d from ../src. PYTHONPATH is set
as well so that worker processes started by the parallel options (which
re-import the package under the 'spawn' start method used on macOS and
Windows) find it too. After `pip install -e .` neither is needed.
'''
import os, sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ['PYTHONPATH'] = SRC + os.pathsep + os.environ.get('PYTHONPATH', '')


def pytest_configure(config):
    # many tests use a deliberately small max_iter on real data; the convergence
    # warning is tested explicitly in test_convergence_warning_on_noise
    config.addinivalue_line('filterwarnings', 'ignore::registration1d.ConvergenceWarning')
