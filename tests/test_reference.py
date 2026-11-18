'''
Agreement with reference implementations.

These tests compare registration1d with independent implementations of the
same algorithms. The reference packages are not dependencies of
registration1d; each test is skipped when its package is not installed
(pip install "registration1d[reference]").

- dtw-python  : the DTW path and distance for the symmetric step patterns and
                the Sakoe-Chiba band are identical (same recursion, same
                local cost |x - y|); for the asymmetric pattern the distance
                is identical and the path may differ only where several paths
                tie.
- fdasrsf     : group SRSF alignment of the Dorn2012 data (lam=0, five
                iterations) gives warps within 0.03 of the domain of
                fdawarp.srsf_align; the dynamic programming grid and the
                numerical derivative are independent implementations, so the
                agreement is close but not exact.
- scikit-fda  : least-squares shift registration gives the same shifts as
                LeastSquaresShiftRegistration once the reference's shifts are
                centered (registration1d removes the mean shift).
'''

import os
import numpy as np
import pytest
import registration1d as reg1d
from registration1d import dtw as rdtw


Q    = 101
t    = np.linspace(0, 1, Q)
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'data')


def _pair():
    rng = np.random.default_rng(0)
    x   = np.exp(-((t - 0.4) / 0.10)**2) + 0.05 * rng.standard_normal(Q)
    y   = np.exp(-((t - 0.5) / 0.12)**2) + 0.05 * rng.standard_normal(Q)
    return x, y


def _dorn():
    with np.load(os.path.join(DATA, 'Dorn2012-reduced.npz'), allow_pickle=True) as z:
        return reg1d.register_linear(list(z['y']), n=Q).y


# ------------------------------------------------------------------ dtw-python

def test_dtw_path_matches_dtw_python():
    dtw   = pytest.importorskip('dtw')
    x, y  = _pair()
    # dtw.dtw(query, reference); registration1d.dtw_path(reference, query) with |x - y| local cost
    for name, pattern in (('symmetric1', dtw.symmetric1), ('symmetric2', dtw.symmetric2)):
        path, dist = rdtw.dtw_path(x, y, step_pattern=name, p=1)
        ref        = dtw.dtw(y, x, step_pattern=pattern)
        assert np.array_equal(np.asarray(path), np.c_[ref.index2, ref.index1]), name
        assert np.isclose(dist, ref.distance)
    # asymmetric: one step per node of the first argument in both implementations
    path, dist = rdtw.dtw_path(x, y, step_pattern='asymmetric', p=1)
    ref        = dtw.dtw(x, y, step_pattern=dtw.asymmetric)
    assert np.isclose(dist, ref.distance)               # the path may differ where paths tie


def test_dtw_band_matches_dtw_python():
    dtw   = pytest.importorskip('dtw')
    x, y  = _pair()
    for nodes in (5, 10, 20):
        path, dist = rdtw.dtw_path(x, y, step_pattern='symmetric2', window=nodes / (Q - 1), p=1)
        ref        = dtw.dtw(y, x, window_type='sakoechiba', window_args=dict(window_size=nodes))
        assert np.array_equal(np.asarray(path), np.c_[ref.index2, ref.index1]), nodes
        assert np.isclose(dist, ref.distance)


# ------------------------------------------------------------------ fdasrsf

def test_srsf_group_matches_fdasrsf():
    fdasrsf = pytest.importorskip('fdasrsf')
    import io, contextlib
    yi = _dorn()
    with contextlib.redirect_stdout(io.StringIO()):
        fw = fdasrsf.fdawarp(yi.T, t)
        fw.srsf_align(MaxItr=5)
    r  = reg1d.register_srsf(yi, lam=0, max_iter=5)
    d  = np.abs(r.warps.asarray() - fw.gam.T).max(axis=1)
    assert d.max() < 0.03 and np.median(d) < 0.01


# ------------------------------------------------------------------ scikit-fda

def test_shift_matches_scikit_fda():
    skfda = pytest.importorskip('skfda')
    from skfda.preprocessing.registration import LeastSquaresShiftRegistration
    shifts = np.array([-0.05, 0.02, 0.04, -0.01, 0.0, 0.03])
    y      = np.array([np.exp(-((t - 0.5 - s) / 0.12)**2)  for s in shifts])
    fd     = skfda.FDataGrid(y, grid_points=t)
    reg    = LeastSquaresShiftRegistration(max_iter=50, tol=1e-8, extrapolation='bounds')
    reg.fit(fd)
    reg.transform(fd)                                  # sets deltas_
    d_ref  = reg.deltas_ - reg.deltas_.mean()
    d      = reg1d.register_shift(y, max_iter=50, tol=1e-8, fill_value='edge').info['shift']
    assert np.abs(d - d_ref).max() < 1e-4
