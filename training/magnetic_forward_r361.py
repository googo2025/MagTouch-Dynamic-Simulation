"""Command-driven magnetic surrogate. No measured field enters rollout.

This is an empirical 5 mm single-indenter model, NOT an identified material law.
Stable exponential feature states model history; outputs are 15 channels in uT.
No hardware, UI, inverse-model or manuscript imports are allowed here.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

SCHEMA = 'MagTouch-CommandForward-R361-v1'
CENTERS = np.stack(np.meshgrid([-6., -3., 0., 3., 6.], [-6., -3., 0., 3., 6.],
                               [.6, 1.4, 2.2, 3.], indexing='ij'), -1).reshape(-1, 3)
WIDTH = np.array([2.5, 2.5, .65])


def spatial_features(q):
    q = np.asarray(q, float).reshape(-1, 3)
    if not np.isfinite(q).all():
        raise ValueError('finite XYZ required')
    z = np.maximum(q[:, 2], 0.)
    x, y, d = q[:, 0]/6., q[:, 1]/6., z/3.
    poly = np.column_stack((np.ones(len(q)), x, y, d, x*x, x*y, y*y, x*d, y*d, d*d))
    distance = (q[:, None, :]-CENTERS[None, :, :])/WIDTH
    rbf = np.exp(-.5*np.sum(distance*distance, axis=2))
    # An unloaded current input has no static delta field. History can persist.
    return np.column_stack((poly, rbf))*(z/(z+.2))[:, None]


def design(q, time_s, taus=(), reset=None):
    q, t = np.asarray(q, float), np.asarray(time_s, float)
    if q.ndim != 2 or q.shape[1] != 3 or t.ndim != 1 or len(q) != len(t) or len(t) == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('strictly increasing timestamps with matching XYZ required')
    known = np.isfinite(q).all(1)
    phi = spatial_features(np.where(known[:, None], q, 0.))
    reset = np.zeros(len(t), bool) if reset is None else np.asarray(reset, bool)
    history = []
    for tau in taus:
        if tau <= 0:
            raise ValueError('time constant must be positive')
        state = np.zeros(phi.shape[1]); part = np.zeros_like(phi)
        for i in range(len(t)):
            if reset[i] or not known[i] or (i and t[i]-t[i-1] > .5):
                state[:] = 0.
            dt = t[i]-t[i-1] if i else 0.
            rho = np.exp(-dt/tau)
            state = rho*state+(1-rho)*phi[i]
            part[i] = phi[i]-state
        history.append(part)
    return np.column_stack([phi]+history)


def fit_readout(x, y, weights, alpha):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.all(w > 0)):
        raise ValueError('invalid fitting data')
    w = w/w.sum()
    scale = np.maximum(np.sqrt(w @ (x*x)), .02)
    a = x/scale
    lhs = a.T @ (a*w[:, None])
    rhs = a.T @ (y*w[:, None])
    coef = np.linalg.solve(lhs+alpha*np.eye(x.shape[1]), rhs)/scale[:, None]
    return coef


class MagneticForward:
    def __init__(self, coef, taus=(), metadata=None):
        self.coef = np.asarray(coef, float)
        self.taus = tuple(float(v) for v in taus)
        self.metadata = metadata or {}
        expected = (len(CENTERS)+10)*(1+len(self.taus))
        if self.coef.shape != (expected, 15) or not np.isfinite(self.coef).all():
            raise ValueError('invalid coefficient dimensions')

    def rollout(self, q, time_s, reset=None):
        return design(q, time_s, self.taus, reset) @ self.coef

    def static(self, q):
        return spatial_features(q) @ self.coef[:len(CENTERS)+10]

    def support(self, q):
        q = np.asarray(q, float)
        # Screening envelope, not a calibrated uncertainty probability.
        bounds = self.metadata.get('training_bounds_mm', [[-6, -6, .4], [6, 6, 3.]])
        return np.isfinite(q).all(1) & (q >= np.array(bounds[0])-1e-6).all(1) & (q <= np.array(bounds[1])+1e-6).all(1)

    def to_dict(self):
        return {'schema': SCHEMA, 'centers_mm': CENTERS.tolist(), 'width_mm': WIDTH.tolist(),
                'taus_s': list(self.taus), 'coef': self.coef.tolist(), 'metadata': self.metadata,
                'input': 'command XYZ and preceding command history only',
                'output': 'delta_B_uT; S1x,S1y,S1z,...,S5x,S5y,S5z',
                'physics_validated': False, 'contact_count': 1}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False), encoding='utf8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf8'))
        if d.get('schema') != SCHEMA or not np.allclose(d['centers_mm'], CENTERS):
            raise ValueError('unsupported model schema')
        return cls(d['coef'], d['taus_s'], d['metadata'])


def accept_physics_export(payload, *, require_validated=True, validation_artifact=None):
    """Read-only bridge for future mechanical/magnetic solver exports.

    Multi-contact data are quarantined unless an independent validation artifact
    is supplied. Single-contact superposition is never accepted as ground truth.
    """
    for k in ('schema', 't_s', 'contacts_xyz_mm', 'delta_b_uT', 'provenance'):
        if k not in payload:
            raise ValueError('missing physics-export field: '+k)
    if payload['schema'] != 'MagTouch-Physics-Forward-1':
        raise ValueError('unknown physics-export schema')
    t = np.asarray(payload['t_s'], float)
    q = np.asarray(payload['contacts_xyz_mm'], float)
    b = np.asarray(payload['delta_b_uT'], float)
    p = payload['provenance']
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('invalid physics-export timestamps')
    if q.ndim != 3 or q.shape[0] != len(t) or q.shape[1] < 1 or q.shape[2] != 3 or b.shape != (len(t), 15):
        raise ValueError('invalid physics-export dimensions')
    if not (np.isfinite(t).all() and np.isfinite(q).all() and np.isfinite(b).all()):
        raise ValueError('nonfinite physics export')
    if p.get('construction') == 'sum_of_single_contacts':
        raise ValueError('single-contact superposition is not validated multi-contact truth')
    if require_validated:
        required=('solver_version','geometry_contract_sha256','material_parameters',
                  'initial_state','contact_law','magnetic_model')
        if any(k not in p for k in required):
            raise ValueError('complete solver, geometry, material and contact provenance required')
        if not p.get('validation_passed') or validation_artifact is None:
            raise ValueError('independent physical validation artifact required before training use')
        artifact=Path(validation_artifact).read_bytes()
        if hashlib.sha256(artifact).hexdigest()!=p.get('validation_sha256'):
            raise ValueError('physical validation artifact hash mismatch')
        report=json.loads(artifact)
        if (report.get('validation_status')!='passed' or
            report.get('reference_kind')!='independent_real_measurement' or
            report.get('contact_count')!=q.shape[1] or
            not report.get('geometry_contract_sha256') or
            report.get('geometry_contract_sha256')!=p.get('geometry_contract_sha256')):
            raise ValueError('physical validation scope does not match the export')
    return t, q, b
