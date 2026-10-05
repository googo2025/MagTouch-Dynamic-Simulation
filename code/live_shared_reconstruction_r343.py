"""A single causal magnetic history for action and localization evidence.

The two heads remain frozen. Only identical history extraction is shared;
the position decoders intentionally retain their distinct state lifecycles.
No hardware access, retraining, new permission or coordinate smoothing.
"""
import numpy as np

from training.unified_response_runtime_r330 import MagneticFeatureHistory
from live_unified_reconstruction_r342 import UnifiedEvidenceReconstructionEngine


class SharedMagneticFeatureHistory:
    """Once per explicit frame, never implicitly reuse a repeated timestamp."""
    def __init__(self):
        self.history = MagneticFeatureHistory()
        self.extractions = 0
        self.cache_hits = 0
        self.reset()

    def reset(self):
        self.history.reset()
        self.begin_frame()

    def begin_frame(self):
        self.key = self.field = self.current = None

    def push(self, time_ns, field, eligible=True):
        b = np.asarray(field, float)
        key = (int(time_ns), bool(eligible))
        if self.current is not None:
            if key != self.key or not np.array_equal(b, self.field):
                raise ValueError('Shared frame must have identical time, field and eligibility')
            self.cache_hits += 1
            return self.current
        current = self.history.push(time_ns, b, eligible)
        # The two consumers read the same immutable packet. Neither can modify
        # a feature that the other head will see later in the current frame.
        for value in current.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        self.key, self.field, self.current = key, b.copy(), current
        self.extractions += 1
        return current


class SharedCausalReconstructionEngine(UnifiedEvidenceReconstructionEngine):
    """R342 outputs, with duplicate history extraction removed exactly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_history = SharedMagneticFeatureHistory()
        self.response_observer.history = self.shared_history
        self.permission.history = self.shared_history
        self.reset_unified()

    def push(self, t, b, *, eligible=True):
        self.shared_history.begin_frame()
        result = super().push(t, b, eligible=eligible)
        result['family'] = 'SharedCausalReconstructionR343'
        result['extra'].update(shared_causal_feature_history=True,
            shared_history_extractions=self.shared_history.extractions,
            shared_history_cache_hits=self.shared_history.cache_hits,
            neural_heads_jointly_retrained=False)
        return result
