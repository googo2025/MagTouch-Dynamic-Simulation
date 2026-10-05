import unittest
import numpy as np
from training.active_candidates_r381 import rank,nearest_position

class ActiveTests(unittest.TestCase):
    def test_diversity_and_source_envelope(self):
        rows=[{'id':str(i),'style':'a' if i<3 else 'b','qualified_source_envelope':i!=5,
            'field_disagreement_p95_noise_units':i,'source_position_novelty_p95':i,'history_fraction_p95':i} for i in range(6)]
        ranked,chosen=rank(rows)
        self.assertEqual(ranked[0]['id'],'5');self.assertNotIn('5',[r['id'] for r in chosen])
        self.assertEqual(sum(r['style']=='a' for r in chosen),2)
        self.assertEqual(sum(r['style']=='b' for r in chosen),2)

    def test_position_novelty_mm(self):
        r=nearest_position(np.array([[0,0,1],[3,4,1.]]),np.array([[0,0,1.]]))
        np.testing.assert_array_equal(r,[0,5])

if __name__=='__main__':unittest.main()
