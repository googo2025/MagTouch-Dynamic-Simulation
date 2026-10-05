"""No hardware tests: causality, masks, frozen artifacts and printed meshes."""
from pathlib import Path
import json, struct, unittest, zipfile, xml.etree.ElementTree as ET
import numpy as np
from training.run_forward_r361 import ROOT, sha
from training.forward_complex_r363 import AxisForward
from training.research_formal_r365 import metrics
from live_retrained_reconstruction_r373 import COMPONENTS

OUT=ROOT/'analysis/r374_matched_real_virtual_20261002'
TIP=ROOT/'analysis/r375_printable_indenters_20261002'

class ResearchTests(unittest.TestCase):
    def test_generator_is_causal(self):
        rng=np.random.default_rng(374);q=np.c_[rng.uniform(-4,4,(60,2)),rng.uniform(.4,1.6,60)];t=np.arange(60)*.1
        for p in [ROOT/'analysis/r363_complex_forward_20261001/models/generator.json',ROOT/'analysis/r376_two_tip_forward_20261002/generator.json']:
            m=AxisForward.load(p);old=m.rollout(q,t);changed=q.copy();changed[31:]=[3,-2,1.2]
            np.testing.assert_array_equal(old[:31],m.rollout(changed,t)[:31])
    def test_missing_outputs_are_not_zero_error(self):
        q=np.zeros((10,3));p=q.copy();p[3:]=np.nan
        r=metrics(p,q,np.arange(10)*.1,np.ones(10,bool),np.isfinite(p).all(1))
        self.assertEqual(r['reference_frames'],10);self.assertEqual(r['output_frames'],3);self.assertEqual(r['coverage'],.3)
    def test_new_weights_still_frozen(self):
        for k,v in COMPONENTS.items():self.assertEqual(sha(ROOT/f'analysis/r372_direct_source_plateau_20261001/{k}/best_model.npz'),v['model_sha256'])
    def test_raw_records_unchanged(self):
        rows=json.loads((OUT/'manifest.json').read_text(encoding='utf8'));self.assertEqual(len(rows),18)
        for row in rows:
            for f,h in row['hashes'].items():self.assertEqual(sha(ROOT/row['path']/f),h)
    def test_print_models_round_trip_and_height(self):
        r=json.loads((TIP/'geometry_audit.json').read_text(encoding='utf8'))
        self.assertAlmostEqual(r['original_contact_geometry']['top_flat_diameter_mm'],3,places=4)
        for m in r['models']:
            p=TIP/(m['name']+'.stl');b=p.read_bytes();count=struct.unpack('<I',b[80:84])[0]
            self.assertEqual(len(b),84+50*count);self.assertEqual(count,m['triangles']);self.assertEqual(m['height_mm'],13)
            with zipfile.ZipFile(TIP/(m['name']+'.3mf')) as z:
                root=ET.fromstring(z.read('3D/3dmodel.model'));self.assertEqual(root.get('unit'),'millimeter')
                # Printing software may keep the mesh in production-extension
                # .model parts instead of the package's root assembly .model.
                # Count the actual stored geometry, not only root components.
                v=[];f=[]
                for name in z.namelist():
                    if not name.endswith('.model'):continue
                    part=ET.fromstring(z.read(name))
                    v.extend(e for e in part.iter() if e.tag.rsplit('}',1)[-1]=='vertex')
                    f.extend(e for e in part.iter() if e.tag.rsplit('}',1)[-1]=='triangle')
                self.assertEqual(len(v),m['vertices']);self.assertEqual(len(f),m['triangles'])
                pts=np.array([[float(a.get(k)) for k in ('x','y','z')] for a in v]);self.assertAlmostEqual(np.ptp(pts[:,2]),13)
    def test_known_geometry_registry(self):
        r=json.loads((ROOT/'analysis/r376_two_tip_forward_20261002/two_tip_registry.json').read_text(encoding='utf8'))
        self.assertEqual(set(r['models']),{'final_5mm','old_1mm_cone'})
        for e in r['models'].values():self.assertEqual(sha(ROOT/e['path']),e['sha256'])
    def test_matched_stream_lengths_and_fields(self):
        for p in (OUT/'streams').glob('*_matched.npz'):
            with np.load(p) as f:
                n=len(f['t']);self.assertEqual(f['b'].shape,(n,15));self.assertEqual(f['virtual_b'].shape,(n,15))
                self.assertTrue((np.diff(f['t'])>0).all())
                for arm in ('real_old','real_new','virtual_new'):
                    self.assertEqual(f[arm].shape,(n,3));self.assertFalse(np.isfinite(f[arm][~f[arm+'_ok']]).any())

if __name__=='__main__':unittest.main()
