"""Four observed Hall nodes -> withheld fifth-node magnetic response.

No commanded XYZ or missing-node observations enter a prediction. Learned
completion is trained using SOURCE labels at known nodes, not arbitrary new
spatial locations. A source-free affine harmonic field is an unfitted baseline.
"""
from __future__ import annotations
import json
import numpy as np
from training.run_forward_r361 import ROOT,inventory,load_row,dump,sha
from training.magnetic_forward_r361 import fit_readout

OUT=ROOT/'analysis/r364_sparse_field_20261001'


def harmonic_basis(points):
    r=np.asarray(points,float)/8.
    gs=np.array([[[1,0,0],[0,-1,0],[0,0,0]],[[1,0,0],[0,1,0],[0,0,-2]],
       [[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],[[0,0,0],[0,0,1],[0,1,0]]],float)
    # B=a+G r, G symmetric and trace-free: div B = curl B = 0 exactly.
    return np.concatenate((np.broadcast_to(np.eye(3),(len(r),3,3)),np.einsum('kij,nj->nik',gs,r)),axis=2)


def completion_features(b,t,reset,observed,causal=True,nonlinear=True):
    bb=np.asarray(b,float)[:,observed];known=np.isfinite(bb).all(1)
    v=np.where(known[:,None],bb/300.,0.);parts=[v]
    if causal:
        for tau in (.3,1.):
            state=np.zeros(12);mem=np.zeros_like(v)
            for i in range(len(v)):
                if reset[i] or not known[i] or (i and t[i]-t[i-1]>.5):state[:]=0
                rho=np.exp(-(t[i]-t[i-1])/tau) if i else 1.
                state=rho*state+(1-rho)*v[i];mem[i]=v[i]-state
            parts.append(mem)
    x=np.column_stack(parts)
    if nonlinear:
        rng=np.random.default_rng(364)
        weights=rng.normal(0,.45,(x.shape[1],128));bias=rng.normal(0,.3,128)
        return np.column_stack((np.ones(len(v)),x,np.tanh(x@weights+bias)))
    return np.column_stack((np.ones(len(v)),x))


def score(pred,truth,mask):
    m=mask&np.isfinite(pred).all(1)&np.isfinite(truth).all(1)
    if not m.any():return {'frames':0}
    e=pred[m]-truth[m];rmse=float(np.sqrt(np.mean(e**2)))
    return {'frames':int(m.sum()),'rmse_uT':rmse,'nrmse':rmse/max(float(np.sqrt(np.mean(truth[m]**2))),1e-6),
       'vector_rmse_uT':float(np.sqrt(np.mean(np.sum(e**2,1)))),
       'constant_field_bias_uT':e.mean(0).tolist(),
       'constant_field_bias_removed_rmse_uT':float(np.sqrt(np.mean((e-e.mean(0))**2)))}


def main():
    for p in (OUT,OUT/'models',OUT/'predictions'):p.mkdir(parents=True,exist_ok=True)
    inv=inventory();train=[load_row(r) for r in inv if r['role']=='train'];valid=[load_row(r) for r in inv if r['role']=='validation']
    geom=json.loads((ROOT/'code/config/hardware_geometry_v1.json').read_text(encoding='utf8'))
    xyz=np.array([[geom['sensors'][f'S{s}'][f'{a}_mm'] for a in 'xyz'] for s in range(1,6)])
    h=harmonic_basis(xyz).reshape(15,8);models={};selection=[];ranks=[]
    methods={'ridge_current':(False,False),'nonlinear_current':(False,True),'nonlinear_causal':(True,True)}
    full=np.vstack([a['b'][a['primary']] for a in train]);weights=np.concatenate([np.full(a['primary'].sum(),1/a['primary'].sum()) for a in train]);weights/=weights.sum()
    mean=weights@full;cov=(full-mean).T@((full-mean)*weights[:,None]);ev,u=np.linalg.eigh(cov);order=np.argsort(ev)[::-1];ev=ev[order];u=u[:,order]
    np.savez_compressed(OUT/'models/source_field_modes.npz',mean=mean,modes=u,eigenvalues=ev)
    selected_methods={}
    for node in range(5):
        hidden=np.arange(node*3,node*3+3);observed=np.setdiff1d(np.arange(15),hidden)
        ranks.append({'held_node':node+1,'rank':int(np.linalg.matrix_rank(h[observed])),
             'singular_values':np.linalg.svd(h[observed],compute_uv=False).tolist()})
        models[(node,'harmonic')]=h[hidden]@np.linalg.pinv(h[observed])
        lowrank_best=None
        for k in (3,6,9,12):
            uk=u[:,:k];uo=uk[observed]
            for alpha in (.0001,.01,.1):
                mapping=uk[hidden]@np.linalg.solve(uo.T@uo+alpha*np.eye(k),uo.T)
                intercept=mean[hidden]-mapping@mean[observed]
                val=[score(a['b'][:,observed]@mapping.T+intercept,a['b'][:,hidden],a['primary']) for a in valid]
                objective=float(np.mean([r['nrmse'] for r in val]))
                if lowrank_best is None or objective<lowrank_best[0]:lowrank_best=(objective,k,alpha,mapping,intercept)
        obj,k,alpha,mapping,intercept=lowrank_best
        models[(node,'lowrank')]=(mapping,intercept)
        np.savez_compressed(OUT/'models'/f'S{node+1}_lowrank.npz',mapping=mapping,intercept=intercept)
        selection.append({'held_node':node+1,'method':'lowrank','modes':k,'alpha':alpha,'source_validation_nrmse':obj})
        for name,(causal,nonlinear) in methods.items():
            xs=[];ys=[];ws=[]
            for a in train:
                mask=a['primary']&np.isfinite(a['b']).all(1)
                xs.append(completion_features(a['b'],a['t'],a['reset'],observed,causal,nonlinear)[mask])
                ys.append(a['b'][mask][:,hidden]);ws.append(np.full(mask.sum(),1/mask.sum()))
            x,y,w=np.vstack(xs),np.vstack(ys),np.concatenate(ws)
            xv=[completion_features(a['b'],a['t'],a['reset'],observed,causal,nonlinear) for a in valid]
            best=None
            for alpha in (.0001,.001,.01,.1):
                coef=fit_readout(x,y,w,alpha)
                scores=[score(v@coef,a['b'][:,hidden],a['primary']) for v,a in zip(xv,valid)]
                objective=float(np.mean([r['nrmse'] for r in scores]))
                if best is None or objective<best[0]:best=(objective,alpha,coef)
            models[(node,name)]=best[2]
            np.savez_compressed(OUT/'models'/f'S{node+1}_{name}.npz',coef=best[2],observed_channels=observed)
            selection.append({'held_node':node+1,'method':name,'alpha':best[1],'source_validation_nrmse':best[0]})
            print('completion',node+1,name,round(best[0],4),flush=True)
        selected_methods[node]=min([r for r in selection if r['held_node']==node+1],key=lambda r:r['source_validation_nrmse'])['method']
    dump(OUT/'selection.json',{'rows':selection,'harmonic_ranks':ranks,'source_selected_methods':selected_methods,
       'source_spatial_mode_cumulative_variance':(np.cumsum(ev)/ev.sum()).tolist(),
       'target_selection':False,'waypoint_selection':False,'geometry_sha256':sha(ROOT/'code/config/hardware_geometry_v1.json'),
       'train_ids':[a['id'] for a in train],'validation_ids':[a['id'] for a in valid],
       'input':'four observed nodes and their causal magnetic history, no command or fifth node',
       'boundary':'node locations are known and SOURCE missing-node labels are used in training; not uncalibrated spatial extrapolation'})
    dump(OUT/'freeze.json',{'model_hashes':{p.name:sha(p) for p in sorted((OUT/'models').iterdir())},'test_opened_after_freeze':True})
    tests=[load_row(r) for r in inv if r['role']=='test'];results=[]
    for a in tests:
        predictions={}
        for node in range(5):
            hidden=np.arange(node*3,node*3+3);observed=np.setdiff1d(np.arange(15),hidden)
            item={'id':a['id'],'device':a['device'],'mode':a['mode'],'held_node':node+1}
            for method in ('harmonic','lowrank',*methods):
                if method=='harmonic':pred=a['b'][:,observed]@models[(node,method)].T
                elif method=='lowrank':
                    mapping,intercept=models[(node,method)];pred=a['b'][:,observed]@mapping.T+intercept
                else:
                    causal,nonlinear=methods[method]
                    pred=completion_features(a['b'],a['t'],a['reset'],observed,causal,nonlinear)@models[(node,method)]
                item[method]=score(pred,a['b'][:,hidden],a['primary'])
                predictions[f'S{node+1}_{method}']=pred
            item['source_selected']=item[selected_methods[node]]
            predictions[f'S{node+1}_source_selected']=predictions[f'S{node+1}_{selected_methods[node]}']
            results.append(item)
        np.savez_compressed(OUT/'predictions'/(a['id']+'.npz'),t=a['t'],b=a['b'],q=a['q'],primary=a['primary'],**predictions)
    macro={}
    for device in ('source','target'):
        ss=[r for r in results if r['device']==device]
        macro[device]={method:{k:float(np.mean([r[method][k] for r in ss if r[method]['frames']])) for k in ('rmse_uT','nrmse','constant_field_bias_removed_rmse_uT')} for method in ('harmonic','lowrank',*methods,'source_selected')}
    node_macro={str(n+1):{device:{m:float(np.mean([r[m]['nrmse'] for r in results if r['device']==device and r['held_node']==n+1])) for m in ('harmonic','lowrank',*methods,'source_selected')} for device in ('source','target')} for n in range(5)}
    dump(OUT/'report.json',{'test':results,'macro':macro,'by_node':node_macro,'test_trial_count':len(tests),
       'node_trial_pairs':len(results),'target_zero_shot':True,'whole_xyz_field_reconstructed':False,
       'multi_touch_demonstrated':False,'primary_scores_unregistered':True,'test_previously_inspected':True,
       'physical_baseline':'8-coefficient affine curl-free and divergence-free field in a source-free region only',
       'spatial_validation_boundary':'withheld input at known sensor locations; no measurements at new spatial coordinates/heights',
       'sources':['https://web.mit.edu/6.013_book/www/chapter8/8.6.html','https://arxiv.org/abs/1309.5637']})
    print(json.dumps(macro,indent=2),flush=True)


if __name__=='__main__':main()
