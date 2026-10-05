"""Re-establish the PAPER'S core modules from virtual supervision only.

Not model replacement or virtual-to-real augmentation. The generator is an
empirical surrogate fitted previously to measurements; this is consequently
conditional mechanism reproduction, NOT independent first-principles FEM.
"""
from pathlib import Path
import json,time
import numpy as np
from training.run_forward_r361 import ROOT,dump,sha
from training.forward_complex_r363 import AxisForward
from training.research_standard_r365 import standard_actions
from training.probe_had239_local_forward_r104 import ForwardAtlas
from training.probe_had239_field_memory_r122 import observed_memory,feature_to_field
from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.probe_causal_xy_neural_r164 import MLP
from training.research_formal_r365 import metrics

OUT=ROOT/'analysis/r367_simulation_mechanism_20261001'
GEN=ROOT/'analysis/r363_complex_forward_20261001'


def build_atlas(generator,sigma):
    axis=np.linspace(-6,6,11);z=np.arange(.4,3.0001,.2)
    q=np.stack(np.meshgrid(axis,axis,z,indexing='ij'),-1).reshape(-1,3)
    # Virtual static calibration, ten independent samples averaged per point.
    b=generator.static(q)+np.random.default_rng(3671).normal(size=(len(q),15))*sigma/np.sqrt(10)
    np.savez_compressed(OUT/'virtual_static_calibration.npz',q=q,b=b)
    return ForwardAtlas(q,b,'unit_logamp'),q,b


def make_trials(generator,sigma):
    train=[];valid=[];test=[];rng=np.random.default_rng(3672)
    for base in standard_actions():
        for split,scale,rotation,speed in [('train',.75,.12,.85),('train',1.05,.48,.95),
                                            ('validation',.9,.31,.9),('test',1.,0.,1.)]:
            q=base['q'].copy();a=q[:,:2].copy();c,s=np.cos(rotation),np.sin(rotation)
            q[:,0]=scale*(a[:,0]*c-a[:,1]*s);q[:,1]=scale*(a[:,0]*s+a[:,1]*c)
            q[:,2]=np.where(q[:,2]>0,q[:,2]*(.85 if scale==.75 else 1.),q[:,2])
            t=base['t']*max(1,scale)/speed
            equilibrium=generator.static(q);history=generator.rollout(q,t)-equilibrium
            noise=rng.normal(size=equilibrium.shape)*sigma
            target=train if split=='train' else valid if split=='validation' else test
            idx=len(target)
            row={'id':split+'_'+base['family']+'_'+str(idx),'role':split,'family':base['family'],
                'q':q,'t':t,'b':equilibrium+history+noise,'off':equilibrium+noise,
                'history_field':history,'noise':noise}
            target.append(row)
    # Independent stationary virtual points; never used for weight training.
    q=np.random.default_rng(3673).uniform([-4,-4,.6],[4,4,2.4],(180,3));t=np.arange(len(q))*.1
    static={'id':'test_stationary_points','role':'test','family':'stationary_points','q':q,'t':t,
        'b':generator.static(q)+rng.normal(size=(len(q),15))*sigma,'independent_frames':True}
    # Validate and score all frames within the declared calibration domain.
    for a in train+valid+test+[static]:
        a['gate']=np.linalg.norm(a['b'],axis=1)>=70.
        a['primary']=(a['q'][:,2]>=.4)&(a['q'][:,2]<=3.)
    return train,valid,test,static


def prepare(a,atlas,basis=None):
    gate=a['gate'];b=a['b'];q0=np.full((len(b),3),np.nan)
    q0[gate]=atlas.predict(b[gate]);phi=atlas.features(b)
    a.update(base=q0,phi=phi)
    if basis is not None:
        observed=np.zeros((len(b),2))
        observed[gate]=observed_memory(atlas,basis,np.column_stack((q0[gate],phi[gate])),q0[gate])
        a['h_obs']=observed


def covariance_basis(train,atlas):
    cov=np.zeros((16,16))
    for a in train:
        mask=a['primary'];d=a['phi'][mask]-atlas.features(atlas.forward(a['q'][mask]))
        cov+=d.T@d/len(d)
    cov/=len(train);ev,u=np.linalg.eigh(cov);order=np.argsort(ev)[::-1]
    return u[:,order[:2]],float(ev[order[:2]].sum()/ev.sum())


def memory(a,atlas,basis,retention,strength):
    n=len(a['b']);h=np.zeros((n,2));previous=np.zeros(2)
    for i in range(n):
        if not a['gate'][i]:previous[:]=0;continue
        if i and a['t'][i]-a['t'][i-1]>.25:previous[:]=0
        if a.get('independent_frames'):previous[:]=0
        rho=retention**((a['t'][i]-a['t'][i-1])/.1) if i else retention
        previous=rho*previous+(1-rho)*a['h_obs'][i];h[i]=previous
    corrected_phi=a['phi']-strength*h@basis.T
    field=feature_to_field(atlas,corrected_phi);p=np.full_like(a['base'],np.nan)
    p[a['gate']]=atlas.predict(field[a['gate']])
    # Paper's original Z safeguard: XY memory correction does not move Z.
    p[a['gate'],2]=a['base'][a['gate'],2]
    return p,h,field


def guard_and_features(a,atlas,basis,retention,strength):
    p,h,field=memory(a,atlas,basis,retention,strength)
    branch=EvidenceGuardedBranchInverse(atlas,spatial_penalty=.03,innovation_gain=2.,
        activation_distance_mm=1.,max_local_cost_ratio=4.)
    hist=CausalXYFeatureHistory();g=np.full_like(p,np.nan);x=np.full((len(p),53),np.nan)
    veto=local=0
    for i in range(len(p)):
        if not a['gate'][i]:branch.reset();hist.reset();continue
        if a.get('independent_frames'):branch.reset();hist.reset()
        now=int(a['t'][i]*1e9)+10_000_000_000
        r=branch.predict_frame(field[i],time_ns=now,b0_id=0,stream_id=0,global_xyz=p[i])
        g[i]=r['xyz_mm'];g[i,2]=a['base'][i,2]
        veto+=bool(branch.last_veto);local+=bool(r['used_local_branch'])
        phi=atlas.features(field[i:i+1])[0]
        x[i]=hist.push(time_ns=now,b0_id=0,stream_id=0,xyz=g[i],phi=phi,memory=h[i])
    a.update(memory=p,guarded=g,x=x,h=h,corrected=field,veto_count=veto,local_count=local)


def evaluate(a,model=None,instant=False):
    p=a['guarded'].copy()
    if model is not None:
        x=a['x'].copy()
        if instant:x[:,19:]=0.
        p[a['gate'],:2]+=model.predict(x[a['gate']])
    return p


def mlp_data(rows,instant=False):
    data=[]
    for i,a in enumerate(rows):
        x=a['x'].copy()
        if instant:x[:,19:]=0.
        data.append({'path':Path(f'Trial_{i:06d}_virtual'),'mode':a['family'],
            'core':a['primary']&a['gate'],'truth':a['q'],'guarded':a['guarded'],
            'time':(a['t']*1e9).astype(np.int64),'_xy_features':x})
    return data


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists():raise FileExistsError('preserve finished virtual discovery run')
    started=time.perf_counter();generator=AxisForward.load(GEN/'models/generator.json')
    sigma=np.array(json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))['noise_sigma_uT'])
    dump(OUT/'declared.json',{'purpose':'Reproduce scientific phenomena and paper-module ablations, NOT replace the formal model',
        'inverse_static_calibration':'121 virtual XY x14 depths; +/-6mm, .4-3mm; ten-noise-sample mean',
        'inverse_training':'12 complete virtual STANDARD motions + virtual static anchors',
        'inverse_validation':'6 complete virtual STANDARD motions, generator frozen beforehand',
        'inverse_test':'6 virtual standard motions with new noise; independent virtual static points; 60 complex virtual paths',
        'real_inverse_weights_loaded':False,'real_inverse_training_rows':0,'generator_is_empirical_not_FEM':True,
        'generator_sha256':sha(GEN/'models/generator.json'),'static_memory_worlds':'same path/noise; history coefficient lambda=0 vs1',
        'fixed_gate':'70uT norm, common to inverse ablations; not a trained contact observer or actual UI gate',
        'z_scope':'original static-Z safeguard only; final R348 depth-network discovery is not reproduced by this core study',
        'selection_grid':{'retention':[.5,.8,.95],'strength':[.25,.5,1.]},'network':'53-64-64-2 tanh, same MSE+0.001 weight decay/Adam as paper',
        'no_hardware_or_paper_change':True})
    atlas,static_q,static_b=build_atlas(generator,sigma)
    train,valid,test,stationary=make_trials(generator,sigma)
    for a in train+valid+test+[stationary]:prepare(a,atlas)
    basis,explained=covariance_basis(train,atlas)
    np.savez_compressed(OUT/'virtual_memory_basis.npz',basis=basis)
    for a in train+valid+test+[stationary]:prepare(a,atlas,basis)
    choices=[]
    for retention in (.5,.8,.95):
        for strength in (.25,.5,1.):
            loss=[]
            for a in valid:
                p,_,_=memory(a,atlas,basis,retention,strength)
                mask=a['primary']&a['gate'];loss.append(np.linalg.norm(p[mask,:2]-a['q'][mask,:2],axis=1).mean())
            choices.append({'retention':retention,'strength':strength,'validation_xy_mm':float(np.mean(loss))})
    pick=min(choices,key=lambda a:a['validation_xy_mm']);dump(OUT/'memory_selection.json',{'choices':choices,'selected':pick,'rank2_variance':explained})
    print('virtual memory selected',pick,'variance',explained,flush=True)
    for a in train+valid+test+[stationary]:
        guard_and_features(a,atlas,basis,pick['retention'],pick['strength'])
        print('virtual core decoded',a['id'],flush=True)
    arrays=[];targets=[]
    for a in train:
        choose=np.flatnonzero(a['primary']&a['gate'])[::2]
        arrays.append(a['x'][choose]);targets.append(np.clip(a['q'][choose,:2]-a['guarded'][choose,:2],-3,3))
    x=np.vstack(arrays);y=np.vstack(targets)
    anchors=np.column_stack((static_q[::3]/[10,10,3],atlas.features(static_b[::3]),np.zeros((len(static_q[::3]),34))))
    models={}
    for name,anchor,instant in [('causal_no_static_anchors',False,False),('causal_static_anchors',True,False),('current_only_static_anchors',True,True)]:
        xx=np.vstack((x,anchors)) if anchor else x.copy();yy=np.vstack((y,np.zeros((len(anchors),2)))) if anchor else y.copy()
        if instant:xx[:,19:]=0.
        model=MLP(64,seed=7).fit(xx,yy,mlp_data(valid,instant),atlas)
        models[name]=model
        np.savez_compressed(OUT/(name+'.npz'),**{k:getattr(model,k) for k in ('mean','scale','w1','b1','w2','b2','w3','b3')})
        dump(OUT/(name+'.json'),{'epoch':model.best_epoch,'objective':model.best_validation,'history':model.history,'train_samples':len(xx),'virtual_only':True})
        print('virtual MLP fitted',name,model.best_epoch,flush=True)
    rows=[]
    def score_trial(a,world):
        score={'F':a['base'],'M':a['memory'],'G':a['guarded'],
            **{key:evaluate(a,model,key.startswith('current_only')) for key,model in models.items()}}
        result={'id':a['id'],'family':a['family'],'world':world,'veto_count':a['veto_count'],'local_count':a['local_count'],
            'scores':{key:metrics(p,a['q'],a['t'],a['primary'],a['gate']) for key,p in score.items()}}
        dest=OUT/'predictions';dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/(a['id']+'_'+world+'.npz'),q=a['q'],t=a['t'],gate=a['gate'],**score)
        return result
    for a in test+[stationary]:
        rows.append(score_trial(a,'history_on'))
        if 'off' in a:
            off={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in a.items() if k in ('id','family','q','t','primary')}
            off['b']=a['off'];off['gate']=np.linalg.norm(off['b'],axis=1)>=70.
            prepare(off,atlas,basis);guard_and_features(off,atlas,basis,pick['retention'],pick['strength'])
            rows.append(score_trial(off,'history_off'))
    data=np.load(GEN/'complex_synthetic.npz');manifest=json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))
    for i,info in enumerate(manifest['trials']):
        lo,hi=data['offsets'][i:i+2];q,t,b=data['q'][lo:hi],data['t'][lo:hi],data['b'][lo:hi]
        a={'id':info['id'],'family':info['family'],'q':q,'t':t,'b':b,'primary':(q[:,2]>=.4)&(q[:,2]<=3.),'gate':np.linalg.norm(b,axis=1)>=70.}
        prepare(a,atlas,basis);guard_and_features(a,atlas,basis,pick['retention'],pick['strength']);rows.append(score_trial(a,'complex_heldout'))
        if i%10==9:print('simulation complex tests',i+1,flush=True)
    dump(OUT/'report.json',{'rows':rows,'rank2_variance':explained,'memory_selection':pick,
        'interpretation':'Simulation-only inverse modules; known empirical generator, not independent material physics. Tests measure mechanism reproduction, not promotion or real-device superiority.',
        'virtual_inverse_training_complete':True,'formal_model_modified':False,'real_weights_loaded':False,
        'elapsed_s':time.perf_counter()-started})
    print('simulation-only core discovery completed',flush=True)


if __name__=='__main__':main()
