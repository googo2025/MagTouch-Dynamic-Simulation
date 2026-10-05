"""Our depth architecture re-established with virtual standard motions only.

Current-field depth profile -> 56D causal neural proposal -> magnetic/support
adjudication -> 59D bounded fast/slow increment with centered per-TRAIN-Trial
targets and increment-difference loss. Formal learned artifacts are not loaded.
"""
import argparse, json, time
from dataclasses import asdict
import numpy as np
from training.simulation_discovery_r367 import (OUT as CORE,GEN,AxisForward,ForwardAtlas,
    make_trials,prepare,guard_and_features,evaluate,MLP,metrics,dump,sha)
from training.virtual_experiment_lab_r368 import OUT,load_virtual_trial
from training.z_evidence_runtime_r334 import DepthPolicy,depth_profiles,MagneticDepthTracker
from training.unified_depth_runtime_r341 import UnifiedDepthPolicy,UnifiedDepthAdjudicator,SourceFeatureSupport
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.fast_depth_features_r344 import FastDepthContext
from training.causal_depth_network_r337 import DepthMLP

DEPTH=OUT/'depth_study'


def core_context():
    a=np.load(CORE/'virtual_static_calibration.npz');atlas=ForwardAtlas(a['q'],a['b'],'unit_logamp')
    basis=np.load(CORE/'virtual_memory_basis.npz')['basis']
    pick=json.loads((CORE/'memory_selection.json').read_text(encoding='utf8'))['selected']
    model=MLP(64,7)
    with np.load(CORE/'causal_static_anchors.npz') as w:
        for k in w.files:setattr(model,k,w[k])
    return atlas,basis,pick,model


def decode_core(a,atlas,basis,pick,xy):
    prepare(a,atlas,basis);guard_and_features(a,atlas,basis,pick['retention'],pick['strength'])
    a['original']=evaluate(a,xy)


def prepare_depth(a,atlas,basis):
    """No q, action label, direction, depth label or future field read here."""
    v=a['gate'];ids=np.flatnonzero(v)
    policy=DepthPolicy(memory_strength=1.,position_source='final',continuity=.01,field_veto_ratio=2.)
    z,cost,phi=depth_profiles(atlas,a['b'][v],a['original'][v,:2],a['h'][v],basis,policy)
    n=len(v);global_z=np.full(n,np.nan);branch_z=global_z.copy()
    x56=np.full((n,56),np.nan);x59=np.full((n,59),np.nan)
    tracker=MagneticDepthTracker(policy);history=CausalXYFeatureHistory();fast=FastDepthContext(atlas)
    raw=atlas.features(a['b']);j=0;veto=0
    for i in range(n):
        if not v[i]:tracker.reset();history.reset();fast.reset();continue
        if a.get('independent_frames'):tracker.reset();history.reset();fast.reset()
        now=int(a['t'][i]*1e9)+10_000_000_000
        r=tracker.step_profile(now,z,cost[j],phi[j],token=0);branch_z[i]=r['z_mm'];global_z[i]=r['global_z_mm']
        veto+=r['field_veto']
        f=history.push(time_ns=now,b0_id=0,stream_id=0,xyz=a['original'][i],phi=raw[i],memory=a['h'][i])
        x56[i]=np.r_[f,r['z_mm']/3,np.log1p(cost[j].min()),np.log1p(np.interp(a['original'][i,2],z,cost[j]))]
        x59[i]=fast.push(now,x56[i],token=0);j+=1
    a.update(z=z,cost=cost,depth_phi=phi,z_global=global_z,z_branch=branch_z,
             x56=x56,x59=x59,profile_veto_count=int(veto))


def z_score(a,p):
    s=metrics(p,a['q'],a['t'],a['primary'],a['gate'])
    mask=a['gate']&a['primary'];pair=mask[1:]&mask[:-1]&(np.diff(a['t'])<=.25)
    if a.get('independent_frames'):
        pair[:]=False
        # Unrelated calibration points are not adjacent samples of a motion.
        s['constant_reference_z_steps']={'pairs':0,'p95_mm':None,'over_0p2_mm':0}
    error=p[:,2]-a['q'][:,2];step=np.abs(np.diff(error))[pair]
    s['z_error_step_p95_mm']=float(np.quantile(step,.95)) if len(step) else 0.
    s['z_error_step_max_mm']=float(step.max()) if len(step) else 0.
    return s


def objective(rows):
    return float(np.mean([r['z_mae_mm']+.25*r['z_error_step_p95_mm']+.05*r['z_error_step_max_mm'] for r in rows]))


def depth_predictions(a,teacher,support,policy,increment=None,gain=0.):
    v=a['gate'];base=a['original'].copy();out={k:base.copy() for k in
        ('static_z','current_profile','field_branch','neural_unconstrained','unified','fast_slow')}
    out['current_profile'][v,2]=a['z_global'][v];out['field_branch'][v,2]=a['z_branch'][v]
    delta=np.clip(teacher.predict(a['x56'][v]).reshape(-1),-1,1)
    proposal=np.clip(a['z_branch'][v]+delta,a['z'][0],a['z'][-1])
    distance=support.distance(a['x56'][v])
    inc=np.zeros(len(proposal)) if increment is None else gain*np.clip(increment.predict(a['x59'][v]).reshape(-1),-.4,.4)
    out['neural_unconstrained'][v,2]=proposal
    verdict=[UnifiedDepthAdjudicator(policy),UnifiedDepthAdjudicator(policy)]
    restrictions=[0,0];j=0
    for i in range(len(v)):
        if not v[i]:
            for d in verdict:d.reset()
            continue
        if a.get('independent_frames'):
            for d in verdict:d.reset()
        now=int(a['t'][i]*1e9)+10_000_000_000
        for k,(key,extra) in enumerate((('unified',0),('fast_slow',inc[j]))):
            r=verdict[k].step(now,a['z'],a['cost'][j],a['depth_phi'][j],
                float(np.clip(proposal[j]+extra,a['z'][0],a['z'][-1])),distance[j],token=0)
            out[key][i,2]=r['z_mm'];restrictions[k]+=r['proposal_field_restricted']
        j+=1
    return out,restrictions


def optimize(x,y,w,valid_check,*,epochs,seed,learning_rate,initial=None,pairs=None):
    """Same tanh depth MLP and Adam equations as the paper's Z training."""
    model=DepthMLP(32,seed);d=x.shape[1];rng=model.rng
    if initial is None:
        model.mean=np.average(x,axis=0,weights=w)
        model.scale=np.maximum(np.sqrt(np.average((x-model.mean)**2,axis=0,weights=w)),.1)
        p=[rng.normal(0,np.sqrt(2/d),(d,32)),np.zeros(32),rng.normal(0,np.sqrt(2/32),(32,32)),
           np.zeros(32),rng.normal(0,.02,(32,1)),np.zeros(1)]
    else:
        model.mean=np.r_[initial.mean,np.average(x[:,56:],axis=0,weights=w)]
        model.scale=np.r_[initial.scale,np.maximum(x[:,56:].std(0),.1)]
        p=[v.copy() for v in initial.params];p[0]=np.vstack((p[0],np.zeros((3,32))));p[4]*=0;p[5]*=0
    u=np.clip((x-model.mean)/model.scale,-8,8);m=[np.zeros_like(a) for a in p];v=[a.copy() for a in m]
    step=0;best=None;best_params=None;history=[]
    def forward(ix):
        h1=np.tanh(u[ix]@p[0]+p[1]);h2=np.tanh(h1@p[2]+p[3]);return h1,h2,h2@p[4]+p[5]
    def gradients(ix,g):
        h1,h2,_=forward(ix);g2=(g@p[4].T)*(1-h2*h2);g1=(g2@p[2].T)*(1-h1*h1)
        return [u[ix].T@g1,g1.sum(0),h1.T@g2,g2.sum(0),h2.T@g,g.sum(0)]
    for epoch in range(1,epochs+1):
        for ix in np.array_split(rng.permutation(len(x)),max(1,(len(x)+255)//256)):
            pred=forward(ix)[2]
            g=(2*(pred-y[ix])+(pred if pairs is not None else 0))*w[ix,None]/len(ix)
            gg=gradients(ix,g)
            if pairs is not None and len(pairs):
                pp=pairs[rng.integers(0,len(pairs),len(ix))];ia,ib=pp[:,0],pp[:,1]
                g=(forward(ib)[2]-forward(ia)[2]-(y[ib]-y[ia]))*((w[ia]+w[ib])/2)[:,None]/len(ix)
                aa,bb=gradients(ia,-g),gradients(ib,g);gg=[g+a+b for g,a,b in zip(gg,aa,bb)]
            for k in (0,2,4):gg[k]+=.001*p[k]
            step+=1
            for par,g,mm,vv in zip(p,gg,m,v):
                mm*=.9;mm+=.1*g;vv*=.999;vv+=.001*g*g
                par-=learning_rate*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
        if epoch%10:continue
        model.params=p;results=valid_check(model)
        for r in results:
            r['epoch']=epoch;history.append(r)
            if r['qualified'] and (best is None or r['objective']<best['objective']):
                best=r.copy();best_params=[a.copy() for a in p]
        print('virtual Z',d,epoch,'eligible',sum(r['qualified'] for r in results),flush=True)
    # Preserve the tested fallback explicitly, never promote by test scores.
    if best is None:
        best_params=[a.copy() for a in p];best_params[4]*=0;best_params[5]*=0
    model.params=best_params;return model,best,history


def train():
    DEPTH.mkdir(parents=True,exist_ok=True)
    if (DEPTH/'freeze.json').exists():print('Preserved virtual depth freeze');return
    started=time.perf_counter();atlas,basis,pick,xy=core_context()
    generator=AxisForward.load(GEN/'models/generator.json')
    sigma=np.array(json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))['noise_sigma_uT'])
    train,valid,_,_=make_trials(generator,sigma)
    declared={'training_ids':[a['id'] for a in train],'validation_ids':[a['id'] for a in valid],
        'fit_real_frames':0,'formal_inverse_weights_loaded':False,'generator_is_empirical':True,
        'teacher_graph':'56-32-32-1 tanh','increment_graph':'59-32-32-1 tanh',
        'teacher_loss':'Trial-equal residual MSE, static zero-residual anchor mass4, L2 .001',
        'increment_loss':'TRAIN-Trial-centered incremental MSE + .5 zero-increment MSE + .5 paired increment-difference MSE + L2 .001',
        'teacher_epochs':80,'increment_epochs':60,'seed':368,'gain_candidates':[.25,.5,1.],
        'sampling':'every second accepted working-domain frame','selection':'virtual standard validation only; per-family MAE <= baseline + .005, then macro MAE+.25 error-step P95+.05 error-step max',
        'test_usage':'60 previously generated complex paths and new parameter grid are not used for selection',
        'memory_policy':'fixed virtual-only R367 basis and selection','xy_policy':'fixed virtual-only R367 causal XY with anchors, seed7',
        'scientific_scope':'same algorithm structure and objectives, parameters fitted anew within virtual world; not numerical reproduction of measured scores'}
    dump(DEPTH/'declared.json',declared)
    for a in train+valid:decode_core(a,atlas,basis,pick,xy);prepare_depth(a,atlas,basis)
    cal=np.load(CORE/'virtual_static_calibration.npz');aq=cal['q'][::3];ap=atlas.features(atlas.forward(aq))
    anchor=np.column_stack((aq/[10,10,3],ap,np.zeros((len(aq),34)),aq[:,2]/3,np.zeros((len(aq),2))))
    xs=[];ys=[];ws=[];reference=[];owners=[]
    for owner,a in enumerate(train):
        ix=np.flatnonzero(a['gate']&a['primary'])[::2]
        xs.append(a['x56'][ix]);ys.append(np.clip(a['q'][ix,2]-a['z_branch'][ix],-1,1)[:,None]);ws.append(np.full(len(ix),1/len(ix)))
        ref=a['x56'][ix[np.linspace(0,len(ix)-1,min(64,len(ix))).astype(int)]]
        reference.append(ref);owners.extend([owner]*len(ref))
    xs.append(anchor);ys.append(np.zeros((len(anchor),1)));ws.append(np.full(len(anchor),4/len(anchor)))
    reference.append(anchor);owners.extend([-1]*len(anchor));reference=np.vstack(reference);owners=np.array(owners)
    x,y,w=np.vstack(xs),np.vstack(ys),np.concatenate(ws);w*=len(w)/w.sum()
    mean=np.average(x,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),.1)
    norm=np.clip((reference-mean)/scale,-8,8);support=SourceFeatureSupport(mean,scale,norm)
    loo=[]
    for owner in range(len(train)):
        loo.extend(SourceFeatureSupport(mean,scale,norm[owners!=owner]).distance(reference[owners==owner][::8]))
    policy=UnifiedDepthPolicy(neural_strength=4.,continuity=.01,support_limit=max(float(np.quantile(loo,.99)),1e-6))
    np.savez_compressed(DEPTH/'support.npz',mean=mean,scale=scale,reference=norm)
    baseline=[]
    for a in valid:
        p=a['original'].copy();p[a['gate'],2]=a['z_branch'][a['gate']];baseline.append(z_score(a,p))
    def check_teacher(model):
        ss=[z_score(a,depth_predictions(a,model,support,policy)[0]['unified']) for a in valid]
        qualified=all(s['z_mae_mm']<=old['z_mae_mm']+.005 for s,old in zip(ss,baseline))
        return [{'objective':objective(ss),'qualified':qualified,'scores':ss}]
    teacher,picked,history=optimize(x,y,w,check_teacher,epochs=80,seed=368,learning_rate=.001)
    teacher.save(DEPTH/'virtual_teacher.npz')
    dump(DEPTH/'teacher_selection.json',{'selected':picked,'history':history,'fallback_zero':picked is None})
    xs=[];ys=[];ws=[];pairs=[];cursor=0;nuisances=[]
    for a in train:
        p=depth_predictions(a,teacher,support,policy)[0]['unified'];ix=np.flatnonzero(a['gate']&a['primary'])[::2]
        residual=a['q'][ix,2]-p[ix,2];nuisance=float(residual.mean())
        xs.append(a['x59'][ix]);ys.append(np.clip(residual-nuisance,-.4,.4)[:,None]);ws.append(np.full(len(ix),1/len(ix)))
        good=np.diff(a['t'][ix])<=.35;ii=np.flatnonzero(good);pairs.extend(zip((cursor+ii).tolist(),(cursor+ii+1).tolist()));cursor+=len(ix)
        nuisances.append({'trial':a['id'],'training_only_constant_z_mm':nuisance})
    xs.append(np.column_stack((anchor,np.zeros((len(anchor),3)))));ys.append(np.zeros((len(anchor),1)));ws.append(np.full(len(anchor),4/len(anchor)))
    x,y,w=np.vstack(xs),np.vstack(ys),np.concatenate(ws);w*=len(w)/w.sum();baseline=[z_score(a,depth_predictions(a,teacher,support,policy)[0]['unified']) for a in valid]
    base_objective=objective(baseline)
    def check_increment(model):
        results=[]
        for gain in (.25,.5,1.):
            ss=[z_score(a,depth_predictions(a,teacher,support,policy,model,gain)[0]['fast_slow']) for a in valid]
            good=all(s['z_mae_mm']<=old['z_mae_mm']+.005 and s['z_error_step_p95_mm']<=old['z_error_step_p95_mm']+.005 for s,old in zip(ss,baseline))
            value=objective(ss);results.append({'gain':gain,'objective':value,'qualified':bool(good and value<.99*base_objective),'scores':ss})
        return results
    increment,inc_pick,inc_history=optimize(x,y,w,check_increment,epochs=60,seed=368,learning_rate=.0002,initial=teacher,pairs=np.array(pairs,int))
    increment.save(DEPTH/'virtual_increment.npz');dump(DEPTH/'increment_selection.json',{'selected':inc_pick,'history':inc_history,'fallback_zero':inc_pick is None,'nuisances_training_only':nuisances})
    dump(DEPTH/'freeze.json',{'policy':asdict(policy),'gain':inc_pick['gain'] if inc_pick else 0.,'virtual_training_only':True,
        'teacher_selected_epoch':picked['epoch'] if picked else None,'increment_selected_epoch':inc_pick['epoch'] if inc_pick else None,
        'teacher_fallback_zero':picked is None,'increment_fallback_zero':inc_pick is None,
        'dependencies':{str(p.relative_to(OUT)):sha(p) for p in (DEPTH/'virtual_teacher.npz',DEPTH/'virtual_increment.npz',DEPTH/'support.npz',DEPTH/'declared.json')},
        'wall_training_s':time.perf_counter()-started})
    print('virtual Z freeze',picked['epoch'] if picked else None,inc_pick,flush=True)


def frozen_context():
    atlas,basis,pick,xy=core_context();f=json.loads((DEPTH/'freeze.json').read_text(encoding='utf8'))
    for path,digest in f['dependencies'].items():
        if sha(OUT/path)!=digest:raise ValueError('Virtual depth freeze changed')
    return atlas,basis,pick,xy,DepthMLP.load(DEPTH/'virtual_teacher.npz'),SourceFeatureSupport.load(DEPTH/'support.npz'),UnifiedDepthPolicy(**f['policy']),DepthMLP.load(DEPTH/'virtual_increment.npz'),f['gain']


def test():
    if (DEPTH/'test_report.json').exists():print('Preserved existing Z tests');return
    ctx=frozen_context();atlas,basis,pick,xy=ctx[:4]
    sigma=np.array(json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))['noise_sigma_uT'])
    _,_,tests,stationary=make_trials(AxisForward.load(GEN/'models/generator.json'),sigma)
    tests.append(stationary)
    data=np.load(GEN/'complex_synthetic.npz');manifest=json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))
    for i,info in enumerate(manifest['trials']):
        lo,hi=data['offsets'][i:i+2];q,t,b=data['q'][lo:hi],data['t'][lo:hi],data['b'][lo:hi]
        tests.append({'id':info['id'],'family':info['family'],'q':q,'t':t,'b':b,
            'gate':np.linalg.norm(b,axis=1)>=70,'primary':(q[:,2]>=.4)&(q[:,2]<=3)})
    rows=[];folder=DEPTH/'test_predictions';folder.mkdir(exist_ok=True)
    for a in tests:
        decode_core(a,atlas,basis,pick,xy);prepare_depth(a,atlas,basis)
        pred,restricted=depth_predictions(a,*ctx[4:])
        np.savez_compressed(folder/(a['id']+'.npz'),q=a['q'],t=a['t'],gate=a['gate'],**pred)
        rows.append({'id':a['id'],'family':a['family'],'scores':{k:z_score(a,p) for k,p in pred.items()},'restricted_proposals':restricted,'profile_veto_count':a['profile_veto_count']})
        if len(rows)%10==0:print('virtual Z tests',len(rows),flush=True)
    dump(DEPTH/'test_report.json',{'rows':rows,'test_selection':False,'xy_and_output_permission_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['train','test']);a=p.parse_args()
    {'train':train,'test':test}[a.task]()
