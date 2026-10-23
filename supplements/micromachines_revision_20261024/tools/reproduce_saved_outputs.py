"""Portable execution of the frozen saved-output reducers, without model inference.

Original reducer and metric source bytes remain unchanged. Only ROOT/output
locations and the archived absolute input-root are mapped at runtime. Small
pure numerical functions are loaded from their pinned source AST so neither
hardware modules nor optional training libraries are imported.
"""
from pathlib import Path
import argparse,ast,collections,csv,hashlib,importlib.util,json,math,os,re,sys,time,types,zipfile
import numpy as np

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pure_module(name,source,functions,extra=None):
    module=types.ModuleType(name)
    module.__dict__.update(np=np,Path=Path,csv=csv,re=re,json=json,math=math,collections=collections)
    if extra:module.__dict__.update(extra)
    tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in functions]
    assert {n.name for n in nodes}==set(functions),(name,functions)
    compiled=compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec',flags=__import__('__future__').annotations.compiler_flag)
    exec(compiled,module.__dict__)
    sys.modules[name]=module
    return module

def scientific_equal(a,b,path='root'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:scientific_equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):scientific_equal(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert isinstance(b,(float,int)) and (a==b or math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)),(path,a,b)
    else:assert a==b,(path,a,b)

def verify_original(root,out,refs,masks):
    scripts=root/'submitted/File_S1_scripts.zip'
    with zipfile.ZipFile(scripts) as z:
        metrics_source=z.read('File_S1_scripts/metrics.py').decode('utf8');ema_source=z.read('File_S1_scripts/control_ema.py').decode('utf8')
    def functions(text,names,extra=None):
        namespace={'np':np}
        if extra:namespace.update(extra)
        tree=ast.parse(text);nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        assert len(nodes)==len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<submitted-pure-metrics>','exec'),namespace)
        return namespace
    metrics=functions(metrics_source,['dwell_rms','trial_metrics']);ema=functions(ema_source,['ema_xy'])['ema_xy'];trial_metrics=metrics['trial_metrics']
    old=root/'analysis/r460_source_real_history_refinement_20261004';spec=load_json(old/'frozen_real_replay_spec.json');arms={'H0':'R373','H1':'history_position','H2':'history_motion_hold_anchor'};cache={}
    for item in spec['entries']:
        streams={}
        for arm in arms.values():
            file=old/'new_trajectory_native_replay/native'/item['device']/arm/(item['id']+'.npz')
            if arm=='R373' and not file.exists():file=root/'analysis/r395_new_real_simulation_paper_20261003/native'/item['device']/arm/(item['id']+'.npz')
            with np.load(file,allow_pickle=False) as f:streams[arm]={k:f[k].copy() for k in f.files}
        a=streams['R373'];t=a['t_ns'];folder=root/item['path'];segments=refs.command_segments(refs.read_rows(folder/'stage_commands.csv'));reference=refs.command_reference(t,segments,refs.reference_origin(load_json(folder/'meta.json')));q=reference['xyz_mm'];phase=np.array([segments[i]['phase'] if i>=0 else 'before_stage' for i in reference['command_segment_id']]);primary,action=masks.mask_for(item['mode'],phase,reference);finite=np.isfinite(a['b']).all(1);primary&=finite;action=action&(q[:,2]>=.4)&finite
        shared=np.logical_and.reduce([s['ok']&np.isfinite(s['xyz']).all(1) for s in streams.values()]);dt=np.r_[1.,np.diff(t)/1e9];velocity=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None];dwell=action&(np.linalg.norm(velocity,axis=1)<.02)
        cache[item['id']]={'t':t,'q':q,'phase':phase,'primary':primary,'action':action,'shared':shared,'dwell':dwell,'xyz':{arm:s['xyz'] for arm,s in streams.items()},'ok':{arm:s['ok'] for arm,s in streams.items()}}
    golden=list(csv.DictReader((root/'submitted/Table_S1_per_trial_metrics.csv').open(encoding='utf-8-sig')));count=0;rows=[]
    mapping={'E_xy_mm':'xy','E_xy_p95_mm':'xy_p95','E_xy_centered_mm':'cxy','E_xyz_mm':'xyz','Z_MAE_mm':'z','step_rms_mm':'step_rms','jump_rate_0p1_pct':'jump_0.1','jump_rate_0p2_pct':'jump_0.2','jump_rate_0p3_pct':'jump_0.3','dwell_rms_mm':'dwell'}
    for row in golden:
        c=cache[row['trial_id']];model=row['model'];arm=arms.get(model)
        if arm is None:
            m=re.fullmatch(r'(H0|H2)\+EMA(?:_)?(0\.\d+)',model);assert m and float(m[2]) in [.02,.03,.05,.075,.1,.15,.2,.3],model
            arm=arms[m[1]];base=dict(c);base['xyz']={'R373':c['xyz'][arm]};base['ok']={'R373':c['ok'][arm]};current=dict(c);current['xyz']=dict(c['xyz']);current['xyz'][arm]=ema(base,float(m[2]));c=current
        result=trial_metrics(c,arm,thresholds=(.1,.2,.3))
        assert result['frames']==int(row['frames']) and result['pairs']==int(row['pairs'])
        for column,key in mapping.items():
            if row[column] in ['', 'nan']:assert not np.isfinite(result[key]);continue
            precision=3 if column.startswith('jump_') else 5
            assert round(result[key],precision)==float(row[column]),(row['trial_id'],model,column,result[key],row[column])
            count+=1
        for column,index in [('offset_x_mm',0),('offset_y_mm',1)]:assert round(result['bias'][index],5)==float(row[column]);count+=1
        rows.append({'trial_id':row['trial_id'],'model':model,'metrics':result})
    target=out/'original';target.mkdir();(target/'reproduced_metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=True),encoding='utf8')
    return {'phase':'original-submitted-table','all_published_cells_match':True,'rows':len(golden),'numeric_cells_checked':count,'source_csv_sha256':digest(root/'submitted/Table_S1_per_trial_metrics.csv'),'rounding':'Exact agreement at submitted CSV precision: 5 decimals for errors/offset, 3 for jump percentages; counts exact.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--phase',choices=['original','matched','archived','both','all'],default='all');args=ap.parse_args()
    root=args.root.resolve();out=args.output.resolve();assert not out.exists(),'Use a fresh output directory; frozen files are never overwritten.'
    out.mkdir(parents=True);start=time.monotonic();cpu0=time.process_time();O=root/'analysis/micromachines_revision_validation_20261023'
    manifest=load_json(root/'FILE_MANIFEST.json')
    for row in manifest['files']:
        file=root/row['path'];assert file.is_file() and file.stat().st_size==row['bytes'] and digest(file)==row['sha256'],row['path']
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if hasattr(os,'sched_getaffinity'):
        # Explicit one-to-four-CPU finite reduction. No other process is touched.
        cpus=sorted(os.sched_getaffinity(0));allowed=set(cpus[:4])
        original_setaffinity=os.sched_setaffinity
        os.sched_setaffinity=lambda pid,requested:original_setaffinity(pid,allowed)
    pkg=types.ModuleType('training');pkg.__path__=[];sys.modules['training']=pkg
    sys.modules['training.domain_randomization_r369']=types.ModuleType('training.domain_randomization_r369')
    refs=pure_module('portable_reference',root/'training/build_trajectory_dataset_v1.py',['read_rows','number','reference_origin','command_segments','command_reference'])
    masks=pure_module('portable_masks',root/'training/analyze_r354_final_online.py',['mask_for'])
    base=pure_module('training.real_simulation_paper_r395',root/'training/real_simulation_paper_r395.py',['clean','metric'],{**{n:getattr(refs,n) for n in ['read_rows','reference_origin','command_segments','command_reference']},'mask_for':masks.mask_for})
    pkg.real_simulation_paper_r395=base
    pure_module('training.real_jump_audit_r402',root/'training/real_jump_audit_r402.py',['score','dwell_score','summarize'])
    pure_module('training.frozen_real_replay_r405',root/'training/frozen_real_replay_r405.py',['dump'])
    pure_module('native_online_replay_r387',root/'code/native_online_replay_r387.py',['raw_csv_group'])
    vector=pure_module('live_normalized_fit',root/'code/replay_backtest.py',['vector'])
    phases=['matched','archived'] if args.phase in ['both','all'] else ([] if args.phase=='original' else [args.phase]);results=[]
    if args.phase in ['original','all']:results.append(verify_original(root,out,refs,masks))
    for phase in phases:
        relative='matched_statistics_v1' if phase=='matched' else 'archived_validation_v1/statistics_v2'
        frozen=O/relative;target=out/phase;target.mkdir()
        contract=load_json(frozen/'protocol_v1.json');(target/'protocol_v1.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf8')
        name='mm_matched_reduction_20261023.py' if phase=='matched' else 'mm_archived_reduce_v2_20261023.py'
        source=O/name;assert digest(source)==contract['worker_sha256']
        spec=importlib.util.spec_from_file_location('frozen_'+phase,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if phase=='matched':module.ROOT=root;module.O=O;module.N=O/'native_matched_v1';module.D=target
        else:
            module.R=root;module.O=O;module.D=O/'archived_validation_v1';module.A=target
            original_read=module.read
            def mapped_read(path):
                value=original_read(path)
                if Path(path).resolve()==(O/'archived_validation_v1/protocol_v1.json').resolve():
                    value['input_root']=str(O/'candidate_inputs_v1/S20261010_A')
                return value
            module.read=mapped_read
        module.main()
        expected=load_json(frozen/'report.json');actual=load_json(target/'report.json')
        sections=['own_summaries','paired_summaries','order_seed_summaries','own_record_details','paired_record_details','input_hashes'] if phase=='matched' else ['summaries','paired_summaries','record_details','paired_record_details','input_hashes','validity_correction_audits']
        for section in sections:scientific_equal(expected[section],actual[section],phase+'.'+section)
        results.append({'phase':phase,'all_scientific_sections_match':True,'sections':sections,'original_report_sha256':digest(frozen/'report.json'),'reproduced_report_sha256':digest(target/'report.json')})
    receipt={'status':'PASS_PORTABLE_SAVED_OUTPUT_REDUCTION','results':results,'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu0,'new_inference_calls':0,'new_training_updates':0,'hardware_operations':0,'root_mapping_only':True,'scope':'Saved-output statistics verified; full model inference and training are distinct, not rerun here.'}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt))

if __name__=='__main__':main()
