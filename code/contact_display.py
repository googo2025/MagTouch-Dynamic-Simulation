"""Display-only contact policy. Never replace raw predictions with motion plans."""
import json
import math
from replay_backtest import RAW_FIELDS, record


def decoded(value, default=None):
    if not isinstance(value,str): return value if value is not None else default
    try: return json.loads(value)
    except (TypeError,ValueError): return default


def truth(value):
    if value is None or value=='': return None
    if isinstance(value,str):
        if value.lower() in ('false','0','no'): return False
        if value.lower() in ('true','1','yes'): return True
        return None
    return bool(value)


def contact_visible(raw, extra, valid=True, normal_state='unknown', family=''):
    """Return (visible, reason). This is inferred contact, not measured ground truth."""
    if truth(valid) is False: return False,'预测标记无有效接触'
    annotated=truth(extra.get('display.contact'))
    if annotated is not None: return annotated,str(extra.get('display.contact_reason','记录中的显示接触判定'))
    mode=raw.get('mode5')
    if isinstance(mode,str): mode=decoded(mode,mode)
    label=str(mode.get('label','') if isinstance(mode,dict) else mode or '').strip().lower()
    state=str(extra.get('adapter.normal_state','unknown') if normal_state in (None,'','unknown') else normal_state).lower()
    if label in ('none','noncontact','no_contact','no-contact','idle','air','release'):
        return False,'模式判定非接触/释放：'+label
    if state in ('release','noncontact','no_contact','idle','air'):
        return False,'派生状态为非接触/释放：'+state
    if label in ('press','slide','slide_x','slide_y','compound','hold'):
        return True,'运动模式判定接触：'+label
    if family=='NormalizedFit': return truth(valid) is True,'拟合响应/相似度接触门控'
    adapter=truth(extra.get('adapter.interaction_valid'))
    if adapter is not None: return adapter,'模型适配器接触门控（非独立测量）'
    # Legacy H1/R15 files may mark every numeric output valid. A coordinate
    # far from the origin is not contact: use a depth head, never XY or |XYZ|.
    for key in ('z','Z','dZ'):
        try:
            value=float(raw[key])
            if math.isfinite(value): return value>0.02,'旧模型深度门控：'+key+'>0.02 mm（非接触真值）'
        except (KeyError,TypeError,ValueError): pass
    return False,'缺少可用接触判定；隐藏未确认轨迹'


def prediction_contact(p):
    if not all(math.isfinite(v) for v in p.visual_xyz_mm): return False,'三维输出无效'
    return contact_visible(p.raw_fields,p.extra_outputs,p.interaction_valid,getattr(p,'normal_state','unknown'),p.family)


def history_record(t, row, xyz):
    raw=decoded(row.get('raw_fields_json'),{})
    if not isinstance(raw,dict): raw={}
    raw=dict(raw)
    for key in RAW_FIELDS:
        if key not in raw:
            value=row.get(key)
            raw[key]=decoded(value,value) if value not in (None,'') else None
    extra=decoded(row.get('extra_outputs_json'),{})
    if not isinstance(extra,dict):extra={}
    extra=dict(extra)
    r=record(t,raw,xyz,row.get('model_family','--'),extra)
    r['valid']=r['valid'] and truth(row.get('interaction_valid')) is not False
    visible,reason=contact_visible(raw,extra,r['valid'],row.get('normal_state','unknown'),r['family'])
    r['extra'].update({'display.contact':visible,'display.contact_reason':reason,
                  'display.source':'recorded live_predictions.csv'})
    r['status']='原记录：'+reason
    return r


def display_point_visible(record):
    """Apply contact semantics without changing any raw prediction value."""
    visible,reason=contact_visible(record['raw'],record['extra'],record['valid'],family=record['family'])
    if not visible or record.get('xyz') is None:return False,reason
    try:valid_xyz=all(math.isfinite(float(value)) for value in record['xyz'][:3])
    except (TypeError,ValueError,IndexError):valid_xyz=False
    return (True,reason) if valid_xyz else (False,'三维输出无效')


def contact_window(records, limit=160, edge_guard_ms=120.0):
    """Hide rising and falling prefixes/suffixes without editing predictions.

    Completed contact segments lose ``edge_guard_ms`` at both ends.  The active
    segment loses only its known rising edge; if it later releases, replay will
    retroactively remove its falling-edge suffix.  ``None`` separates touches.
    """
    try:guard_ns=max(0,int(float(edge_guard_ms)*1e6))
    except (TypeError,ValueError):guard_ns=120_000_000
    out=[];segment=[];previous=None

    def flush(release_t=None):
        nonlocal segment
        if not segment:return
        onset=segment[0][0]
        end=(None if release_t is None else int(release_t)-guard_ns)
        points=[xyz for t,xyz in segment
                if t-onset>=guard_ns and (end is None or t<=end)]
        if points:
            if out and out[-1] is not None:out.append(None)
            out.extend(points)
            if release_t is not None:out.append(None)
        segment=[]

    for r in records[-limit:]:
        t=int(r['t_ns'])
        if previous is not None and (t<=previous or t-previous>1_000_000_000):
            flush(t)
        visible,_=display_point_visible(r)
        if visible and r['xyz'] is not None:segment.append((t,r['xyz']))
        else:flush(t)
        previous=t
    flush(None)
    return out
