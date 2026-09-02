#!/usr/bin/env python3
"""Exploratory lag/directed-information validation for frozen-slot historical draws.

This script NEVER changes or contributes to the frozen prospective e-process.
It asks whether the weak full-sample lag-MI signal among historically frozen
RNG-marked slots has out-of-sample predictive value after controlling for the
slot schedule itself.
"""
import json, math, os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

DATA = Path(os.getenv('LOTO_ARCHIVE', '.research-sportloto/draws.json'))
OUT = Path(os.getenv('LOTO_OUT', '.research-sportloto/out'))
OUT.mkdir(parents=True, exist_ok=True)
P0 = np.array([0.375644995,0.462332301,0.148606811,0.013209494,0.000206398], dtype=float)
SLOTS = ('12:07','13:52','16:07','16:22','20:07','23:22')
FREEZE = datetime.fromisoformat('2026-08-26T14:21:00+03:00')
EPS=1e-15


def dt(x):
    return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))

def kval(r):
    return len(set(r['fieldA']) & set(r['fieldB']))

def norm(v):
    v=np.asarray(v,float); s=v.sum(); return v/s if s else np.ones_like(v)/len(v)

def mi(seq):
    if len(seq)<2: return 0.0
    x=np.asarray(seq[:-1],int); y=np.asarray(seq[1:],int); n=len(x)
    cx=np.bincount(x,minlength=5); cy=np.bincount(y,minlength=5)
    cxy=np.zeros((5,5),int)
    np.add.at(cxy,(x,y),1)
    out=0.0
    for i in range(5):
        for j in range(5):
            c=cxy[i,j]
            if c:
                out += (c/n)*math.log((c*n)/(cx[i]*cy[j]))
    return out

def cmi(seq, slots):
    # I(K_t ; K_{t-1} | previous-slot,current-slot), weighted by transition context.
    groups=defaultdict(list)
    for t in range(1,len(seq)):
        groups[(slots[t-1],slots[t])].append((seq[t-1],seq[t]))
    n=sum(len(v) for v in groups.values())
    total=0.0
    for pairs in groups.values():
        if len(pairs)<2: continue
        x=np.array([a for a,b in pairs],int); y=np.array([b for a,b in pairs],int); m=len(x)
        cx=np.bincount(x,minlength=5); cy=np.bincount(y,minlength=5); cxy=np.zeros((5,5),int)
        np.add.at(cxy,(x,y),1)
        local=0.0
        for i in range(5):
            for j in range(5):
                z=cxy[i,j]
                if z:
                    local += (z/m)*math.log((z*m)/(cx[i]*cy[j]))
        total += (m/n)*local
    return total

def slot_permutation_null(K, slots, reps=20000, seed=20260902):
    """Shuffle K independently within each slot, preserving slot marginals/schedule."""
    rng=np.random.default_rng(seed)
    K=np.asarray(K,int); slots=np.asarray(slots,object)
    idx={s:np.where(slots==s)[0] for s in SLOTS}
    vals={s:K[ix].copy() for s,ix in idx.items()}
    obs_mi=mi(K.tolist()); obs_cmi=cmi(K.tolist(),slots.tolist())
    ge_mi=1; ge_cmi=1
    samples_mi=[]; samples_cmi=[]
    for b in range(reps):
        z=K.copy()
        for s in SLOTS:
            z[idx[s]]=rng.permutation(vals[s])
        m=mi(z.tolist()); cm=cmi(z.tolist(),slots.tolist())
        ge_mi += (m >= obs_mi-EPS)
        ge_cmi += (cm >= obs_cmi-EPS)
        if b<2000:
            samples_mi.append(m); samples_cmi.append(cm)
    return {
        'reps': reps,
        'observedMI_nats': obs_mi,
        'slotPreservingPermutationP_MI': ge_mi/(reps+1),
        'observedConditionalMI_nats_givenPrevCurrentSlot': obs_cmi,
        'slotPreservingPermutationP_CMI': ge_cmi/(reps+1),
        'nullMI_mean_first2000': float(np.mean(samples_mi)),
        'nullCMI_mean_first2000': float(np.mean(samples_cmi)),
    }

def fit_models(K, slots, cut):
    K=np.asarray(K,int); slots=np.asarray(slots,object)
    train_idx=np.arange(cut)
    # Slot baseline, shrunk toward IID p0 with fixed 20 pseudo-observations.
    slot_counts={s:np.bincount(K[train_idx][slots[train_idx]==s],minlength=5).astype(float) for s in SLOTS}
    slot_p={s:norm(slot_counts[s] + 20*P0) for s in SLOTS}
    # Transition-by-slot-pair baseline and lag model, both trained only before cut.
    pair_counts=defaultdict(lambda: np.zeros(5,float))
    lag_counts=defaultdict(lambda: np.zeros(5,float))
    for t in range(1,cut):
        c=(slots[t-1],slots[t]); pair_counts[c][K[t]] += 1
        lag_counts[(slots[t-1],slots[t],int(K[t-1]))][K[t]] += 1
    pair_p={}
    lag_p={}
    for c,cnt in pair_counts.items():
        # Prior center current-slot baseline; strength 20 fixed a priori for this exploratory validation.
        pair_p[c]=norm(cnt + 20*slot_p[c[1]])
    for key,cnt in lag_counts.items():
        c=key[:2]
        center=pair_p.get(c,slot_p[c[1]])
        lag_p[key]=norm(cnt + 20*center)

    rows=[]
    start=max(cut,1)
    for t in range(start,len(K)):
        prev=int(K[t-1]); y=int(K[t]); s=slots[t]; c=(slots[t-1],s)
        iid=P0
        sp=slot_p[s]
        pp=pair_p.get(c,sp)
        lp=lag_p.get((slots[t-1],s,prev),pp)
        one=np.zeros(5); one[y]=1
        brier=lambda p: float(np.sum((p-one)**2))
        rows.append({
            'draw_index':int(t),'prevK':prev,'K':y,'prevSlot':str(slots[t-1]),'slot':str(s),
            'loggain_slot_vs_iid':float(math.log(max(sp[y],EPS)/max(iid[y],EPS))),
            'loggain_pairslot_vs_slot':float(math.log(max(pp[y],EPS)/max(sp[y],EPS))),
            'loggain_lag_vs_pairslot':float(math.log(max(lp[y],EPS)/max(pp[y],EPS))),
            'loggain_lag_vs_iid':float(math.log(max(lp[y],EPS)/max(iid[y],EPS))),
            'brierDiff_lag_minus_pairslot':brier(lp)-brier(pp),
            'brierDiff_lag_minus_iid':brier(lp)-brier(iid),
        })
    return rows

def moving_block_ci(values, block=12, reps=10000, seed=260826):
    x=np.asarray(values,float); n=len(x)
    if n==0: return [None,None]
    rng=np.random.default_rng(seed); means=np.empty(reps)
    starts=np.arange(max(1,n-block+1))
    nb=math.ceil(n/block)
    for b in range(reps):
        inds=[]
        for st in rng.choice(starts,size=nb,replace=True):
            inds.extend(range(int(st),min(int(st)+block,n)))
        means[b]=x[np.asarray(inds[:n],int)].mean()
    return [float(np.quantile(means,0.025)),float(np.quantile(means,0.975))]

def transition_matrix(seq):
    M=np.zeros((5,5),int)
    for a,b in zip(seq[:-1],seq[1:]): M[int(a),int(b)]+=1
    return M.tolist()

def summarize_half(seq):
    return {'n':len(seq),'counts':np.bincount(np.asarray(seq,int),minlength=5).tolist(),'meanK':float(np.mean(seq)),'MI_nats':mi(seq),'transitionCounts':transition_matrix(seq)}

def main():
    arc=json.load(open(DATA))
    rows=[]
    for r in arc['draws']:
        d=dt(r['date']); s=d.strftime('%H:%M')
        if d < FREEZE and s in SLOTS:
            rows.append((r['number'],d,s,kval(r)))
    rows.sort(key=lambda z:z[1])
    K=[r[3] for r in rows]; slots=[r[2] for r in rows]
    n=len(K); cut=n//2
    null=slot_permutation_null(K,slots)
    pred=fit_models(K,slots,cut)
    g=np.array([r['loggain_lag_vs_pairslot'] for r in pred])
    gi=np.array([r['loggain_lag_vs_iid'] for r in pred])
    bd=np.array([r['brierDiff_lag_minus_pairslot'] for r in pred])
    report={
        'generatedAt':datetime.now().astimezone().isoformat(),
        'archiveLast':arc['last'],
        'scope':'EXPLORATORY ONLY; frozen prospective protocol unchanged',
        'historicalFrozenSlotSequence':{
            'n':n,'firstDraw':rows[0][0] if rows else None,'lastDraw':rows[-1][0] if rows else None,
            'firstDate':rows[0][1].isoformat() if rows else None,'lastDate':rows[-1][1].isoformat() if rows else None,
            'slotCounts':dict(Counter(slots)),'full':summarize_half(K),
        },
        'slotPreservingLagNull':null,
        'chronologicalSplit':{
            'cut':cut,
            'trainFirstDraw':rows[0][0],'trainLastDraw':rows[cut-1][0],
            'testFirstDraw':rows[cut][0],'testLastDraw':rows[-1][0],
            'train':summarize_half(K[:cut]),'test':summarize_half(K[cut:]),
        },
        'outOfSampleLagModel':{
            'testN':len(pred),
            'fixedShrinkagePseudoCount':20,
            'cumulativeLogGain_lag_vs_slotPairBaseline':float(g.sum()),
            'meanLogGain_lag_vs_slotPairBaseline':float(g.mean()),
            'movingBlock95CI_meanLogGain_lag_vs_slotPairBaseline':moving_block_ci(g),
            'cumulativeLogGain_lag_vs_IID':float(gi.sum()),
            'meanBrierDiff_lag_minus_slotPairBaseline':float(bd.mean()),
            'operationalReplicates': bool(g.sum()>0 and moving_block_ci(g)[0] is not None and moving_block_ci(g)[0]>0),
        },
        'interpretation':{
            'fullSampleLagMI_isNominallyInteresting': bool(null['slotPreservingPermutationP_MI']<0.05),
            'conditionalMI_isNominallyInteresting': bool(null['slotPreservingPermutationP_CMI']<0.05),
            'lagAddsOutOfSampleInformationBeyondSlotSchedule': bool(g.sum()>0 and moving_block_ci(g)[0] is not None and moving_block_ci(g)[0]>0),
        }
    }
    out=OUT/'lag-validation.json'; json.dump(report,open(out,'w'),ensure_ascii=False,indent=2)
    md=[]
    md.append('# Frozen-slot lag validation (exploratory only)')
    md.append(f"N={n}; train={cut}, test={n-cut}; archiveLast={arc['last']}")
    md.append(f"Full MI={null['observedMI_nats']:.6g}, slot-preserving p={null['slotPreservingPermutationP_MI']:.6g}")
    md.append(f"Conditional MI given prev/current slot={null['observedConditionalMI_nats_givenPrevCurrentSlot']:.6g}, slot-preserving p={null['slotPreservingPermutationP_CMI']:.6g}")
    md.append(f"OOS lag vs slot-pair baseline: cumulative loggain={g.sum():.6g}; mean={g.mean():.6g}; block95={moving_block_ci(g)}; mean Brier diff={bd.mean():.6g}")
    md.append(f"Replicates out of sample: {report['outOfSampleLagModel']['operationalReplicates']}")
    (OUT/'lag-validation.md').write_text('\n\n'.join(md)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
