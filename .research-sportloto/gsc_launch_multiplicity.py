#!/usr/bin/env python3
import json, math, os, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
TARGET_DATE='2025-10-15'
TARGET_SLOTS={'12:07','16:07','16:22','20:07'}

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def k_of(a,b): return len(set(a)&set(b))
def le(k): return math.log(Q[k]/P0[k])
def exp_safe(x): return 0.0 if x<-745 else ('>=1e304' if x>700 else math.exp(x))

def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw; src.sort(key=lambda x:int(x['number'])); rows=[]
    for i,d in enumerate(src):
        A=list(map(int,d.get('fieldA',d.get('field1')))); B=list(map(int,d.get('fieldB',d.get('field2')))); z=dt(d.get('date',d.get('draw_date'))).astimezone(TZ)
        rows.append({'i':i,'number':int(d['number']),'k':k_of(A,B),'slot':z.strftime('%H:%M'),'date':z.date().isoformat()})
    return raw,rows

def stable_eras(rows):
    by=defaultdict(list)
    for r in rows: by[r['date']].append(r)
    raw=[]
    for d in sorted(by):
        sig=tuple(sorted({r['slot'] for r in by[d]})); a=min(r['i'] for r in by[d]); b=max(r['i'] for r in by[d])
        if raw and raw[-1]['sig']==sig:
            raw[-1]['end']=d; raw[-1]['last']=b; raw[-1]['days']+=1
        else: raw.append({'start':d,'end':d,'first':a,'last':b,'days':1,'sig':sig})
    return [e for e in raw if e['days']>=3]

def summary_idxs(rows,idxs,ks=None):
    vals=[(ks[i] if ks is not None else rows[i]['k']) for i in idxs]; c=[0]*5
    for x in vals:c[x]+=1
    lr=sum(c[i]*le(i) for i in range(5)); return {'n':len(vals),'counts':c,'meanK':sum(vals)/len(vals),'sumK':sum(vals),'logLR':lr,'E':exp_safe(lr),'firstDraw':rows[idxs[0]]['number'],'lastDraw':rows[idxs[-1]]['number']}

def cohort_indices(rows,eras):
    cohorts=[]
    for j in range(1,len(eras)):
        prev,cur=eras[j-1],eras[j]; added=set(cur['sig'])-set(prev['sig'])
        if not added: continue
        idx=[r['i'] for r in rows if cur['start']<=r['date']<=cur['end'] and r['slot'] in added][:50]
        if len(idx)>=25: cohorts.append({'start':cur['start'],'addedSlots':sorted(added),'idx':idx})
    return cohorts

def targeted_indices(rows):
    allx=[r['i'] for r in rows if r['date']>=TARGET_DATE and r['slot'] in TARGET_SLOTS]
    first=allx[:50]; nxt=allx[50:100]
    return first,nxt

def perm_two_sample(rows,a,b,reps=100000,seed=26090315):
    vals=[rows[i]['k'] for i in a+b]; na=len(a); obs=sum(rows[i]['k'] for i in a)/len(a)-sum(rows[i]['k'] for i in b)/len(b); rng=random.Random(seed); lo=0
    for _ in range(reps):
        x=vals[:]; rng.shuffle(x); d=sum(x[:na])/na-sum(x[na:])/len(b)
        lo += d <= obs + 1e-15
    return {'observedMeanKDiff_first_minus_next':obs,'lowerTailP':(lo+1)/(reps+1),'reps':reps}

def slot_preserving_max_test(rows,cohorts,target_idx,reps=50000,seed=26090316):
    orig=[r['k'] for r in rows]; byslot=defaultdict(list)
    for r in rows:byslot[r['slot']].append(r['i'])
    target_obs=sum(le(orig[i]) for i in target_idx); max_obs=max(sum(le(orig[i]) for i in c['idx']) for c in cohorts)
    rng=random.Random(seed); ge_target=ge_max=0; maxvals=[]
    work=orig[:]
    for _ in range(reps):
        for ids in byslot.values():
            vals=[orig[i] for i in ids]; rng.shuffle(vals)
            for i,v in zip(ids,vals): work[i]=v
        tv=sum(le(work[i]) for i in target_idx); mv=max(sum(le(work[i]) for i in c['idx']) for c in cohorts)
        ge_target += tv >= target_obs-1e-12; ge_max += mv >= max_obs-1e-12; maxvals.append(mv)
    maxvals.sort()
    return {'targetObservedLogLR':target_obs,'observedMaxLogLRAcrossCohorts':max_obs,'targetSlotPreservingP':(ge_target+1)/(reps+1),'familywiseMaxP':(ge_max+1)/(reps+1),'nullMax95':[maxvals[int(.025*reps)],maxvals[int(.975*reps)]],'reps':reps,'null':'shuffle K within exact HH:MM slot; cohort positions fixed'}

def iid_lr_tail(n,thr):
    # exact multinomial enumeration for frozen rounded q/p0 at n=50; used only at n=50
    from math import factorial
    if n!=50:return None
    logs=[le(i) for i in range(5)]; prob=0.0
    for c0 in range(n+1):
      for c1 in range(n-c0+1):
       for c2 in range(n-c0-c1+1):
        for c3 in range(n-c0-c1-c2+1):
         c4=n-c0-c1-c2-c3; cs=[c0,c1,c2,c3,c4]
         if sum(cs[i]*logs[i] for i in range(5)) < thr-1e-12: continue
         coef=factorial(n)
         for c in cs: coef//=factorial(c)
         pr=float(coef)
         for i,c in enumerate(cs): pr*=P0[i]**c
         prob+=pr
    return prob

def main():
    raw,rows=load(); eras=stable_eras(rows); cohorts=cohort_indices(rows,eras); first,nxt=targeted_indices(rows)
    ranked=[]
    for c in cohorts:
        s=summary_idxs(rows,c['idx']); ranked.append({'start':c['start'],'addedSlots':c['addedSlots'],**s})
    ranked.sort(key=lambda x:x['logLR'],reverse=True)
    ts=summary_idxs(rows,first); ns=summary_idxs(rows,nxt)
    target_cohort=next((c for c in cohorts if c['start']==TARGET_DATE and set(c['addedSlots'])==TARGET_SLOTS),None)
    if target_cohort is None: raise RuntimeError('target launch cohort not found')
    mt=slot_preserving_max_test(rows,cohorts,target_cohort['idx'])
    single=iid_lr_tail(50,ts['logLR']); bon=min(1.0,single*len(cohorts)) if single is not None else None
    report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'count':len(rows),'last':rows[-1]['number'],'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},'frozenChanged':False,'target':{'date':TARGET_DATE,'slots':sorted(TARGET_SLOTS),'first50':ts,'next50':ns,'firstVsNext':perm_two_sample(rows,first,nxt),'singleCohortExactIIDTailP':single,'bonferroniAcrossObservedCohorts':bon},'cohortMultiplicity':{'count':len(cohorts),'ranked':ranked,'targetRank':1+sum(x['logLR']>ts['logLR']+1e-12 for x in ranked),'slotPreservingPermutation':mt},'interpretation':{'confirmatoryStatus':'unchanged; this file is historical exploratory validation only','manipulationClaim':False}}
    OUT.mkdir(parents=True,exist_ok=True); json.dump(report,open(OUT/'gsc-launch-multiplicity.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
