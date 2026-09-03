#!/usr/bin/env python3
import itertools, json, math, os, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
TARGET_DATE='2025-10-15'
SLOTS=('12:07','16:07','16:22','20:07')
REPS=50000
SEED=26090351

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def k_of(a,b): return len(set(a)&set(b))
def le(k): return math.log(Q[k]/P0[k])
def safeexp(x): return 0.0 if x < -745 else math.exp(x)

def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw
    src.sort(key=lambda x:int(x['number'])); rows=[]
    for i,d in enumerate(src):
        A=list(map(int,d.get('fieldA',d.get('field1')))); B=list(map(int,d.get('fieldB',d.get('field2'))))
        z=dt(d.get('date',d.get('draw_date'))).astimezone(TZ)
        rows.append({'i':i,'number':int(d['number']),'k':k_of(A,B),'slot':z.strftime('%H:%M'),'date':z.date().isoformat()})
    return raw,rows

def main():
    raw,rows=load(); orig=[r['k'] for r in rows]
    target_all=[r['i'] for r in rows if r['date']>=TARGET_DATE and r['slot'] in SLOTS][:50]
    if len(target_all)!=50: raise RuntimeError('target first50 not found')
    target_by_slot={s:[i for i in target_all if rows[i]['slot']==s] for s in SLOTS}
    subsets=[]
    for m in range(1,len(SLOTS)+1):
        for comb in itertools.combinations(SLOTS,m):
            ids=[i for i in target_all if rows[i]['slot'] in comb]
            obs=sum(le(orig[i]) for i in ids)
            subsets.append({'slots':comb,'ids':ids,'n':len(ids),'obs':obs})

    byslot=defaultdict(list)
    for r in rows:
        if r['slot'] in SLOTS: byslot[r['slot']].append(r['i'])
    rng=random.Random(SEED)
    sims=[[0.0]*REPS for _ in subsets]
    work=orig[:]
    for b in range(REPS):
        for s,ids in byslot.items():
            vals=[orig[i] for i in ids]; rng.shuffle(vals)
            for i,v in zip(ids,vals): work[i]=v
        for j,sub in enumerate(subsets):
            sims[j][b]=sum(le(work[i]) for i in sub['ids'])

    obs_ps=[]
    for j,sub in enumerate(subsets):
        ge=sum(x>=sub['obs']-1e-12 for x in sims[j])
        p=(ge+1)/(REPS+1)
        obs_ps.append(p); sub['rawP']=p
    obs_min=min(obs_ps)

    # Westfall-Young style minP using empirical upper-tail ranks within each subset's own null.
    sorted_sims=[sorted(v) for v in sims]
    import bisect
    ge_min=0
    minp_null=[]
    for b in range(REPS):
        mp=1.0
        for j in range(len(subsets)):
            arr=sorted_sims[j]; x=sims[j][b]
            left=bisect.bisect_left(arr,x-1e-12)
            p=(REPS-left+1)/(REPS+1)
            if p<mp: mp=p
        minp_null.append(mp)
        if mp<=obs_min+1e-15: ge_min+=1
    fwer=(ge_min+1)/(REPS+1)

    ranked=[]
    for sub in subsets:
        counts=[0]*5
        for i in sub['ids']: counts[orig[i]]+=1
        ranked.append({'slots':list(sub['slots']),'n':sub['n'],'slotCounts':{s:len(target_by_slot[s]) for s in sub['slots']},'countsK':counts,'meanK':sum(k*c for k,c in enumerate(counts))/sub['n'],'logLR':sub['obs'],'E':safeexp(sub['obs']),'rawSlotPreservingP':sub['rawP']})
    ranked.sort(key=lambda x:(x['rawSlotPreservingP'],-x['logLR']))
    report={
      'generatedAt':datetime.now().astimezone().isoformat(),
      'archive':{'count':len(rows),'last':rows[-1]['number'],'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},
      'frozenChanged':False,
      'target':{'date':TARGET_DATE,'firstDraw':rows[target_all[0]]['number'],'lastDraw':rows[target_all[-1]]['number'],'allSlots':list(SLOTS),'perSlotN':{s:len(target_by_slot[s]) for s in SLOTS}},
      'subsetSearch':{'numberOfNonemptySubsets':len(subsets),'reps':REPS,'null':'shuffle K within each exact HH:MM slot over the full observed history; fixed launch positions; search all 15 nonempty subsets','observedMinRawP':obs_min,'familywiseMinP':fwer,'ranked':ranked},
      'interpretation':{'confirmatoryStatus':'unchanged; historical exploratory robustness only','selectionWarning':'slot subsets are post-hoc, so familywiseMinP is the relevant quantity; raw subset p-values are not standalone evidence','manipulationClaim':False}
    }
    OUT.mkdir(parents=True,exist_ok=True)
    json.dump(report,open(OUT/'gsc-launch-subset-robustness.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
