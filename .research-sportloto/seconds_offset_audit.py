#!/usr/bin/env python3
import json, math, os, random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
REPS=200000
SEED=2609030617

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00')).astimezone(TZ)
def k_of(a,b): return len(set(a)&set(b))
def le(k): return math.log(Q[k]/P0[k])
def safeexp(x): return 0.0 if x < -745 else math.exp(x)

def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw
    src.sort(key=lambda x:int(x['number']))
    rows=[]
    for i,d in enumerate(src):
        A=list(map(int,d.get('fieldA',d.get('field1')))); B=list(map(int,d.get('fieldB',d.get('field2')))); z=dt(d.get('date',d.get('draw_date')))
        rows.append({'i':i,'number':int(d['number']),'dt':z,'date':z.date().isoformat(),'slot':z.strftime('%H:%M:%S'),'minute':z.strftime('%H:%M'),'second':z.second,'k':k_of(A,B)})
    return raw,rows

def summarize(ids,rows):
    c=Counter(rows[i]['k'] for i in ids); n=len(ids); counts=[c[k] for k in range(5)]
    ll=sum(le(rows[i]['k']) for i in ids)
    return {'n':n,'countsK':counts,'sumK':sum(k*counts[k] for k in range(5)),'meanK':sum(k*counts[k] for k in range(5))/n if n else None,'logLR':ll,'E':safeexp(ll),'firstDraw':rows[ids[0]]['number'] if ids else None,'lastDraw':rows[ids[-1]]['number'] if ids else None}

def exact_sum_tail(n,s):
    dist=[1.0]
    for _ in range(n):
        nd=[0.0]*(len(dist)+4)
        for a,pa in enumerate(dist):
            for k,pk in enumerate(P0): nd[a+k]+=pa*pk
        dist=nd
    return sum(dist[:s+1])

def daily_new_slots(rows,date):
    dates=sorted(set(r['date'] for r in rows))
    j=dates.index(date); prev=dates[j-1]
    cur={r['slot'] for r in rows if r['date']==date}; old={r['slot'] for r in rows if r['date']==prev}
    return prev,sorted(cur-old)

def first_n_in_slots(rows,start_date,slots,n):
    return [r['i'] for r in rows if r['date']>=start_date and r['slot'] in set(slots)][:n]

def main():
    raw,rows=load()
    launch='2025-10-15'
    prev,newslots=daily_new_slots(rows,launch)
    expected={'12:07:30','16:07:00','16:22:30','20:07:30'}
    launchslots=sorted(expected & set(newslots))
    if set(launchslots)!=expected: raise RuntimeError(f'launch slots mismatch: {launchslots}, daily new={newslots}')
    first50=first_n_in_slots(rows,launch,launchslots,50)
    sec30=[i for i in first50 if rows[i]['second']==30]; sec00=[i for i in first50 if rows[i]['second']==0]
    s30=summarize(sec30,rows); s00=summarize(sec00,rows)
    s30['iidLowerTailSumK']=exact_sum_tail(s30['n'],s30['sumK'])

    # Externally-defined seconds grouping, calibrated by shuffling K within each exact slot over all observed history.
    pools=defaultdict(list)
    for r in rows: pools[r['slot']].append(r['k'])
    need=Counter(rows[i]['slot'] for i in sec30)
    obs=s30['logLR']; rng=random.Random(SEED); ge=0
    for _ in range(REPS):
        sim=0.0
        for slot,m in need.items():
            vals=rng.sample(pools[slot],m)
            sim += sum(le(k) for k in vals)
        if sim >= obs-1e-12: ge += 1
    s30['slotPreservingP']=(ge+1)/(REPS+1)
    s30['permutationReps']=REPS

    # Same externally defined :30 slots, immediately following matched quota after the launch observations.
    last_i=max(first50); after=[r['i'] for r in rows if r['i']>last_i and r['slot'] in need]
    quotas=dict(need); matched=[]; used=Counter()
    for i in after:
        s=rows[i]['slot']
        if used[s]<quotas[s]: matched.append(i); used[s]+=1
        if all(used[s]>=m for s,m in quotas.items()): break
    nextMatched=summarize(matched,rows)

    # Long-run same-slot controls after the launch 50, separating :30 and :00.
    afterLaunch=[r['i'] for r in rows if r['i']>last_i and r['slot'] in set(launchslots)]
    later30=[i for i in afterLaunch if rows[i]['second']==30]; later00=[i for i in afterLaunch if rows[i]['second']==0]
    later30s=summarize(later30,rows); later00s=summarize(later00,rows)

    # Independent schedule transition 2026-08-20: identify slots newly present vs previous date, take first 50 occurrences, and partition by second offset.
    d2='2026-08-20'; prev2,new2=daily_new_slots(rows,d2); cohort2=first_n_in_slots(rows,d2,new2,50)
    c230=[i for i in cohort2 if rows[i]['second']==30]; c200=[i for i in cohort2 if rows[i]['second']==0]

    report={
      'generatedAt':datetime.now().astimezone().isoformat(),
      'archive':{'count':len(rows),'last':rows[-1]['number'],'lastDate':rows[-1]['dt'].isoformat(),'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},
      'frozenChanged':False,
      'launch':{
        'date':launch,'previousDate':prev,'dailyNewSlots':newslots,'targetSlots':launchslots,
        'first50':summarize(first50,rows),
        'seconds30Slots':sorted({rows[i]['slot'] for i in sec30}),'seconds30':s30,
        'seconds00Slots':sorted({rows[i]['slot'] for i in sec00}),'seconds00':s00,
        'nextMatchedSeconds30':nextMatched,
        'laterSameSlotsSeconds30':later30s,'laterSameSlotsSeconds00':later00s
      },
      'independentScheduleTransition':{
        'date':d2,'previousDate':prev2,'newSlots':new2,'first50AllNew':summarize(cohort2,rows),
        'seconds30':summarize(c230,rows),'seconds00':summarize(c200,rows)
      },
      'interpretation':{
        'status':'historical exploratory; timestamp-second grouping was not preregistered',
        'mechanisticLead':'At the 2025-10-15 launch, the three externally identifiable :30-second slots are exactly the three low-overlap slots; test whether this survives same-slot and independent-transition controls.',
        'manipulationClaim':False
      }
    }
    OUT.mkdir(parents=True,exist_ok=True)
    json.dump(report,open(OUT/'seconds-offset-audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
