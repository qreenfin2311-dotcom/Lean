#!/usr/bin/env python3
import json, math, os, random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out-seconds'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
FROZEN_MINUTES=('12:07','13:52','16:07','16:22','20:07','23:22')
LAUNCH_DATE='2025-10-15'
LAUNCH_MINUTES=('12:07','16:07','16:22','20:07')
REPS=100000
SEED=2609030631


def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00')).astimezone(TZ)
def k_of(a,b): return len(set(a)&set(b))
def le(k): return math.log(Q[k]/P0[k])
def safeexp(x): return 0.0 if x < -745 else (float('inf') if x > 709 else math.exp(x))

def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw
    src.sort(key=lambda x:int(x['number']))
    rows=[]
    for i,d in enumerate(src):
        A=list(map(int,d.get('fieldA',d.get('field1')))); B=list(map(int,d.get('fieldB',d.get('field2')))); z=dt(d.get('date',d.get('draw_date')))
        rows.append({'i':i,'number':int(d['number']),'dt':z,'date':z.date().isoformat(),'minute':z.strftime('%H:%M'),'second':z.second,'slot':z.strftime('%H:%M:%S'),'k':k_of(A,B)})
    return raw,rows

def summary_ids(ids,rows,vals=None):
    ks=[(vals[i] if vals is not None else rows[i]['k']) for i in ids]; c=Counter(ks); n=len(ks)
    ll=sum(le(x) for x in ks)
    return {'n':n,'countsK':[c[k] for k in range(5)],'sumK':sum(ks),'meanK':sum(ks)/n if n else None,'logLR':ll,'E':safeexp(ll),'firstDraw':rows[ids[0]]['number'] if ids else None,'lastDraw':rows[ids[-1]]['number'] if ids else None,'firstDateTime':rows[ids[0]]['dt'].isoformat() if ids else None,'lastDateTime':rows[ids[-1]]['dt'].isoformat() if ids else None}

def compress_second_epochs(ids,rows):
    epochs=[]
    for i in ids:
        s=rows[i]['second']
        if epochs and epochs[-1]['second']==s:
            epochs[-1]['ids'].append(i)
        else:
            epochs.append({'second':s,'ids':[i]})
    return epochs

def event_record(minute,a,b,rows,w=50):
    pre=a['ids'][-min(w,len(a['ids'])):]; post=b['ids'][:min(w,len(b['ids']))]
    sa=summary_ids(pre,rows); sb=summary_ids(post,rows)
    return {
      'minute':minute,'fromSecond':a['second'],'toSecond':b['second'],
      'transitionAfterDraw':rows[a['ids'][-1]]['number'],'transitionBeforeDraw':rows[b['ids'][0]]['number'],
      'lastOldDateTime':rows[a['ids'][-1]]['dt'].isoformat(),'firstNewDateTime':rows[b['ids'][0]]['dt'].isoformat(),
      'oldEpochN':len(a['ids']),'newEpochN':len(b['ids']),
      'preWindow':sa,'postWindow':sb,
      'meanKDiffPostMinusPre':sb['meanK']-sa['meanK'],
      'absMeanKDiff':abs(sb['meanK']-sa['meanK']),
      'preIds':pre,'postIds':post
    }

def main():
    raw,rows=load(); byminute=defaultdict(list)
    for r in rows:
        if r['minute'] in FROZEN_MINUTES: byminute[r['minute']].append(r['i'])

    minute_epochs={}; events=[]
    for minute in FROZEN_MINUTES:
        eps=compress_second_epochs(byminute[minute],rows); minute_epochs[minute]=[]
        for ep in eps:
            s=summary_ids(ep['ids'],rows); minute_epochs[minute].append({'second':ep['second'],**s})
        for a,b in zip(eps,eps[1:]):
            if len(a['ids'])>=10 and len(b['ids'])>=10:
                events.append(event_record(minute,a,b,rows,50))

    # Event-specific two-sided permutation and familywise max over all observed exact-second transitions.
    rng=random.Random(SEED)
    ge_event=[0]*len(events); ge_max=[0]*len(events)
    obs=[e['absMeanKDiff'] for e in events]
    work=[r['k'] for r in rows]
    pools={m:byminute[m][:] for m in FROZEN_MINUTES}
    for _ in range(REPS):
        for m,ids in pools.items():
            vals=[rows[i]['k'] for i in ids]; rng.shuffle(vals)
            for i,v in zip(ids,vals): work[i]=v
        stats=[]
        for e in events:
            pre=e['preIds']; post=e['postIds']
            d=abs(sum(work[i] for i in post)/len(post)-sum(work[i] for i in pre)/len(pre))
            stats.append(d)
        mx=max(stats) if stats else 0.0
        for j,d in enumerate(stats):
            if d>=obs[j]-1e-12: ge_event[j]+=1
            if mx>=obs[j]-1e-12: ge_max[j]+=1
    for j,e in enumerate(events):
        e['rawWithinMinutePermutationP']=(ge_event[j]+1)/(REPS+1)
        e['familywiseMaxP']=(ge_max[j]+1)/(REPS+1)
        for key in ('preIds','postIds'): e.pop(key,None)

    # Launch: verify the exact second offsets actually present on 2025-10-15 and first 50 launch observations.
    daily_launch=[r for r in rows if r['date']==LAUNCH_DATE and r['minute'] in LAUNCH_MINUTES]
    launch_slots=sorted({r['slot'] for r in daily_launch})
    launch_ids=[r['i'] for r in rows if r['date']>=LAUNCH_DATE and r['minute'] in LAUNCH_MINUTES][:50]
    launch_by_min={m:summary_ids([i for i in launch_ids if rows[i]['minute']==m],rows) for m in LAUNCH_MINUTES}
    launch_by_min_seconds={m:sorted({rows[i]['second'] for i in launch_ids if rows[i]['minute']==m}) for m in LAUNCH_MINUTES}

    # Identify the first 16:07 exact-second transition after launch and relate it to the launch transient.
    e1607=[e for e in events if e['minute']=='16:07' and e['firstNewDateTime']>=LAUNCH_DATE]
    first1607=e1607[0] if e1607 else None
    if first1607:
        cut=first1607['transitionBeforeDraw']
        before_shift=[r['i'] for r in rows if r['date']>=LAUNCH_DATE and r['minute']=='16:07' and r['number']<cut]
        first1607['all16_07FromLaunchBeforeShift']=summary_ids(before_shift,rows)

    # For completeness, summarize exact-second values by frozen minute over full pre-freeze history.
    second_counts={m:{str(s):sum(1 for i in byminute[m] if rows[i]['second']==s) for s in sorted({rows[i]['second'] for i in byminute[m]})} for m in FROZEN_MINUTES}

    report={
      'generatedAt':datetime.now().astimezone().isoformat(),
      'archive':{'count':len(rows),'last':rows[-1]['number'],'lastDateTime':rows[-1]['dt'].isoformat(),'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},
      'frozenChanged':False,
      'frozenMinuteSlots':list(FROZEN_MINUTES),
      'exactSecondCounts':second_counts,
      'exactSecondEpochs':minute_epochs,
      'transitionEvents':events,
      'launch2025_10_15':{
        'exactSlotsOnLaunchDate':launch_slots,
        'first50':summary_ids(launch_ids,rows),
        'perMinute':launch_by_min,
        'secondsSeenWithinFirst50ByMinute':launch_by_min_seconds,
        'allFourLaunchMinutesShareSameSecondOffset':len({rows[i]['second'] for i in launch_ids})==1,
        'first16_07SecondTransitionAfterLaunch':first1607
      },
      'interpretation':{
        'confirmatoryStatus':'unchanged; exact seconds are exploratory and frozen inclusion remains minute-level exactly as preregistered',
        'keyQuestion':'Does an externally observable HH:MM:SS scheduler change coincide with a reproducible K regime change?',
        'multiplicity':'familywiseMaxP controls over all exact-second transitions with >=10 observations on each side among the six frozen minute slots',
        'manipulationClaim':False
      }
    }
    OUT.mkdir(parents=True,exist_ok=True)
    json.dump(report,open(OUT/'seconds-offset-audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
