#!/usr/bin/env python3
import csv, json, math, os
from datetime import datetime
from pathlib import Path

BASE = Path(os.getenv('LOTO_ARCHIVE', '.research-sportloto/draws.json'))
HEAD = Path(os.getenv('LOTO_HEAD', '.research-sportloto/official-head-raw.json'))
OUT = Path(os.getenv('LOTO_OUT', '.research-sportloto/out'))
P0 = [0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q = [0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
GAMMA = -0.5933185
SLOTS = {'12:07','13:52','16:07','16:22','20:07','23:22'}
FREEZE = datetime.fromisoformat('2026-08-26T14:21:00+03:00')

def parse_dt(x):
    return datetime.fromisoformat(str(x).replace('+0300','+03:00').replace('Z','+00:00'))

def normalize_raw(draw):
    structured = ((draw or {}).get('combination') or {}).get('structured')
    if not isinstance(structured, list) or len(structured) != 8:
        return None
    vals = [int(x) for x in structured]
    a, b = vals[:4], vals[4:]
    if any(len(set(f)) != 4 or any(x < 1 or x > 20 for x in f) for f in (a,b)):
        return None
    if draw.get('status') != 'COMPLETED' or draw.get('completed') is False:
        return None
    n = int(draw.get('number'))
    return {'number': n, 'date': draw.get('date'), 'fieldA': a, 'fieldB': b, 'source': 'official-live-head'}

def brier(p,k):
    return sum((p[i] - (1.0 if i == k else 0.0))**2 for i in range(5))

base = json.load(open(BASE, encoding='utf-8'))
raw = json.load(open(HEAD, encoding='utf-8'))
if raw.get('requestStatus') != 'success' or not isinstance(raw.get('draws'), list):
    raise SystemExit('official head schema mismatch')
head_rows = [x for x in (normalize_raw(d) for d in raw['draws']) if x]
if not head_rows:
    raise SystemExit('official head returned no completed 4x20 draws')

by = {}
for d in base.get('draws', []):
    by[int(d['number'])] = {'number': int(d['number']), 'date': d['date'], 'fieldA': list(map(int,d['fieldA'])), 'fieldB': list(map(int,d['fieldB'])), 'source': 'verified-snapshot'}
base_last = int(base.get('last') or max(by))
conflicts = []
for d in head_rows:
    old = by.get(d['number'])
    if old and (old['date'] != d['date'] or old['fieldA'] != d['fieldA'] or old['fieldB'] != d['fieldB']):
        conflicts.append(d['number'])
    by[d['number']] = d
if conflicts:
    raise SystemExit(f'official live head conflicts with verified snapshot: {conflicts[:10]}')

rows = [by[n] for n in sorted(by)]
head_max = max(x['number'] for x in head_rows)
new_rows = [x for x in rows if x['number'] > base_last]
ledger=[]; lr=0.0; bq=0.0; b0=0.0
for r in rows:
    t=parse_dt(r['date'])
    slot=t.strftime('%H:%M')
    if t <= FREEZE or slot not in SLOTS:
        continue
    k=len(set(r['fieldA']) & set(r['fieldB']))
    e=Q[k]/P0[k]
    lr += math.log(e)
    sq=brier(Q,k); s0=brier(P0,k); bq += sq; b0 += s0
    ledger.append({
        'draw_id':r['number'],
        'time_msk':t.astimezone(FREEZE.tzinfo).isoformat(),
        'slot':slot,
        'K':k,
        'e_t':e,
        'cumulative_logLR':lr,
        'cumulative_E':math.exp(lr) if -745 < lr < 700 else (0.0 if lr <= -745 else '>=1e304'),
        'cumulative_delta_logloss':lr,
        'Brier_q':sq,
        'Brier_IID':s0,
        'Brier_difference':sq-s0,
        'source':r['source'],
    })

OUT.mkdir(parents=True, exist_ok=True)
summary={
    'generatedAt':datetime.now().astimezone().isoformat(),
    'snapshot':{'retrievedAt':base.get('retrievedAt'),'last':base_last,'count':base.get('count')},
    'officialHead':{'completedCountOnPage':len(head_rows),'minDraw':min(x['number'] for x in head_rows),'maxDraw':head_max,'maxDate':max(head_rows,key=lambda x:x['number'])['date']},
    'newSinceSnapshot':[{'draw_id':x['number'],'date':x['date'],'slot':parse_dt(x['date']).strftime('%H:%M'),'K':len(set(x['fieldA'])&set(x['fieldB']))} for x in new_rows],
    'frozen':{'freeze':FREEZE.isoformat(),'slots':sorted(SLOTS),'q':Q,'gamma':GAMMA,'p0':P0,'n':len(ledger),'cumulativeLogLR':lr,'cumulativeE':math.exp(lr) if -745 < lr < 700 else (0.0 if lr <= -745 else '>=1e304'),'cumulativeDeltaLogloss':lr,'BrierQ':bq,'BrierIID':b0,'BrierDifference':bq-b0,'rows':ledger},
}
json.dump(summary, open(OUT/'live-prospective.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
with open(OUT/'live-prospective-ledger.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['draw_id','time_msk','slot','K','e_t','cumulative_logLR','cumulative_E','cumulative_delta_logloss','Brier_q','Brier_IID','Brier_difference','source'])
    w.writeheader(); w.writerows(ledger)
print(json.dumps({'snapshotLast':base_last,'officialHeadMax':head_max,'newSinceSnapshot':summary['newSinceSnapshot'],'frozenN':len(ledger),'E':summary['frozen']['cumulativeE']},ensure_ascii=False))
