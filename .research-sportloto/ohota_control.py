#!/usr/bin/env python3
import json, math, os, time, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path

ENDPOINT='https://www.stoloto.ru/p/api/mobile/api/v35/service/draws/archive'
HEADERS={
 'accept':'*/*','content-type':'application/x-www-form-urlencoded','device-platform':'DESKTOP','device-type':'STOLOTO',
 'gosloto-partner':'bXMjXFRXZ3coWXh6R3s1NTdUX3dnWlBMLUxmdg','referer':'https://www.stoloto.ru/oxota/archive',
 'user-agent':'Mozilla/5.0 LotoOS-GitHub-Ohota-Control/1.0'}
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out')); OUT.mkdir(parents=True,exist_ok=True)

def fetch(game,page,count=200):
    qs=urllib.parse.urlencode({'game':game,'count':count,'page':page})
    req=urllib.request.Request(ENDPOINT+'?'+qs,headers=HEADERS)
    for a in range(4):
        try:
            with urllib.request.urlopen(req,timeout=20) as r: return json.load(r)
        except Exception:
            if a==3: raise
            time.sleep(a+1)

def norm(d):
    if d.get('status')!='COMPLETED' or d.get('completed') is False: return None
    s=((d.get('combination') or {}).get('structured'))
    if not isinstance(s,list) or len(s)!=8: return None
    z=[int(x) for x in s]; a=z[:4]; b=z[4:]
    if any(len(set(f))!=4 or any(x<1 or x>20 for x in f) for f in (a,b)): return None
    return {'number':int(d['number']),'date':d.get('date'),'A':a,'B':b,'K':len(set(a)&set(b))}

# Probe likely slugs. The first one returning valid 8-number completed draws wins.
game=None; probe={}
for cand in ['oxota','ohota','hunt','hunting']:
    try:
        x=fetch(cand,1,50); valid=[norm(d) for d in x.get('draws',[])]; valid=[r for r in valid if r]
        probe[cand]={'requestStatus':x.get('requestStatus'),'rawN':len(x.get('draws',[])),'validN':len(valid),'firstValid':valid[0] if valid else None}
        if x.get('requestStatus')=='success' and valid:
            game=cand; break
    except Exception as e: probe[cand]={'error':str(e)[:200]}
if not game:
    json.dump({'probe':probe},open(OUT/'ohota-control.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    raise SystemExit('No valid Ohota API game slug found')

# Page through the official archive. Deduplicate and reject conflicts.
by={}; conflicts=[]; page=1; pages=0
while page<=400:
    x=fetch(game,page,200); raw=x.get('draws',[])
    if x.get('requestStatus')!='success' or not isinstance(raw,list): raise SystemExit('schema mismatch')
    if not raw: break
    for d in raw:
        r=norm(d)
        if not r: continue
        old=by.get(r['number'])
        if old and (old['date'],old['A'],old['B'])!=(r['date'],r['A'],r['B']): conflicts.append(r['number'])
        by[r['number']]=r
    pages+=1
    if len(raw)<200: break
    page+=1
if conflicts: raise SystemExit(f'conflicts {conflicts[:10]}')
rows=[by[k] for k in sorted(by)]
if len(rows)<1000: raise SystemExit(f'too few rows {len(rows)}')

c=[0]*5
for r in rows:c[r['K']]+=1
n=len(rows); mean=sum(k*c[k] for k in range(5))/n
lr=sum(c[k]*math.log(Q[k]/P0[k]) for k in range(5))
# multinomial Pearson and G2; chi-square df4 p via regularized gamma integer/half generic fallback using scipy unavailable.
pear=sum((c[k]-n*P0[k])**2/(n*P0[k]) for k in range(5))
g2=2*sum(c[k]*math.log(c[k]/(n*P0[k])) for k in range(5) if c[k])
# mean K variance under exact IID baseline
mu=sum(k*P0[k] for k in range(5)); var=sum((k-mu)**2*P0[k] for k in range(5)); z=(mean-mu)/math.sqrt(var/n)
# Brier mean difference q-IID
bd=0.0
for k in range(5):
    bq=sum((Q[j]-(1 if j==k else 0))**2 for j in range(5)); b0=sum((P0[j]-(1 if j==k else 0))**2 for j in range(5))
    bd+=c[k]*(bq-b0)
bd/=n
# split chronologically into 4 equal quartiles to test reproducibility
quart=[]
for j in range(4):
    a=j*n//4;b=(j+1)*n//4; rr=rows[a:b];cc=Counter(r['K'] for r in rr); nn=len(rr)
    ll=sum(cc[k]*math.log(Q[k]/P0[k]) for k in range(5)); mm=sum(k*cc[k] for k in range(5))/nn
    quart.append({'part':j+1,'n':nn,'first':rr[0]['number'],'last':rr[-1]['number'],'counts':[cc[k] for k in range(5)],'meanK':mm,'fixedQ_logLR':ll,'fixedQ_E':math.exp(ll) if ll>-745 else 0.0})
rep={
 'source':'official Stoloto archive API','gameSlug':game,'probe':probe,'pages':pages,'n':n,
 'first':rows[0]['number'],'firstDate':rows[0]['date'],'last':rows[-1]['number'],'lastDate':rows[-1]['date'],
 'counts':c,'meanK':mean,'iidMean':mu,'meanZ':z,'pearson':pear,'g2':g2,
 'frozenQ_logLR_vs_IID':lr,'frozenQ_E':math.exp(lr) if lr>-745 else 0.0,'meanBrierDiff_q_minus_IID':bd,
 'quartiles':quart,
 'interpretation':'Independent same-formula RNG control only; does not prove the same physical RNG unit as Sportloto 4x20.'}
json.dump(rep,open(OUT/'ohota-control.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
with open(OUT/'ohota-k.csv','w',encoding='utf-8') as f:
    f.write('draw_id,date,K\n')
    for r in rows:f.write(f"{r['number']},{r['date']},{r['K']}\n")
print(json.dumps(rep,ensure_ascii=False))
