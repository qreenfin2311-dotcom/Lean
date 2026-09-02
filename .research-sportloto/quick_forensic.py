#!/usr/bin/env python3
import json, math, os, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/quick-out'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
SLOTS={'12:07','13:52','16:07','16:22','20:07','23:22'}
LAUNCH={'12:07','16:07','16:22','20:07'}
FREEZE=datetime.fromisoformat('2026-08-26T14:21:00+03:00')

def dt(x):return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def kval(a,b):return len(set(a)&set(b))
def loge(k):return math.log(Q[k]/P0[k])
def exp_safe(x):return 0.0 if x<-745 else math.exp(x) if x<700 else float('inf')
def summ(rs):
 c=[0]*5
 for r in rs:c[r['k']]+=1
 lr=sum(c[i]*loge(i) for i in range(5)); sk=sum(i*c[i] for i in range(5));n=len(rs)
 return {'n':n,'counts':c,'sumK':sk,'meanK':sk/n if n else None,'logLR':lr,'E':exp_safe(lr),'first':rs[0]['number'] if n else None,'last':rs[-1]['number'] if n else None}

def load():
 raw=json.load(open(DATA,encoding='utf-8'));src=raw['draws'] if isinstance(raw,dict) else raw;src.sort(key=lambda d:int(d['number']));rows=[]
 for d in src:
  A=list(map(int,d.get('fieldA',d.get('field1'))));B=list(map(int,d.get('fieldB',d.get('field2'))));z=dt(d.get('date',d.get('draw_date'))).astimezone(FREEZE.tzinfo)
  rows.append({'number':int(d['number']),'A':A,'B':B,'k':kval(A,B),'slot':z.strftime('%H:%M'),'date':z.date().isoformat(),'time':z})
 return raw,rows

def residual(rs):
 n=len(rs);a=[0]*20;b=[0]*20;o=[[0]*20 for _ in range(20)]
 for r in rs:
  for x in r['A']:a[x-1]+=1
  for y in r['B']:b[y-1]+=1
  for x in r['A']:
   for y in r['B']:o[x-1][y-1]+=1
 R=[[0.0]*20 for _ in range(20)];D=[];od=ed=0.0
 for i in range(20):
  e=a[i]*b[i]/n;od+=o[i][i];ed+=e;D.append((o[i][i]-e)/math.sqrt(e) if e else 0.0)
  for j in range(20):
   ee=a[i]*b[j]/n;R[i][j]=(o[i][j]-ee)/math.sqrt(ee) if ee else 0.0
 return {'D':D,'R':R,'diagObserved':od,'diagExpected':ed,'diagDeficit':od-ed}
def flat(M):return [v for row in M for v in row]
def cos(a,b):
 d=sum(x*y for x,y in zip(a,b));aa=sum(x*x for x in a);bb=sum(y*y for y in b);return d/math.sqrt(aa*bb) if aa*bb else 0.0

def identity_perm(a,b,reps=30000,seed=1):
 od=cos(a['D'],b['D']);om=cos(flat(a['R']),flat(b['R']));rng=random.Random(seed);gd=gm=0;ds=[];ms=[]
 for _ in range(reps):
  p=list(range(20));rng.shuffle(p);D=[b['D'][p[i]] for i in range(20)];R=[b['R'][p[i]][p[j]] for i in range(20) for j in range(20)];cd=cos(a['D'],D);cm=cos(flat(a['R']),R);gd+=cd>=od;gm+=cm>=om;ds.append(cd);ms.append(cm)
 ds.sort();ms.sort();return {'diagCos':od,'diagUpperP':(gd+1)/(reps+1),'diagNull95':[ds[int(.025*reps)],ds[int(.975*reps)]],'matrixCos':om,'matrixUpperP':(gm+1)/(reps+1),'matrixNull95':[ms[int(.025*reps)],ms[int(.975*reps)]],'reps':reps}

def first_by_slot(rows,start,slots,total=50):return [r for r in rows if r['date']>=start and r['slot'] in slots][:total]
def slot_breakdown(first,rows):
 out={}
 for s in sorted({r['slot'] for r in first}):
  a=[r for r in first if r['slot']==s]; later=[r for r in rows if r['slot']==s and r['number']>first[-1]['number']][:max(50,len(a))]
  out[s]={'launchPart':summ(a),'laterControl':summ(later)}
 return out

def strat_boot(first,rows,reps=200000,seed=9759):
 # empirical same-slot null, excluding target launch observations; sample independently from later same-slot histories
 pools={}
 targetnums={r['number'] for r in first}
 for s in sorted({r['slot'] for r in first}):pools[s]=[r['k'] for r in rows if r['slot']==s and r['number'] not in targetnums and r['number']>first[-1]['number']]
 if any(not v for v in pools.values()):return None
 obs=sum(r['k'] for r in first);rng=random.Random(seed);leq=0;lrge=0;obslr=sum(loge(r['k']) for r in first)
 for _ in range(reps):
  sk=0;lr=0.0
  for r in first:
   x=rng.choice(pools[r['slot']]);sk+=x;lr+=loge(x)
  leq+=sk<=obs;lrge+=lr>=obslr-1e-12
 return {'observedSumK':obs,'observedLogLR':obslr,'sameSlotLaterBootstrapP_sumK_lower':(leq+1)/(reps+1),'sameSlotLaterBootstrapP_logLR_upper':(lrge+1)/(reps+1),'reps':reps,'poolSizes':{s:len(v) for s,v in pools.items()}}

def cp_scan(seq,minseg=20):
 # two-segment Bernoulli-free categorical multinomial MLE vs one distribution; choose max LR cp, calibrate permutation
 def ll(seg):
  n=len(seg);c=[0]*5
  for x in seg:c[x]+=1
  return sum(v*math.log(v/n) for v in c if v)
 base=ll(seq);best=(-1,None)
 for t in range(minseg,len(seq)-minseg+1):
  z=2*(ll(seq[:t])+ll(seq[t:])-base)
  if z>best[0]:best=(z,t)
 return best

def cp_perm(seq,reps=20000,seed=11618):
 obs,t=cp_scan(seq);rng=random.Random(seed);ge=0
 for _ in range(reps):
  z=seq[:];rng.shuffle(z);mx,_=cp_scan(z);ge+=mx>=obs-1e-12
 return {'best2logLR':obs,'bestChangeAfterObservation':t,'permutationP':(ge+1)/(reps+1),'reps':reps}

def main():
 raw,rows=load();train=[r for r in rows if 9759<=r['number']<=9808];launch=first_by_slot(rows,'2025-10-15',LAUNCH,50);launch100=first_by_slot(rows,'2025-10-15',LAUNCH,100);nxt=launch100[50:100]
 ft,fl,fn=residual(train),residual(launch),residual(nxt)
 ledger=[];lr=0
 for r in rows:
  if r['time']<=FREEZE or r['slot'] not in SLOTS:continue
  e=Q[r['k']]/P0[r['k']];lr+=math.log(e);ledger.append({'draw_id':r['number'],'time':r['time'].isoformat(),'slot':r['slot'],'K':r['k'],'e_t':e,'cumE':exp_safe(lr),'cumLogLR':lr})
 report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'count':len(rows),'first':rows[0]['number'],'last':rows[-1]['number'],'lastDate':rows[-1]['time'].isoformat(),'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},'frozen':{'changed':False,'n':len(ledger),'cumLogLR':lr,'E':exp_safe(lr),'rows':ledger},'scalar':{'train':summ(train),'launchFirst50':summ(launch),'launchNext50':summ(nxt),'slotBreakdown':slot_breakdown(launch,rows),'sameSlotLaterBootstrap':strat_boot(launch,rows),'launch100ChangePoint':cp_perm([r['k'] for r in launch100])},'fingerprint':{'train':{'diagObserved':ft['diagObserved'],'diagExpected':ft['diagExpected'],'diagDeficit':ft['diagDeficit']},'launch':{'diagObserved':fl['diagObserved'],'diagExpected':fl['diagExpected'],'diagDeficit':fl['diagDeficit']},'next':{'diagObserved':fn['diagObserved'],'diagExpected':fn['diagExpected'],'diagDeficit':fn['diagDeficit']},'trainVsLaunch':identity_perm(ft,fl,30000,1),'trainVsNext':identity_perm(ft,fn,30000,2),'launchVsNext':identity_perm(fl,fn,30000,3)},'manipulationClaim':False}
 OUT.mkdir(parents=True,exist_ok=True);json.dump(report,open(OUT/'quick-forensic.json','w'),ensure_ascii=False,indent=2,default=str);print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
