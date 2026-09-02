#!/usr/bin/env python3
import json, math, os, random
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
LAUNCH_SLOTS={'12:07','16:07','16:22','20:07'}

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw; src.sort(key=lambda x:int(x['number'])); rows=[]
    for d in src:
        A=list(map(int,d.get('fieldA',d.get('field1')))); B=list(map(int,d.get('fieldB',d.get('field2')))); z=dt(d.get('date',d.get('draw_date'))).astimezone(TZ)
        rows.append({'number':int(d['number']),'A':A,'B':B,'slot':z.strftime('%H:%M'),'date':z.date().isoformat()})
    return raw,rows

def residual(block):
    n=len(block); a=[0]*20;b=[0]*20;o=[[0]*20 for _ in range(20)]
    for r in block:
        for x in r['A']:a[x-1]+=1
        for y in r['B']:b[y-1]+=1
        for x in r['A']:
            for y in r['B']:o[x-1][y-1]+=1
    R=[[0.0]*20 for _ in range(20)]; D=[]; obsdiag=expdiag=0.0
    for i in range(20):
        e=a[i]*b[i]/n; obsdiag+=o[i][i];expdiag+=e; D.append((o[i][i]-e)/math.sqrt(e) if e else 0.0)
        for j in range(20):
            ee=a[i]*b[j]/n;R[i][j]=(o[i][j]-ee)/math.sqrt(ee) if ee else 0.0
    return {'R':R,'D':D,'diagObserved':obsdiag,'diagExpectedFromMarginals':expdiag,'diagDeficit':obsdiag-expdiag,'margA':a,'margB':b}

def cosine_vec(x,y):
    d=sum(a*b for a,b in zip(x,y)); nx=sum(a*a for a in x);ny=sum(b*b for b in y)
    return d/math.sqrt(nx*ny) if nx*ny else 0.0

def flat(M):return [v for row in M for v in row]

def permute_fingerprint(fp,p):
    D=[fp['D'][p[i]] for i in range(20)]
    R=[[fp['R'][p[i]][p[j]] for j in range(20)] for i in range(20)]
    return D,R

def identity_test(a,b,reps=100000,seed=26090317):
    od=cosine_vec(a['D'],b['D']); om=cosine_vec(flat(a['R']),flat(b['R'])); rng=random.Random(seed); gd=gm=0; valsd=[];valsm=[]
    for _ in range(reps):
        p=list(range(20));rng.shuffle(p);D,R=permute_fingerprint(b,p);cd=cosine_vec(a['D'],D);cm=cosine_vec(flat(a['R']),flat(R));gd+=cd>=od;gm+=cm>=om;valsd.append(cd);valsm.append(cm)
    valsd.sort();valsm.sort()
    return {'diagIdentityCosine':od,'diagIdentityUpperP':(gd+1)/(reps+1),'diagNull95':[valsd[int(.025*reps)],valsd[int(.975*reps)]],'matrixIdentityCosine':om,'matrixIdentityUpperP':(gm+1)/(reps+1),'matrixNull95':[valsm[int(.025*reps)],valsm[int(.975*reps)]],'reps':reps,'null':'same permutation of number labels applied to both axes of launch fingerprint'}

def diag_details(fp):
    return [{'number':i+1,'z':fp['D'][i],'fieldA_count':fp['margA'][i],'fieldB_count':fp['margB'][i]} for i in range(20)]

def main():
    raw,rows=load();train=[r for r in rows if 9759<=r['number']<=9808];launch=[r for r in rows if r['date']>='2025-10-15' and r['slot'] in LAUNCH_SLOTS][:50];next50=[r for r in rows if r['date']>='2025-10-15' and r['slot'] in LAUNCH_SLOTS][50:100]
    if len(train)!=50 or len(launch)!=50 or len(next50)!=50:raise RuntimeError('expected 50-row blocks')
    ft=residual(train);fl=residual(launch);fn=residual(next50)
    report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'count':len(rows),'last':rows[-1]['number'],'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},'blocks':{'training9759_9808':{'first':train[0]['number'],'last':train[-1]['number'],'diagObserved':ft['diagObserved'],'diagExpectedFromMarginals':ft['diagExpectedFromMarginals'],'diagDeficit':ft['diagDeficit'],'diagByNumber':diag_details(ft)},'gscLaunchFirst50':{'first':launch[0]['number'],'last':launch[-1]['number'],'diagObserved':fl['diagObserved'],'diagExpectedFromMarginals':fl['diagExpectedFromMarginals'],'diagDeficit':fl['diagDeficit'],'diagByNumber':diag_details(fl)},'gscLaunchNext50':{'first':next50[0]['number'],'last':next50[-1]['number'],'diagObserved':fn['diagObserved'],'diagExpectedFromMarginals':fn['diagExpectedFromMarginals'],'diagDeficit':fn['diagDeficit'],'diagByNumber':diag_details(fn)}},'identitySimilarity':{'training_vs_launch':identity_test(ft,fl),'training_vs_next50':identity_test(ft,fn,seed=26090318),'launch_vs_next50':identity_test(fl,fn,seed=26090319)},'frozenChanged':False,'interpretation':{'question':'does aggregate REPULSION recur with the same number-specific 20x20/diagonal fingerprint?','manipulationClaim':False}}
    OUT.mkdir(parents=True,exist_ok=True);json.dump(report,open(OUT/'repulsion-fingerprint.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
