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

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def k(a,b): return len(set(a)&set(b))
def loge(x): return math.log(Q[x]/P0[x])
def ex(x): return 0.0 if x<-745 else ('>=1e304' if x>700 else math.exp(x))
def bdiff(x):
    return sum((Q[i]-(1 if i==x else 0))**2 for i in range(5))-sum((P0[i]-(1 if i==x else 0))**2 for i in range(5))
def summ(rs):
    c=[0]*5
    for r in rs:c[r['k']]+=1
    n=len(rs);lr=sum(c[i]*loge(i) for i in range(5));sk=sum(i*c[i] for i in range(5))
    return {'n':n,'counts':c,'meanK':sk/n if n else None,'logLR':lr,'E':ex(lr),'meanBrierDiff':sum(bdiff(r['k']) for r in rs)/n if n else None,'firstDraw':rs[0]['number'] if n else None,'lastDraw':rs[-1]['number'] if n else None}
def perm_diff(a,b,reps=50000,seed=97599808):
    vals=[r['k'] for r in a+b];na=len(a);obs=sum(r['k'] for r in a)/na-sum(r['k'] for r in b)/len(b);rng=random.Random(seed);lo=hi=0
    for _ in range(reps):
        x=vals[:];rng.shuffle(x);d=sum(x[:na])/na-sum(x[na:])/len(b);lo+=d<=obs;hi+=d>=obs
    pl=(lo+1)/(reps+1);ph=(hi+1)/(reps+1)
    return {'observedMeanKDiff_A_minus_B':obs,'lowerP':pl,'upperP':ph,'twoSidedP':min(1,2*min(pl,ph)),'reps':reps}
def load():
    raw=json.load(open(DATA));src=raw['draws'] if isinstance(raw,dict) else raw;src.sort(key=lambda x:int(x['number']));rs=[]
    for d in src:
        A=list(map(int,d.get('fieldA',d.get('field1'))));B=list(map(int,d.get('fieldB',d.get('field2'))));z=dt(d.get('date',d.get('draw_date'))).astimezone(TZ)
        rs.append({'number':int(d['number']),'k':k(A,B),'slot':z.strftime('%H:%M'),'date':z.date().isoformat(),'time':z})
    return raw,rs
def sig_by_date(rs):
    by=defaultdict(set)
    for r in rs:by[r['date']].add(r['slot'])
    return by
def eras(rs):
    by=sig_by_date(rs);out=[]
    for d in sorted(by):
        s=tuple(sorted(by[d]))
        if out and out[-1]['sig']==s:out[-1]['end']=d;out[-1]['days']+=1
        else:out.append({'start':d,'end':d,'days':1,'sig':s})
    return [e for e in out if e['days']>=3]
def cohort_after(rs,start,slots,n=50):
    return [r for r in rs if r['date']>=start and r['slot'] in slots][:n]
def main():
    raw,rs=load();es=eras(rs)
    launch='2025-10-15';prev=next(e for e in es if e['end']<'2025-10-15' and e['days']>=3);cur=next(e for e in es if e['start']=='2025-10-15')
    old=set(prev['sig']);new=set(cur['sig']);added=new-old;retained=new&old
    launch_all=[r for r in rs if r['date']>=launch and r['slot'] in added]
    a1=launch_all[:50];a2=launch_all[50:100];ctrl=cohort_after(rs,launch,retained,50)
    prefix={}
    for n in (10,20,25,30,40,50,60,75,100):prefix[str(n)]=summ(launch_all[:n])
    cohorts=[]
    for j in range(1,len(es)):
        e0,e1=es[j-1],es[j];add=set(e1['sig'])-set(e0['sig'])
        if not add:continue
        x=[r for r in rs if e1['start']<=r['date']<=e1['end'] and r['slot'] in add][:50]
        if len(x)>=25:cohorts.append({'start':e1['start'],'addedSlots':sorted(add),**summ(x)})
    cohorts.sort(key=lambda x:x['logLR'],reverse=True)
    rank=1+sum(c['logLR']>summ(a1)['logLR']+1e-12 for c in cohorts)
    report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'count':len(rs),'last':rs[-1]['number'],'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},'launch':{'date':launch,'previousSignature':list(prev['sig']),'newSignature':list(cur['sig']),'addedSlots':sorted(added),'retainedSlots':sorted(retained),'first50Added':summ(a1),'next50Added':summ(a2),'first50Retained':summ(ctrl),'firstVsNextPermutation':perm_diff(a1,a2),'firstVsRetainedPermutation':perm_diff(a1,ctrl,seed=97599809),'prefix':prefix},'addedSlotCohorts':cohorts,'launchRankByLogLR':rank,'cohortCount':len(cohorts)}
    OUT.mkdir(parents=True,exist_ok=True);json.dump(report,open(OUT/'gsc-launch-replication.json','w'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
