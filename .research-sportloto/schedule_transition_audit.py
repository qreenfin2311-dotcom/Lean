#!/usr/bin/env python3
import json, math, os, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'))
OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
P0=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
FREEZE_TZ=datetime.fromisoformat('2026-08-26T14:21:00+03:00').tzinfo
EPS=1e-15

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def k_of(a,b): return len(set(a)&set(b))
def loge(k): return math.log(Q[k]/P0[k])
def exp_safe(x):
    if x>700:return '>=1e304'
    if x<-745:return 0.0
    return math.exp(x)
def brier_diff(k):
    return sum((Q[i]-(1 if i==k else 0))**2 for i in range(5))-sum((P0[i]-(1 if i==k else 0))**2 for i in range(5))
def summ(rows):
    n=len(rows); c=[0]*5
    for r in rows:c[r['k']]+=1
    lr=sum(c[i]*loge(i) for i in range(5)); sk=sum(i*c[i] for i in range(5))
    return {'n':n,'counts':c,'meanK':sk/n if n else None,'logLR_q_vs_IID':lr,'E':exp_safe(lr),'meanBrierDiff_q_minus_IID':sum(brier_diff(r['k']) for r in rows)/n if n else None,'firstDraw':rows[0]['number'] if n else None,'lastDraw':rows[-1]['number'] if n else None}

def load():
    raw=json.load(open(DATA,encoding='utf-8')); src=raw['draws'] if isinstance(raw,dict) else raw; src.sort(key=lambda x:int(x['number'])); rows=[]
    for d in src:
        A=list(map(int,d.get('fieldA',d.get('field1'))));B=list(map(int,d.get('fieldB',d.get('field2'))));z=dt(d.get('date',d.get('draw_date'))).astimezone(FREEZE_TZ)
        rows.append({'number':int(d['number']),'A':A,'B':B,'k':k_of(A,B),'date':z.date().isoformat(),'slot':z.strftime('%H:%M'),'time':z})
    return raw,rows

def daily_signatures(rows):
    by=defaultdict(list)
    for i,r in enumerate(rows):by[r['date']].append((i,r['slot']))
    dates=[]
    for d in sorted(by):
        sig='|'.join(sorted({s for _,s in by[d]})); dates.append((d,sig,min(i for i,_ in by[d]),max(i for i,_ in by[d])))
    eras=[]
    for d,s,a,b in dates:
        if eras and eras[-1]['signature']==s:
            eras[-1]['end']=d;eras[-1]['lastIndex']=b;eras[-1]['days']+=1
        else:eras.append({'start':d,'end':d,'signature':s,'firstIndex':a,'lastIndex':b,'days':1})
    return [e for e in eras if e['days']>=3]

def boundary_stats(rows,eras,w):
    out=[]
    for j in range(1,len(eras)):
        e=eras[j];i=e['firstIndex']
        if i<w or i+w>len(rows):continue
        pre=rows[i-w:i];post=rows[i:i+w]
        out.append({'date':e['start'],'firstDraw':rows[i]['number'],'from':eras[j-1]['signature'],'to':e['signature'],'pre':summ(pre),'post':summ(post),'deltaMeanK':summ(post)['meanK']-summ(pre)['meanK'],'postMeanLogE':summ(post)['logLR_q_vs_IID']/w})
    return out

def shift_test(rows,eras,w,reps=20000,seed=260903):
    idx=[e['firstIndex'] for e in eras[1:] if e['firstIndex']>=w and e['firstIndex']+w<=len(rows)]
    ks=[r['k'] for r in rows];n=len(ks)
    def stat(off):
        ds=[]
        for i in idx:
            pre=[ks[(t+off)%n] for t in range(i-w,i)];post=[ks[(t+off)%n] for t in range(i,i+w)]
            ds.append(sum(post)/w-sum(pre)/w)
        return sum(ds)/len(ds)
    obs=stat(0);rng=random.Random(seed);lo=hi=0;vals=[]
    for _ in range(reps):
        off=rng.randrange(1,n);v=stat(off);vals.append(v);lo+=v<=obs;hi+=v>=obs
    vals.sort()
    return {'boundaries':len(idx),'window':w,'observedMeanDeltaK_post_minus_pre':obs,'circularShiftLowerP':(lo+1)/(reps+1),'circularShiftUpperP':(hi+1)/(reps+1),'twoSidedP':min(1,2*min((lo+1)/(reps+1),(hi+1)/(reps+1))),'null95':[vals[int(.025*reps)],vals[int(.975*reps)]],'reps':reps}

def exact_era_probe(rows,eras,start,wlist=(25,50,100,200)):
    e=next((x for x in eras if x['start']==start),None)
    if not e:return None
    i=e['firstIndex']; out={'start':start,'firstDraw':rows[i]['number'],'signature':e['signature'],'windows':{}}
    for w in wlist:
        if i+w<=len(rows):out['windows'][str(w)]={'post':summ(rows[i:i+w]),'pre':summ(rows[max(0,i-w):i])}
    return out

def gsc_initial(rows):
    slots={'12:07','13:52','16:07','16:22','20:07','23:22'}
    x=[r for r in rows if r['date']>='2025-10-15' and r['slot'] in slots]
    out={'n':len(x),'blocks':[],'prefix':{}}
    for w in (25,50,100,200,500,1000):
        if len(x)>=w:out['prefix'][str(w)]=summ(x[:w])
    for i in range(0,min(len(x),400),25):
        b=x[i:i+25]
        if len(b)==25:out['blocks'].append({'block':i//25+1,'startDraw':b[0]['number'],'endDraw':b[-1]['number'],**summ(b)})
    return out

def main():
    raw,rows=load();eras=daily_signatures(rows)
    report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'count':len(rows),'first':rows[0]['number'],'last':rows[-1]['number'],'retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None},'scheduleEras':eras,'boundaryTests':{},'specific':{},'gscInitial':gsc_initial(rows)}
    for w in (25,50,100):
        report['boundaryTests'][str(w)]={'boundaries':boundary_stats(rows,eras,w),'globalShiftTest':shift_test(rows,eras,w)}
    for d in ('2024-09-23','2025-10-15','2025-12-17','2026-07-17','2026-08-20'):
        report['specific'][d]=exact_era_probe(rows,eras,d)
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'schedule-transition-audit.json';json.dump(report,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps({'archive':report['archive'],'global':{w:report['boundaryTests'][w]['globalShiftTest'] for w in report['boundaryTests']},'specific':report['specific'],'gscPrefix':report['gscInitial']['prefix'],'gscBlocks':report['gscInitial']['blocks'][:8]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
