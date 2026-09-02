#!/usr/bin/env python3
import json, math, os
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
DATA=Path(os.getenv('LOTO_ARCHIVE','.research-sportloto/draws.json'));OUT=Path(os.getenv('LOTO_OUT','.research-sportloto/out'))
P0=np.array([0.375644995,0.462332301,0.148606811,0.013209494,0.000206398],float);Q=np.array([0.553487068,0.376364933,0.066837266,0.003282397,0.000028336],float)
FREEZE=datetime.fromisoformat('2026-08-26T14:21:00+03:00');SLOTS={'12:07','13:52','16:07','16:22','20:07','23:22'}
def dt(x):return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def mask(a):
 m=0
 for x in a:m|=1<<(x-1)
 return m
def score_mem(p,field):
 y=np.zeros(20);y[np.array(field)-1]=1;p=np.clip(np.asarray(p,float),1e-8,1-1e-8);ll=float(-(y*np.log(p)+(1-y)*np.log(1-p)).sum());br=float(((p-y)**2).sum());hits=int(y[np.argsort(-p,kind='stable')[:4]].sum());return ll,br,hits
def score_k(p,k):
 y=np.zeros(5);y[k]=1;p=np.asarray(p,float);return -math.log(max(1e-15,p[k])),float(((p-y)**2).sum())
def normalize4(v):
 x=np.maximum(1e-8,np.asarray(v,float))
 for _ in range(8):x=np.clip(x*(4/x.sum()),1e-8,.999999)
 return x
def qgamma(g):
 x=P0*np.exp(g*np.arange(5));return x/x.sum()
def mean_gamma(g):return float(np.dot(np.arange(5),qgamma(g)))
def gamma_for_mean(target):
 lo,hi=-12.,12.
 for _ in range(70):
  mid=(lo+hi)/2
  if mean_gamma(mid)<target:lo=mid
  else:hi=mid
 return (lo+hi)/2
def parse():
 raw=json.load(open(DATA));src=raw['draws'] if isinstance(raw,dict) else raw;src.sort(key=lambda d:int(d['number']));rows=[]
 for d in src:
  A=list(map(int,d.get('fieldA',d.get('field1'))));B=list(map(int,d.get('fieldB',d.get('field2'))));z=dt(d.get('date',d.get('draw_date'))).astimezone(FREEZE.tzinfo);rows.append({'number':int(d['number']),'A':A,'B':B,'ma':mask(A),'mb':mask(B),'k':len(set(A)&set(B)),'time':z,'slot':z.strftime('%H:%M'),'date':z.date().isoformat()})
 return raw,rows
def metric_out(vals):
 n=len(vals);return {'n':n,'cumulativeLogGain':sum(x[0] for x in vals),'meanLogGain':sum(x[0] for x in vals)/n,'meanBrierDiff':sum(x[1] for x in vals)/n,'meanHitsBoth':sum(x[2] for x in vals)/n}
def nw(xs):
 x=np.asarray(xs);n=len(x);m=float(x.mean());z=x-m;L=max(1,int(4*(n/100)**(2/9)));om=float(np.dot(z,z)/n)
 for l in range(1,L+1):om+=2*(1-l/(L+1))*float(np.dot(z[l:],z[:-l])/n)
 se=math.sqrt(max(0,om)/n);zz=m/se if se else 0;return {'mean':m,'se':se,'z':zz,'pTwoSided':math.erfc(abs(zz)/math.sqrt(2)),'lag':L}
def block_ci(xs,block=25,reps=2000,seed=20260902):
 x=np.asarray(xs);n=len(x);rng=np.random.default_rng(seed);out=np.empty(reps)
 for r in range(reps):
  vals=[]
  while len(vals)<n:
   s=int(rng.integers(0,n));take=min(block,n-len(vals));idx=(s+np.arange(take))%n;vals.extend(x[idx])
  out[r]=np.mean(vals)
 return {'low':float(np.quantile(out,.025)),'high':float(np.quantile(out,.975)),'block':block,'reps':reps}
def analog_audit(rows,burn=1000,maxlook=2500,kn=25,kk=40,prior_n=25,prior_k=30):
 N=len(rows);v=int(N*.70);tst=int(N*.85);out={s:[] for s in ('development','validation','test')};outk={s:[] for s in out}
 for t in range(max(3,burn),N):
  st='development' if t<v else ('validation' if t<tst else 'test');cur=rows[t-1];lo=max(1,t-maxlook);d=np.fromiter(((cur['ma']^rows[j-1]['ma']).bit_count()+(cur['mb']^rows[j-1]['mb']).bit_count()+2*abs(cur['k']-rows[j-1]['k']) for j in range(lo,t)),dtype=np.int16,count=t-lo);order=np.argsort(d,kind='stable');idxn=(order[:kn]+lo).tolist();idxk=(order[:kk]+lo).tolist();ca=np.zeros(20);cb=np.zeros(20);ck=np.zeros(5)
  for j in idxn:ca[np.array(rows[j]['A'])-1]+=1;cb[np.array(rows[j]['B'])-1]+=1
  for j in idxk:ck[rows[j]['k']]+=1
  pa=normalize4((ca+prior_n*.2)/(len(idxn)+prior_n));pb=normalize4((cb+prior_n*.2)/(len(idxn)+prior_n));pk=(ck+prior_k*P0)/(len(idxk)+prior_k);sa=score_mem(pa,rows[t]['A']);sb=score_mem(pb,rows[t]['B']);ba=score_mem(np.full(20,.2),rows[t]['A']);bb=score_mem(np.full(20,.2),rows[t]['B']);out[st].append(((ba[0]+bb[0])-(sa[0]+sb[0]),(sa[1]+sb[1])-(ba[1]+bb[1]),sa[2]+sb[2]));sk=score_k(pk,rows[t]['k']);bk=score_k(P0,rows[t]['k']);outk[st].append((bk[0]-sk[0],sk[1]-bk[1],0))
 return {'number':{s:metric_out(x) for s,x in out.items()},'K':{s:metric_out(x) for s,x in outk.items()},'numberTestHAC':nw([x[0] for x in out['test']]),'numberTestCI':block_ci([x[0] for x in out['test']]),'KTestHAC':nw([x[0] for x in outk['test']]),'KTestCI':block_ci([x[0] for x in outk['test']],seed=20260903)}
def gof(counts):
 c=np.asarray(counts,float);n=int(c.sum());e=n*P0;pear=float(np.sum((c-e)**2/e));g2=float(2*np.sum(np.where(c>0,c*np.log(c/e),0)));sf4=lambda x:math.exp(-x/2)*(1+x/2);mean=float(np.dot(c,np.arange(5))/n);mu=float(np.dot(P0,np.arange(5)));var=float(np.dot(P0,(np.arange(5)-mu)**2));z=(mean-mu)/math.sqrt(var/n);gh=gamma_for_mean(mean);qh=qgamma(gh);llr=float(np.dot(c,np.log(qh/P0)));p1=math.erfc(math.sqrt(max(0,2*llr)/2));return {'n':n,'counts':[int(x) for x in c],'expected':[float(x) for x in e],'pearson':pear,'pearsonP_df4':sf4(pear),'G2':g2,'G2P_df4':sf4(g2),'meanK':mean,'meanZ':z,'meanTwoSidedP':math.erfc(abs(z)/math.sqrt(2)),'gammaMLE':gh,'tiltLogLRVsIID':llr,'tiltLrtP_df1':p1}
def scan_actual(rows,w=50,forward=50):
 le=np.log(Q/P0);x=np.array([le[r['k']] for r in rows]);cs=np.r_[0,np.cumsum(x)];thr=math.log(20);sig=[];cool=-1
 for t in range(w-1,len(rows)-1):
  if t<=cool:continue
  lr=float(cs[t+1]-cs[t+1-w])
  if lr>=thr:
   end=min(len(rows),t+1+forward);rr=rows[t+1:end];c=Counter(r['k'] for r in rr);sig.append({'signalDraw':rows[t]['number'],'windowStart':rows[t+1-w]['number'],'windowLogLR':lr,'windowE':math.exp(lr),'forwardN':len(rr),'forwardCounts':[c[i] for i in range(5)],'forwardMeanK':sum(r['k'] for r in rr)/len(rr),'forwardLogLR':float(cs[end]-cs[t+1])});cool=t+forward
 return sig
def post_signal_split(signals):
 def aggregate(ss):
  c=np.sum(np.array([s['forwardCounts'] for s in ss]),axis=0);return c.astype(int),int(c.sum())
 cut=len(signals)//2;dev=signals[:cut];test=signals[cut:];cd,nd=aggregate(dev);ct,nt=aggregate(test);mean_d=float(np.dot(cd,np.arange(5))/nd);g=gamma_for_mean(mean_d);q=qgamma(g);lr=float(np.dot(ct,np.log(q/P0)));bd=sum(n*(score_k(q,k)[1]-score_k(P0,k)[1]) for k,n in enumerate(ct));ca,na=aggregate(signals);return {'developmentSignals':len(dev),'testSignals':len(test),'developmentCounts':cd.tolist(),'testCounts':ct.tolist(),'allCounts':ca.tolist(),'developmentGammaMLE':g,'testLogLR_earlyModel_vs_IID':lr,'testE':math.exp(lr),'testMeanBrierDiff':bd/nt,'allGOF':gof(ca)}
def scan_null(N,obs_count,obs_agg,obs_maxE,reps=20000,batch=250,seed=20260902):
 rng=np.random.default_rng(seed);le=np.log(Q/P0);thr=math.log(20);counts=[];aggs=[];maxes=[];allneg=[]
 for start in range(0,reps,batch):
  b=min(batch,reps-start);seq=rng.choice(5,size=(b,N),p=P0/P0.sum());vals=le[seq];cs=np.c_[np.zeros(b),np.cumsum(vals,axis=1)];roll=cs[:,50:]-cs[:,:-50]
  for r in range(b):
   idx=np.flatnonzero(roll[r,:-1]>=thr);sel=[];last=-10**9
   for ii in idx:
    t=ii+49
    if t<=last+50:continue
    sel.append(t);last=t
   f=[]
   for t in sel:
    end=min(N,t+51);f.append(float(cs[r,end]-cs[r,t+1]))
   counts.append(len(sel));aggs.append(sum(f));allneg.append(bool(f) and all(x<0 for x in f));maxes.append(float(np.max(roll[r])))
 counts=np.array(counts);aggs=np.array(aggs);maxes=np.array(maxes);allneg=np.array(allneg);same=counts==obs_count
 return {'reps':reps,'meanSignalCount':float(counts.mean()),'countQuantiles':{str(q):float(np.quantile(counts,q)) for q in (.01,.05,.5,.95,.99)},'P_count_ge_observed':float(np.mean(counts>=obs_count)),'P_maxE_ge_observed':float(np.mean(maxes>=math.log(obs_maxE))),'sameCountN':int(same.sum()),'P_aggregateForwardLogLR_le_observed_given_sameCount':float(np.mean(aggs[same]<=obs_agg)) if same.any() else None,'P_allForwardBlocksNegative_given_sameCount':float(np.mean(allneg[same])) if same.any() else None,'aggregateForwardExpectedGivenSameCount':float(aggs[same].mean()) if same.any() else None}
def main():
 raw,rows=parse();audit=json.load(open(OUT/'audit.json'));frozen=[r for r in rows if r['time']<=FREEZE and r['slot'] in SLOTS];c=Counter(r['k'] for r in frozen);analog=analog_audit(rows);signals=scan_actual(rows);obs_agg=sum(s['forwardLogLR'] for s in signals);top=audit['repulsionPersistence']['topNonOverlapping50Windows'];obs_max=max(x['E'] for x in top);report={'generatedAt':datetime.now().astimezone().isoformat(),'archiveLast':rows[-1]['number'],'frozenSlotsIIDGoodnessOfFit':gof([c[i] for i in range(5)]),'frozenSlotBySlotIIDGoodnessOfFit':{},'nonlinearJohnsonAnalog':analog,'repulsionSignalForward':signals,'postSignalChronologicalSplit':post_signal_split(signals),'rollingScanNullCalibration':scan_null(len(rows),len(signals),obs_agg,obs_max)}
 for s in sorted(SLOTS):
  z=Counter(r['k'] for r in frozen if r['slot']==s);report['frozenSlotBySlotIIDGoodnessOfFit'][s]=gof([z[i] for i in range(5)])
 json.dump(report,open(OUT/'supplement.json','w'),indent=2,ensure_ascii=False);lines=['# Supplement: multiplicity, IID fit, nonlinear analogue','',f"Archive through #{rows[-1]['number']}.",'','## Frozen historical GSCH slots vs IID',f"N={report['frozenSlotsIIDGoodnessOfFit']['n']}; meanK={report['frozenSlotsIIDGoodnessOfFit']['meanK']:.6f}; Pearson p≈{report['frozenSlotsIIDGoodnessOfFit']['pearsonP_df4']:.6f}; one-parameter tilt p≈{report['frozenSlotsIIDGoodnessOfFit']['tiltLrtP_df1']:.6f}.",'','## Nonlinear Johnson analogue',f"Number test mean log gain={analog['number']['test']['meanLogGain']:.8f}, CI=[{analog['numberTestCI']['low']:.8f},{analog['numberTestCI']['high']:.8f}]. K test mean log gain={analog['K']['test']['meanLogGain']:.8f}, CI=[{analog['KTestCI']['low']:.8f},{analog['KTestCI']['high']:.8f}].",'','## Rolling REPULSION scan multiplicity',f"Observed signals={len(signals)}; IID simulation mean={report['rollingScanNullCalibration']['meanSignalCount']:.3f}; P(count≥observed)={report['rollingScanNullCalibration']['P_count_ge_observed']:.6f}; P(maxE≥observed)={report['rollingScanNullCalibration']['P_maxE_ge_observed']:.6f}.",'','## Post-signal chronological test',f"Early fitted gamma={report['postSignalChronologicalSplit']['developmentGammaMLE']:.6f}; later logLR vs IID={report['postSignalChronologicalSplit']['testLogLR_earlyModel_vs_IID']:.6f}; E={report['postSignalChronologicalSplit']['testE']:.6f}; Brier diff={report['postSignalChronologicalSplit']['testMeanBrierDiff']:.8f}."];(OUT/'supplement.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
