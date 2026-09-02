#!/usr/bin/env python3
import json, math, os, random
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

DATA = Path(os.getenv('LOTO_ARCHIVE', '.research-sportloto/draws.json'))
OUT = Path(os.getenv('LOTO_OUT', '.research-sportloto/out'))
P0 = [0.375644995,0.462332301,0.148606811,0.013209494,0.000206398]
Q = [0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
GAMMA = -0.5933185
SLOTS = {'12:07','13:52','16:07','16:22','20:07','23:22'}
FREEZE = datetime.fromisoformat('2026-08-26T14:21:00+03:00')
EPS = 1e-12
GAMMAS = [-1.2,-0.8,-0.5,-0.25,0,0.25,0.5,0.8,1.2]
GPRIOR = [0.025,0.06,0.12,0.18,0.23,0.18,0.12,0.06,0.025]

def dt(x): return datetime.fromisoformat(x.replace('+0300','+03:00').replace('Z','+00:00'))
def k_of(a,b): return len(set(a)&set(b))
def mask(a):
    m=0
    for x in a: m |= 1<<(x-1)
    return m

def cat_score(p,k):
    return -math.log(max(EPS,p[k])), sum((p[i]-(1 if i==k else 0))**2 for i in range(5))

def mem_score(p,field):
    s=set(field); ll=br=0.0
    for i,x in enumerate(p,1):
        x=min(1-1e-8,max(1e-8,x)); y=1.0 if i in s else 0.0
        ll-=y*math.log(x)+(1-y)*math.log(1-x); br+=(x-y)**2
    top=sorted(range(20),key=lambda i:(-p[i],i))[:4]
    return ll,br,sum(1 for i in top if i+1 in s)

def norm4(v):
    x=[max(1e-8,float(z)) for z in v]
    for _ in range(8):
        sc=4/sum(x); x=[min(.999999,max(1e-8,z*sc)) for z in x]
    return x

def qgamma(g):
    x=[P0[k]*math.exp(g*k) for k in range(5)]; z=sum(x); return [v/z for v in x]
def mean_gamma(g): return sum(k*p for k,p in enumerate(qgamma(g)))
def gamma_for_mean(target):
    lo,hi=-12.0,12.0
    for _ in range(70):
        mid=(lo+hi)/2
        if mean_gamma(mid)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def exp_safe(x):
    if x>700: return '>=1e304'
    if x<-745: return 0.0
    return math.exp(x)
def brier_diff(p,k): return cat_score(p,k)[1]-cat_score(P0,k)[1]

def dist_summary(rows):
    c=[0]*5
    for r in rows: c[r['k']]+=1
    n=len(rows); sk=sum(i*c[i] for i in range(5)); lr=sum(c[i]*math.log(Q[i]/P0[i]) for i in range(5))
    return {'n':n,'counts':c,'sumK':sk,'meanK':sk/n if n else None,'fixedQ_logLR':lr,'fixedQ_E':exp_safe(lr),
            'meanBrierDiff_q_minus_IID':sum(brier_diff(Q,r['k']) for r in rows)/n if n else None,
            'first':rows[0]['number'] if n else None,'last':rows[-1]['number'] if n else None}

def prefix_counts(rows,key):
    out=[[0]*20]
    for r in rows:
        z=out[-1].copy()
        for x in r[key]: z[x-1]+=1
        out.append(z)
    return out

def roll_prob(pref,t,w,beta=1.0,prior=100):
    a=max(0,t-w); n=t-a; v=[]
    for i in range(20):
        f=(pref[t][i]-pref[a][i]+prior*.2)/(n+prior); v.append(.2+beta*(f-.2))
    return norm4(v)

def slot_prob(stat,field,prior=200):
    if stat is None: return [.2]*20
    return norm4([(stat[field][i]+prior*.2)/(stat['n']+prior) for i in range(20)])

def trans_prob(mat,occ,orig,prior=30,strength=.5):
    if orig is None: return [.2]*20
    v=[0.0]*20
    for o in orig:
        i=o-1; den=occ[i]+prior
        for j in range(20): v[j]+=(mat[i][j]+prior*.2)/den
    return norm4([.2+strength*(z/len(orig)-.2) for z in v])

def update_trans(mat,occ,orig,target):
    if orig is None:return
    for o in orig:
        i=o-1; occ[i]+=1
        for x in target: mat[i][x-1]+=1

def analog_prob(rows,t,field,maxlook=2500,neighbors=25,prior=25):
    if t<3:return [.2]*20
    lo=max(1,t-maxlook); cur=rows[t-1]; cand=[]
    for j in range(lo,t):
        prev=rows[j-1]
        d=(cur['ma']^prev['ma']).bit_count()+(cur['mb']^prev['mb']).bit_count()+2*abs(cur['k']-prev['k'])
        cand.append((d,j))
    cand.sort(key=lambda x:(x[0],-x[1])); chosen=cand[:neighbors]; c=[0]*20
    for _,j in chosen:
        for x in rows[j][field]: c[x-1]+=1
    return norm4([(c[i]+prior*.2)/(len(chosen)+prior) for i in range(20)])

def analog_k(rows,t,maxlook=2500,neighbors=40,prior=30):
    if t<3:return P0[:]
    lo=max(1,t-maxlook); cur=rows[t-1]; cand=[]
    for j in range(lo,t):
        prev=rows[j-1]
        d=(cur['ma']^prev['ma']).bit_count()+(cur['mb']^prev['mb']).bit_count()+2*abs(cur['k']-prev['k'])
        cand.append((d,j))
    cand.sort(key=lambda x:(x[0],-x[1])); c=[0]*5
    for _,j in cand[:neighbors]: c[rows[j]['k']]+=1
    return [(c[i]+prior*P0[i])/(min(neighbors,len(cand))+prior) for i in range(5)]

def nw(xs):
    n=len(xs); m=sum(xs)/n; z=[x-m for x in xs]; L=max(1,int(4*(n/100)**(2/9))); om=sum(x*x for x in z)/n
    for l in range(1,L+1): om+=2*(1-l/(L+1))*sum(z[t]*z[t-l] for t in range(l,n))/n
    se=math.sqrt(max(0,om)/n); zz=m/se if se else 0
    return {'n':n,'mean':m,'lag':L,'se':se,'z':zz,'pTwoSided':math.erfc(abs(zz)/math.sqrt(2))}

def block_ci(xs,block=25,reps=1200,seed=260902):
    rnd=random.Random(seed); n=len(xs); vals=[]
    for _ in range(reps):
        s=0.0; used=0
        while used<n:
            st=rnd.randrange(n); take=min(block,n-used)
            for j in range(take): s+=xs[(st+j)%n]
            used+=take
        vals.append(s/n)
    vals.sort(); return {'low':vals[int(.025*reps)],'high':vals[int(.975*reps)],'block':block,'reps':reps}

def metric(): return {'n':0,'ll':0.0,'br':0.0,'hits':0,'gain':0.0,'bd':0.0,'diffs':[]}
def add_mem(m,sc,base):
    m['n']+=1;m['ll']+=sc[0];m['br']+=sc[1];m['hits']+=sc[2];d=base[0]-sc[0];m['gain']+=d;m['bd']+=sc[1]-base[1];m['diffs'].append(d)
def metric_out(m):
    n=m['n'];return {'n':n,'meanLogLoss':m['ll']/n,'meanBrier':m['br']/n,'meanHitsBothFields':m['hits']/n,
                     'cumulativePseudoLogGainVsUniform':m['gain'],'meanPseudoLogGainVsUniform':m['gain']/n,'meanBrierDiffVsUniform':m['bd']/n}
def kmetric(): return {'n':0,'ll':0.0,'br':0.0,'gain':0.0,'bd':0.0,'diffs':[]}
def add_k(m,sc,base):
    m['n']+=1;m['ll']+=sc[0];m['br']+=sc[1];d=base[0]-sc[0];m['gain']+=d;m['bd']+=sc[1]-base[1];m['diffs'].append(d)
def kmetric_out(m):
    n=m['n'];return {'n':n,'meanLogLoss':m['ll']/n,'meanBrier':m['br']/n,'cumulativeLogLRVsIID':m['gain'],
                     'E':exp_safe(m['gain']),'meanLogGainVsIID':m['gain']/n,'meanBrierDiffVsIID':m['bd']/n}

def leading_sv(R):
    v=[1/math.sqrt(20)]*20
    for _ in range(50):
        rv=[sum(R[i][j]*v[j] for j in range(20)) for i in range(20)]; w=[sum(R[i][j]*rv[i] for i in range(20)) for j in range(20)]
        no=math.sqrt(sum(x*x for x in w)) or 1; v=[x/no for x in w]
    rv=[sum(R[i][j]*v[j] for j in range(20)) for i in range(20)]; return math.sqrt(sum(x*x for x in rv))

def graph_matrix(rows):
    n=len(rows); a=[0]*20;b=[0]*20;o=[[0]*20 for _ in range(20)]
    for r in rows:
        for x in r['A']:a[x-1]+=1
        for y in r['B']:b[y-1]+=1
        for x in r['A']:
            for y in r['B']:o[x-1][y-1]+=1
    R=[[0.0]*20 for _ in range(20)];do=de=fr=0.0
    for i in range(20):
        for j in range(20):
            e=a[i]*b[j]/n if n else 0; z=(o[i][j]-e)/math.sqrt(e) if e else 0;R[i][j]=z;fr+=z*z
            if i==j:do+=o[i][j];de+=e
    return R,do,de,math.sqrt(fr)

def graph_stats(rows,perms=150):
    if len(rows)<20:return None
    R,do,de,fr=graph_matrix(rows);sv=leading_sv(R);dl=sh=0;n=len(rows)
    for p in range(1,perms+1):
        shift=1+int(((p*.61803398875)%1)*(n-1)); rr=[]
        for i,r in enumerate(rows): z=dict(r);z['B']=rows[(i+shift)%n]['B'];rr.append(z)
        Rp,dop,dep,frp=graph_matrix(rr);svp=leading_sv(Rp);dl+=dop<=do;sh+=svp>=sv
    return {'n':n,'diagonalObserved':do,'diagonalExpectedFromMarginals':de,'diagonalDeficit':do-de,
            'diagonalLowerTailPermutationP':(dl+1)/(perms+1),'residualFrobenius':fr,'residualLeadingSingular':sv,'singularPermutationP':(sh+1)/(perms+1)}

def cosine_graph(a,b):
    if len(a)<20 or len(b)<20:return None
    A,*_=graph_matrix(a);B,*_=graph_matrix(b);dot=x=y=0.0
    for i in range(20):
        for j in range(20):dot+=A[i][j]*B[i][j];x+=A[i][j]**2;y+=B[i][j]**2
    return dot/math.sqrt(x*y) if x*y else None

def mi_lag(vals):
    c=[[0]*5 for _ in range(5)];a=[0]*5;b=[0]*5;n=len(vals)-1
    for t in range(1,len(vals)):x=vals[t-1];y=vals[t];c[x][y]+=1;a[x]+=1;b[y]+=1
    mi=0.0
    for i in range(5):
        for j in range(5):
            if c[i][j]:
                pij=c[i][j]/n;mi+=pij*math.log(pij/((a[i]/n)*(b[j]/n)))
    return mi

def mi_test(vals,perms=500):
    if len(vals)<30:return None
    obs=mi_lag(vals);rng=random.Random(991);hi=0
    for _ in range(perms): y=vals[:];rng.shuffle(y);hi+=mi_lag(y)>=obs
    return {'n':len(vals),'mutualInformationNats':obs,'permutationP':(hi+1)/(perms+1)}

def scan_repulsion(rows,w=50,threshold=math.log(20),forward=50):
    le=[math.log(Q[r['k']]/P0[r['k']]) for r in rows];pr=[0.0]
    for x in le:pr.append(pr[-1]+x)
    sig=[];cool=-1
    for t in range(w-1,len(rows)-1):
        if t<=cool:continue
        lr=pr[t+1]-pr[t+1-w]
        if lr>=threshold:
            e=min(len(rows),t+1+forward);f=pr[e]-pr[t+1];rr=rows[t+1:e]
            sig.append({'signalDraw':rows[t]['number'],'date':rows[t]['date'],'windowStart':rows[t+1-w]['number'],'windowLogLR':lr,'windowE':exp_safe(lr),
                        'forwardN':len(rr),'forwardLogLR':f,'forwardE':exp_safe(f),'forwardMeanK':sum(x['k'] for x in rr)/len(rr) if rr else None})
            cool=t+forward
    return sig

def top_windows(rows,w=50,top=20):
    le=[math.log(Q[r['k']]/P0[r['k']]) for r in rows];pr=[0.0]
    for x in le:pr.append(pr[-1]+x)
    cand=sorted([(pr[i+w]-pr[i],i) for i in range(len(rows)-w+1)],reverse=True);out=[]
    for lr,i in cand:
        if any(not(i+w<=j or j+w<=i) for _,j in out):continue
        out.append((lr,i))
        if len(out)>=top:break
    return [{'start':rows[i]['number'],'end':rows[i+w-1]['number'],'logLR':lr,'E':exp_safe(lr),'meanK':sum(r['k'] for r in rows[i:i+w])/w} for lr,i in out]

def schedule_eras(rows):
    by=defaultdict(set)
    for r in rows:by[r['mskDate']].add(r['slot'])
    eras=[]
    for d in sorted(by):
        s='|'.join(sorted(by[d]))
        if eras and eras[-1]['signature']==s:eras[-1]['end']=d;eras[-1]['days']+=1
        else:eras.append({'start':d,'end':d,'days':1,'signature':s})
    return [e for e in eras if e['days']>=3]

def main():
    raw=json.load(open(DATA,encoding='utf-8'));src=raw if isinstance(raw,list) else raw['draws'];src.sort(key=lambda x:int(x['number']));rows=[]
    for i,d in enumerate(src):
        n=int(d['number']);A=list(map(int,d.get('fieldA',d.get('field1'))));B=list(map(int,d.get('fieldB',d.get('field2'))))
        if len(A)!=4 or len(B)!=4 or len(set(A))<4 or len(set(B))<4 or min(A+B)<1 or max(A+B)>20:raise ValueError(f'bad draw {n}')
        if i and n!=int(src[i-1]['number'])+1:raise ValueError(f'gap before {n}')
        z=dt(d.get('date',d.get('draw_date'))).astimezone(FREEZE.tzinfo)
        rows.append({'number':n,'date':d.get('date',d.get('draw_date')),'A':A,'B':B,'ma':mask(A),'mb':mask(B),'k':k_of(A,B),'slot':z.strftime('%H:%M'),'mskDate':z.date().isoformat(),'time':z})
    N=len(rows);burn=1000;vstart=int(N*.70);tstart=int(N*.85);stage=lambda t:'development' if t<vstart else ('validation' if t<tstart else 'test')
    ledger=[];lr=0.0;bq=b0=0.0
    for r in rows:
        if r['time']<=FREEZE or r['slot'] not in SLOTS:continue
        e=Q[r['k']]/P0[r['k']];lr+=math.log(e);sq=cat_score(Q,r['k'])[1];s0=cat_score(P0,r['k'])[1];bq+=sq;b0+=s0
        ledger.append({'draw_id':r['number'],'time_msk':f"{r['mskDate']} {r['slot']}",'K':r['k'],'e_t':e,'cumulative_logLR':lr,'cumulative_E':exp_safe(lr),'cumulative_delta_logloss':lr,'Brier_q':sq,'Brier_IID':s0,'Brier_difference':sq-s0})
    prefA=prefix_counts(rows,'A');prefB=prefix_counts(rows,'B')
    n_names=['uniform']+[f'hot{w}' for w in (25,50,100,200,500,1000)]+[f'cold{w}' for w in (50,200,500)]+['slot','same_markov','cross_markov','analog_joint','blend']
    nm={x:{s:metric() for s in ('development','validation','test')} for x in n_names};slotstat={};sameA=[[0]*20 for _ in range(20)];sameB=[[0]*20 for _ in range(20)];crossA=[[0]*20 for _ in range(20)];crossB=[[0]*20 for _ in range(20)];oa=[0]*20;ob=[0]*20;oca=[0]*20;ocb=[0]*20;prev=None
    k_names=['iid','frozen_q']+[f'dir{w}' for w in (20,50,100,250,500)]+[f'tilt{w}' for w in (20,50,100,250,500)]+['slot50','slot200','ctx1','ctx2','analog_k']+[f'hmm{h}' for h in ('005','02','05','10')]
    km={x:{s:kmetric() for s in ('development','validation','test')} for x in k_names};kp=[[0]*5];kslot={};ctx1=defaultdict(lambda:[0]*5);ctx1n=Counter();ctx2=defaultdict(lambda:[0]*5);ctx2n=Counter();hpost={.005:GPRIOR[:],.02:GPRIOR[:],.05:GPRIOR[:],.10:GPRIOR[:]}
    for t,r in enumerate(rows):
        st=stage(t);ss=slotstat.get(r['slot']);pa={'uniform':[.2]*20};pb={'uniform':[.2]*20}
        for w in (25,50,100,200,500,1000):pa[f'hot{w}']=roll_prob(prefA,t,w,1);pb[f'hot{w}']=roll_prob(prefB,t,w,1)
        for w in (50,200,500):pa[f'cold{w}']=roll_prob(prefA,t,w,-1);pb[f'cold{w}']=roll_prob(prefB,t,w,-1)
        pa['slot']=slot_prob(ss,'A');pb['slot']=slot_prob(ss,'B');pa['same_markov']=trans_prob(sameA,oa,prev['A'] if prev else None);pb['same_markov']=trans_prob(sameB,ob,prev['B'] if prev else None);pa['cross_markov']=trans_prob(crossA,oca,prev['B'] if prev else None);pb['cross_markov']=trans_prob(crossB,ocb,prev['A'] if prev else None);pa['analog_joint']=analog_prob(rows,t,'A');pb['analog_joint']=analog_prob(rows,t,'B')
        pa['blend']=norm4([.5*.2+.5*sum(pa[x][i] for x in ('hot200','slot','same_markov','cross_markov','analog_joint'))/5 for i in range(20)]);pb['blend']=norm4([.5*.2+.5*sum(pb[x][i] for x in ('hot200','slot','same_markov','cross_markov','analog_joint'))/5 for i in range(20)])
        ba=mem_score(pa['uniform'],r['A']);bb=mem_score(pb['uniform'],r['B'])
        if t>=burn:
            for name in n_names:
                a=mem_score(pa[name],r['A']);b=mem_score(pb[name],r['B']);add_mem(nm[name][st],(a[0]+b[0],a[1]+b[1],a[2]+b[2]),(ba[0]+bb[0],ba[1]+bb[1],1.6))
        pred={'iid':P0[:],'frozen_q':Q[:]}
        for w in (20,50,100,250,500):
            a=max(0,t-w);c=[kp[t][i]-kp[a][i] for i in range(5)];nn=sum(c);pred[f'dir{w}']=[(c[i]+50*P0[i])/(nn+50) for i in range(5)];target=(sum(i*c[i] for i in range(5))+50*.8)/(nn+50);pred[f'tilt{w}']=qgamma(gamma_for_mean(target))
        kz=kslot.get(r['slot']);pred['slot50']=[((kz['c'][i] if kz else 0)+50*P0[i])/((kz['n'] if kz else 0)+50) for i in range(5)];pred['slot200']=[((kz['c'][i] if kz else 0)+200*P0[i])/((kz['n'] if kz else 0)+200) for i in range(5)]
        key1=rows[t-1]['k'] if t else None;key2=(rows[t-2]['k'],rows[t-1]['k']) if t>1 else None;pred['ctx1']=[(ctx1[key1][i]+30*P0[i])/(ctx1n[key1]+30) for i in range(5)] if key1 is not None else P0[:];pred['ctx2']=[(ctx2[key2][i]+40*P0[i])/(ctx2n[key2]+40) for i in range(5)] if key2 is not None else P0[:];pred['analog_k']=analog_k(rows,t);hsteps={}
        for h,label in ((.005,'005'),(.02,'02'),(.05,'05'),(.10,'10')):
            pw=[(1-h)*hpost[h][i]+h*GPRIOR[i] for i in range(len(GAMMAS))];pred[f'hmm{label}']=[sum(pw[j]*qgamma(GAMMAS[j])[i] for j in range(len(GAMMAS))) for i in range(5)];hsteps[h]=pw
        base=cat_score(P0,r['k'])
        if t>=burn:
            for name in k_names:add_k(km[name][st],cat_score(pred[name],r['k']),base)
        z=slotstat.get(r['slot'],{'n':0,'A':[0]*20,'B':[0]*20});z['n']+=1
        for x in r['A']:z['A'][x-1]+=1
        for x in r['B']:z['B'][x-1]+=1
        slotstat[r['slot']]=z;update_trans(sameA,oa,prev['A'] if prev else None,r['A']);update_trans(sameB,ob,prev['B'] if prev else None,r['B']);update_trans(crossA,oca,prev['B'] if prev else None,r['A']);update_trans(crossB,ocb,prev['A'] if prev else None,r['B'])
        z=kslot.get(r['slot'],{'n':0,'c':[0]*5});z['n']+=1;z['c'][r['k']]+=1;kslot[r['slot']]=z
        if key1 is not None:ctx1[key1][r['k']]+=1;ctx1n[key1]+=1
        if key2 is not None:ctx2[key2][r['k']]+=1;ctx2n[key2]+=1
        for h,pw in hsteps.items():
            post=[pw[j]*qgamma(GAMMAS[j])[r['k']] for j in range(len(GAMMAS))];zz=sum(post);hpost[h]=[x/zz for x in post]
        z=kp[-1].copy();z[r['k']]+=1;kp.append(z);prev=r
    nrank=[{'name':name,'development':metric_out(nm[name]['development']),'validation':metric_out(nm[name]['validation'])} for name in n_names[1:]];nrank.sort(key=lambda x:x['validation']['meanPseudoLogGainVsUniform'],reverse=True);ns=nrank[0];ngate=ns['development']['meanPseudoLogGainVsUniform']>0 and ns['validation']['cumulativePseudoLogGainVsUniform']>0 and ns['validation']['meanBrierDiffVsUniform']<0;nt=metric_out(nm[ns['name']]['test']);nh=nw(nm[ns['name']]['test']['diffs']);nci=block_ci(nm[ns['name']]['test']['diffs'])
    krank=[{'name':name,'development':kmetric_out(km[name]['development']),'validation':kmetric_out(km[name]['validation'])} for name in k_names[1:]];krank.sort(key=lambda x:x['validation']['meanLogGainVsIID'],reverse=True);ks=krank[0];kgate=ks['development']['meanLogGainVsIID']>0 and ks['validation']['cumulativeLogLRVsIID']>=math.log(5) and ks['validation']['meanBrierDiffVsIID']<0;kt=kmetric_out(km[ks['name']]['test']);kh=nw(km[ks['name']]['test']['diffs']);kci=block_ci(km[ks['name']]['test']['diffs'],seed=260903)
    train=[r for r in rows if 9759<=r['number']<=9808];post=[r for r in rows if 9809<=r['number']<=9858];ext=[r for r in rows if 9809<=r['number']<=10049];blind=[r for r in rows if 7830<=r['number']<=7883];frozenpre=[r for r in rows if r['time']<=FREEZE and r['slot'] in SLOTS];mid=len(frozenpre)//2;testrows=rows[tstart:];signals=scan_repulsion(rows)
    report={'generatedAt':datetime.now().astimezone().isoformat(),'archive':{'source':raw.get('source','unknown') if isinstance(raw,dict) else 'array','retrievedAt':raw.get('retrievedAt') if isinstance(raw,dict) else None,'count':N,'first':rows[0]['number'],'last':rows[-1]['number'],'lastDate':rows[-1]['date'],'continuous':True},'split':{'burn':burn,'development':[rows[burn]['number'],rows[vstart-1]['number']],'validation':[rows[vstart]['number'],rows[tstart-1]['number']],'untouchedTest':[rows[tstart]['number'],rows[-1]['number']]},'frozenConfirmatory':{'freeze':'2026-08-26T14:21:00+03:00','slots':sorted(SLOTS),'q':Q,'gamma':GAMMA,'p0':P0,'n':len(ledger),'cumulativeLogLR':lr,'cumulativeE':exp_safe(lr),'cumulativeDeltaLogloss':lr,'BrierQ':bq,'BrierIID':b0,'BrierDifference':bq-b0,'rows':ledger},'numberSearch':{'candidateCount':len(n_names),'topValidation':nrank[:10],'selectedBeforeTest':ns['name'],'activationGatePassed':ngate,'selectedUntouchedTest':nt,'testHAC':nh,'testBlockCI':nci,'operationalDecision':'activate' if ngate and nt['cumulativePseudoLogGainVsUniform']>0 and nt['meanBrierDiffVsUniform']<0 and nci['low']>0 else 'ABSTAIN'},'kSearch':{'candidateCount':len(k_names),'topValidation':krank[:10],'selectedBeforeTest':ks['name'],'activationGatePassed':kgate,'selectedUntouchedTest':kt,'testHAC':kh,'testBlockCI':kci,'operationalDecision':'activate' if kgate and kt['cumulativeLogLRVsIID']>0 and kt['meanBrierDiffVsIID']<0 and kci['low']>0 else 'IID'},'historical':{'training9759_9808':dist_summary(train),'immediate9809_9858':dist_summary(post),'extended9809_10049':dist_summary(ext),'blind7830_7883':dist_summary(blind),'frozenSlotsPreFreezeAll':dist_summary(frozenpre),'frozenSlotsPreFreezeFirstHalf':dist_summary(frozenpre[:mid]),'frozenSlotsPreFreezeSecondHalf':dist_summary(frozenpre[mid:]),'frozenSlotsBySlot':{s:dist_summary([r for r in frozenpre if r['slot']==s]) for s in sorted(SLOTS)}},'repulsionPersistence':{'signals':signals,'count':len(signals),'aggregateForwardLogLR':sum(x['forwardLogLR'] for x in signals),'topNonOverlapping50Windows':top_windows(rows)},'graph':{'training':graph_stats(train),'immediate':graph_stats(post),'frozenSlotsPreFreeze':graph_stats(frozenpre),'untouchedTest':graph_stats(testrows),'cosTrainImmediate':cosine_graph(train,post),'cosTrainFrozenSlots':cosine_graph(train,frozenpre),'cosTrainTest':cosine_graph(train,testrows)},'lagInformation':{'all':mi_test([r['k'] for r in rows]),'training':mi_test([r['k'] for r in train],300),'frozenSlotsPreFreeze':mi_test([r['k'] for r in frozenpre]),'untouchedTest':mi_test([r['k'] for r in testrows])},'scheduleEras':schedule_eras(rows),'interpretation':{'frozenChanged':False,'manipulationClaim':False,'numberScore':'prequential marginal membership log/Brier score','chaosProbe':'joint-state nearest-neighbour analogue predictor using Johnson/Hamming geometry, past-only'}}
    OUT.mkdir(parents=True,exist_ok=True);json.dump(report,open(OUT/'audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2,default=str)
    with open(OUT/'prospective-ledger.csv','w',encoding='utf-8') as f:
        f.write('draw_id,time_msk,K,e_t,cumulative_logLR,cumulative_E,cumulative_delta_logloss,Brier_q,Brier_IID,Brier_difference\n')
        for x in ledger:f.write(','.join(str(x[k]) for k in ('draw_id','time_msk','K','e_t','cumulative_logLR','cumulative_E','cumulative_delta_logloss','Brier_q','Brier_IID','Brier_difference'))+'\n')
    lines=['# Sportloto 4x20 — frozen + blind predictive audit','',f"Archive #{rows[0]['number']}–#{rows[-1]['number']} (N={N}), retrieved {report['archive']['retrievedAt']}",'','## Frozen confirmatory',f"Eligible N={len(ledger)}; logLR={lr:.6f}; E={exp_safe(lr)}; Brier difference={bq-b0:.6f}. q/gamma/slots/freeze unchanged.",'','## Exact-number predictive search',f"Selected on validation: {ns['name']}; gate={ngate}; untouched-test gain={nt['cumulativePseudoLogGainVsUniform']:.6f}; mean gain={nt['meanPseudoLogGainVsUniform']:.8f}; Brier diff={nt['meanBrierDiffVsUniform']:.8f}; block CI=[{nci['low']:.8f},{nci['high']:.8f}]; decision={report['numberSearch']['operationalDecision']}.",'','## K-regime search',f"Selected on validation: {ks['name']}; gate={kgate}; untouched-test logLR={kt['cumulativeLogLRVsIID']:.6f}; E={kt['E']}; Brier diff={kt['meanBrierDiffVsIID']:.8f}; block CI=[{kci['low']:.8f},{kci['high']:.8f}]; decision={report['kSearch']['operationalDecision']}.",'','## Historical fixed-q controls']
    for n,v in report['historical'].items():
        if isinstance(v,dict) and 'n' in v:lines.append(f"- {n}: N={v['n']}, counts={v['counts']}, meanK={v['meanK']}, logLR={v['fixedQ_logLR']:.6f}, E={v['fixedQ_E']}, BrierDiff={v['meanBrierDiff_q_minus_IID']}")
    lines += ['', '## Persistence scan',f"E>=20 signals from preceding 50: {len(signals)}; aggregate next-50 logLR={report['repulsionPersistence']['aggregateForwardLogLR']:.6f}.",'','## Decision','A pattern is called an algorithm only if it improves untouched chronological data. Otherwise the operational result is abstention/IID. No result is evidence of manipulation or a promise of winnings.']
    (OUT/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'archive':report['archive'],'frozen':{'n':len(ledger),'E':exp_safe(lr)},'number':report['numberSearch'],'k':report['kSearch'],'historical':report['historical'],'persistence':report['repulsionPersistence']},ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
