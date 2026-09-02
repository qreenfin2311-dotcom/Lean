#!/usr/bin/env python3
import json,math,urllib.parse,urllib.request
from collections import Counter
from pathlib import Path
E='https://www.stoloto.ru/p/api/mobile/api/v35/service/draws/archive';H={'accept':'*/*','device-platform':'DESKTOP','device-type':'STOLOTO','gosloto-partner':'bXMjXFRXZ3coWXh6R3s1NTdUX3dnWlBMLUxmdg','referer':'https://www.stoloto.ru/oxota/archive','user-agent':'Mozilla/5.0 LotoOS/1.0'}
P=[0.375644995,0.462332301,0.148606811,0.013209494,0.000206398];Q=[0.553487068,0.376364933,0.066837266,0.003282397,0.000028336]
rows=[]
for page in range(1,21):
 u=E+'?'+urllib.parse.urlencode({'game':'oxota','count':50,'page':page});d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=20))
 for x in d.get('draws',[]):
  s=((x.get('combination') or {}).get('structured'))
  if x.get('status')=='COMPLETED' and isinstance(s,list) and len(s)==8:
   a=set(map(int,s[:4]));b=set(map(int,s[4:]));
   if len(a)==4 and len(b)==4:rows.append((int(x['number']),x['date'],len(a&b)))
if len(rows)<900:raise SystemExit(f'valid={len(rows)}')
# API returns newest first; sort chronologically
rows=sorted({r[0]:r for r in rows}.values())
n=len(rows);c=[0]*5
for _,_,k in rows:c[k]+=1
mu=sum(k*P[k] for k in range(5));mean=sum(k*c[k] for k in range(5))/n;var=sum((k-mu)**2*P[k] for k in range(5));z=(mean-mu)/math.sqrt(var/n)
lr=sum(c[k]*math.log(Q[k]/P[k]) for k in range(5));pear=sum((c[k]-n*P[k])**2/(n*P[k]) for k in range(5));g2=2*sum(c[k]*math.log(c[k]/(n*P[k])) for k in range(5) if c[k])
bd=0
for k in range(5):
 bq=sum((Q[j]-(j==k))**2 for j in range(5));b0=sum((P[j]-(j==k))**2 for j in range(5));bd+=c[k]*(bq-b0)
bd/=n
parts=[]
for j in range(4):
 rr=rows[j*n//4:(j+1)*n//4];cc=Counter(k for _,_,k in rr);nn=len(rr);ll=sum(cc[k]*math.log(Q[k]/P[k]) for k in range(5));parts.append({'part':j+1,'n':nn,'first':rr[0][0],'last':rr[-1][0],'counts':[cc[k] for k in range(5)],'meanK':sum(k*cc[k] for k in range(5))/nn,'logLR':ll,'E':math.exp(ll) if ll>-745 else 0})
r={'source':'official Stoloto API','lottery':'Охота','n':n,'firstDraw':rows[0][0],'firstDate':rows[0][1],'lastDraw':rows[-1][0],'lastDate':rows[-1][1],'counts':c,'meanK':mean,'iidMean':mu,'zMean':z,'pearson':pear,'g2':g2,'frozenQ_logLR':lr,'frozenQ_E':math.exp(lr) if lr>-745 else 0,'meanBrierDiff_q_minus_IID':bd,'quartiles':parts,'note':'Same 4+4 of 20 formula and official GSC control; exact physical RNG identity with Sportloto 4x20 is not established.'}
Path('.research-sportloto/out').mkdir(parents=True,exist_ok=True);json.dump(r,open('.research-sportloto/out/ohota-1000-control.json','w'),ensure_ascii=False,indent=2);print(json.dumps(r,ensure_ascii=False))
