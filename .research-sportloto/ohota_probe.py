#!/usr/bin/env python3
import json, urllib.parse, urllib.request
from pathlib import Path
ENDPOINT='https://www.stoloto.ru/p/api/mobile/api/v35/service/draws/archive'
HEADERS={'accept':'*/*','device-platform':'DESKTOP','device-type':'STOLOTO','gosloto-partner':'bXMjXFRXZ3coWXh6R3s1NTdUX3dnWlBMLUxmdg','referer':'https://www.stoloto.ru/oxota/archive','user-agent':'Mozilla/5.0 LotoOS-Ohota-Probe/1.0'}
OUT=Path('.research-sportloto/out');OUT.mkdir(parents=True,exist_ok=True)
res={}
for cand in ['oxota','ohota','hunt','hunting','4x20hunt','4x20-ohota','ohota4x20']:
    try:
        url=ENDPOINT+'?'+urllib.parse.urlencode({'game':cand,'count':5,'page':1})
        with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=20) as r:
            d=json.load(r)
        draws=d.get('draws') if isinstance(d,dict) else None
        res[cand]={'requestStatus':d.get('requestStatus') if isinstance(d,dict) else None,'keys':list(d) if isinstance(d,dict) else None,'n':len(draws) if isinstance(draws,list) else None,'draws':draws[:2] if isinstance(draws,list) else draws}
    except Exception as e:res[cand]={'error':repr(e)}
json.dump(res,open(OUT/'ohota-probe.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='draws'} for k,v in res.items()},ensure_ascii=False))
