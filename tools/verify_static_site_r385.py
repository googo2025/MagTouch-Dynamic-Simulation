"""Standard-library-only static path audit; safe for GitHub Actions."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json, re
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'web/magnetic-forward-r361'

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.links.append(v)

def check():
    problems=[];checked=0;external=[]
    for p in SITE.rglob('*.html'):
        doc=Links();doc.feed(p.read_text(encoding='utf8'))
        for url in doc.links:
            v=urlsplit(url)
            if v.scheme or v.netloc:external.append(url);continue
            if not v.path:continue
            target=(p.parent/unquote(v.path)).resolve()
            if v.path.startswith('/') or not target.is_relative_to(SITE.resolve()):
                problems.append({'page':p.relative_to(SITE).as_posix(),'url':url,'error':'not project-subpath safe'})
            elif not target.is_file():problems.append({'page':p.relative_to(SITE).as_posix(),'url':url,'error':'missing file'})
            checked+=1
    # All records referenced by the dynamic experiment selector are bundled.
    catalog=SITE/'lab-r368/catalog.js'
    ids=set(re.findall(r'"id"\s*:\s*"([^"]+)"',catalog.read_text(encoding='utf8')))
    for id in ids:
        p=SITE/'lab-r368'/(id+'.json')
        if not p.is_file():problems.append({'id':id,'error':'missing dynamic record'})
    total=sum(p.stat().st_size for p in SITE.rglob('*') if p.is_file())
    assert total < 1_000_000_000, 'Site exceeds conservative 1 GB limit'
    assert not problems,json.dumps(problems,ensure_ascii=False,indent=2)
    return {'status':'PASS','local_links_checked':checked,'dynamic_records':len(ids),'website_bytes':total,
            'external_links':sorted(set(external)),'upload_or_hardware':False}
if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
