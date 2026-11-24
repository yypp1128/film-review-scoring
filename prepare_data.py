"""Download a fixed research-only sample using GitHub API; retain provenance."""
import argparse
import base64
import csv
import hashlib
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
REPO='bbublm/Douban-Movie-Review-Dataset'
COMMIT='83ce1656fc1ad7e61479d40178f7cd05f06a884b'

def get_json(url):
    response=requests.get(url, timeout=(15, 60), headers={'Accept':'application/vnd.github+json'})
    response.raise_for_status()
    return response.json()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--movies',type=int,default=40)
    parser.add_argument('--expanded', action='store_true')
    args=parser.parse_args()
    dest=ROOT/'data'; dest.mkdir(exist_ok=True)
    cache=dest/'cache'; cache.mkdir(exist_ok=True)
    tree_path=cache/'tree.json'
    if tree_path.exists():
        tree=json.loads(tree_path.read_text(encoding='utf-8'))
    else:
        tree=get_json(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1')
        tree_path.write_text(json.dumps(tree,ensure_ascii=False),encoding='utf-8')
    files=[x for x in tree['tree'] if x['path'].startswith('data_all/')
           and x['path'].endswith('.csv') and (args.expanded or 20000 <= x.get('size',0) <= 150000)]
    random.Random(42).shuffle(files)
    if not args.expanded:
        files=files[:args.movies]
    if not args.expanded and len(files)<args.movies:
        raise ValueError('没有足够的候选作品文件。')

    def fetch(item):
        path=cache/(item['sha']+'.csv')
        if path.exists():
            content=path.read_bytes()
        else:
            if args.expanded:
                url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/'+quote(item['path'])
                for attempt in range(3):
                    try:
                        response=requests.get(url, timeout=(10, 30))
                        response.raise_for_status()
                        content=response.content
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise
            else:
                content=base64.b64decode(get_json(item['url'])['content'])
            path.write_bytes(content)
        # Git blob checksum also verifies cached downloads against the fixed tree.
        digest=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if digest != item['sha']:
            raise ValueError('下载文件校验失败。')
        movie=Path(item['path']).stem
        movie_id=hashlib.sha256(item['path'].encode()).hexdigest()[:16]
        rows=[]
        for row in csv.reader(io.StringIO(content.decode('utf-8-sig'))):
            if len(row)!=7:
                continue
            # Original: date,user,text,star,star_description,votes,precomputed_sentiment.
            # User identity and precomputed sentiment are intentionally discarded.
            rows.append({'movie_id':movie_id,'movie_name':movie,'text':row[2],
                         'star':row[3],'date':row[0],'source':REPO})
        return rows,{'path':item['path'],'sha':item['sha'],'rows':len(rows)}

    rows=[]; sources=[]; failures=[]
    with ThreadPoolExecutor(max_workers=16 if args.expanded else 3) as pool:
        jobs={pool.submit(fetch,item):item for item in files}
        for job in as_completed(jobs):
            try:
                batch,source=job.result(); rows.extend(batch); sources.append(source)
                print(f"Downloaded {len(sources)}/{len(files)}; rows={len(rows)}",flush=True)
            except Exception as exc:
                failures.append({'path':jobs[job]['path'],'error':str(exc)})
    # Stable order makes the resulting CSV reproducible despite concurrent requests.
    rows.sort(key=lambda x:(x['movie_id'],x['date'],x['text']))
    if rows and not failures:
        with (dest/('expanded_reviews.csv' if args.expanded else 'reviews.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest={'repository':f'https://github.com/{REPO}','commit':COMMIT,
              'selection_seed':42,'requested_movies':len(files),
              'usage':'发布者说明仅供学术研究和个人学习，禁止商业使用；不代表平台额外授权。',
              'sources':sorted(sources,key=lambda x:x['path']),'failures':failures,
              'original_columns_used':['text','star','date'],
              'excluded_columns':['username','precomputed_sentiment','votes','star_description']}
    (dest/('expanded_provenance.json' if args.expanded else 'provenance.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} files failed; successful downloads cached. Rerun to complete.')

if __name__=='__main__': main()

