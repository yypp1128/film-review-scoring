"""Optional single-page Douban parser. No login, proxies or challenge bypass."""
import argparse
import csv
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

def parse_comments(content,movie_id,movie_name):
    soup=BeautifulSoup(content,'html.parser'); result=[]
    for item in soup.select('.comment-item'):
        text=item.select_one('.short')
        rating=item.select_one('[class*="allstar"]')
        classes=' '.join(rating.get('class',[])) if rating else ''
        match=re.search(r'allstar([1-5])0\b',classes)
        if text:
            result.append({'movie_id':movie_id,'movie_name':movie_name,
                           'text':text.get_text(' ',strip=True),'star':int(match[1]) if match else '',
                           'source':'douban_single_page'})
    return result

def main():
    parser=argparse.ArgumentParser(description='仅访问一页；请先确认当前平台规则和使用条件。')
    parser.add_argument('--url',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--output',type=Path,default=Path('data/collected_page.csv'))
    args=parser.parse_args()
    url=urlparse(args.url)
    match=re.fullmatch(r'/subject/(\d+)/comments/?',url.path)
    if url.scheme!='https' or url.hostname!='movie.douban.com' or not match:
        raise SystemExit('仅支持 https://movie.douban.com/subject/数字/comments 路径。')
    session=requests.Session(); session.headers['User-Agent']='MovieReviewPractice/0.1 (academic single-page collector)'
    robots=session.get('https://movie.douban.com/robots.txt',timeout=20)
    if robots.status_code!=200:
        raise SystemExit('无法确认 robots 规则，停止采集。')
    rp=RobotFileParser(); rp.parse(robots.text.splitlines())
    if not rp.can_fetch('MovieReviewPractice',args.url):
        raise SystemExit('当前 robots 规则不允许访问，停止采集。')
    response=session.get(args.url,timeout=20,allow_redirects=False)
    if response.status_code!=200:
        raise SystemExit(f'访问状态 {response.status_code}，停止，不重试或绕过。')
    if any(word in response.text for word in ['验证码','异常请求','检测到有异常请求']):
        raise SystemExit('遇到访问验证，停止采集。')
    rows=parse_comments(response.text,match[1],args.name)
    if not rows: raise SystemExit('未找到评论，可能页面变化或访问受限；停止采集。')
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with args.output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    print(f'已保存 {len(rows)} 条评论，未自动翻页。')

if __name__=='__main__':main()
