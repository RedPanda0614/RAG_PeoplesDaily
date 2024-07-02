import requests
import bs4
import os
import datetime
import time
import json


def fetchUrl(url):
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except requests.exceptions.RequestException as e:
        print(f'Error fetching URL {url}: {e}')
        return None


def getPageList(year, month, day):
    url = f'http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/nbs.D110000renmrb_01.htm'
    html = fetchUrl(url)
    if html is None:
        return []
    bsobj = bs4.BeautifulSoup(html, 'html.parser')
    temp = bsobj.find('div', attrs={'id': 'pageList'})
    if temp:
        pageList = temp.ul.find_all('div', attrs={'class': 'right_title-name'})
    else:
        pageList = bsobj.find('div', attrs={
                              'class': 'swiper-container'}).find_all('div', attrs={'class': 'swiper-slide'})
    linkList = []
    for page in pageList:
        link = page.a["href"]
        url = f'http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/{link}'
        linkList.append(url)
    return linkList


def getTitleList(year, month, day, pageUrl):
    html = fetchUrl(pageUrl)
    if html is None:
        return []
    bsobj = bs4.BeautifulSoup(html, 'html.parser')
    temp = bsobj.find('div', attrs={'id': 'titleList'})
    if temp:
        titleList = temp.ul.find_all('li')
    else:
        titleList = bsobj.find(
            'ul', attrs={'class': 'news-list'}).find_all('li')
    linkList = []
    for title in titleList:
        tempList = title.find_all('a')
        for temp in tempList:
            link = temp["href"]
            if 'nw.D110000renmrb' in link:
                url = f'http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/{link}'
                linkList.append(url)
    return linkList


def getContent(html, url):
    bsobj = bs4.BeautifulSoup(html, 'html.parser')
    try:
        title = bsobj.h3.text + '\n' + bsobj.h1.text + '\n' + bsobj.h2.text + '\n'
        pList = bsobj.find('div', attrs={'id': 'ozoom'}).find_all('p')
        content = ''
        for p in pList:
            content += p.text + '\n'
        resp = {"url": url, "title": title.strip(), "content": content.strip()}
    except AttributeError as e:
        print(f'Error parsing content from {url}: {e}')
        return None
    return resp


def saveJsonFile(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('文章已写入：' + os.path.join(path, filename))


def download_rmrb(year, month, day, destdir, data_dict):
    pageList = getPageList(year, month, day)
    for page in pageList:
        titleList = getTitleList(year, month, day, page)
        for url in titleList:
            html = fetchUrl(url)
            if html is None:
                continue
            content = getContent(html, url)
            if content is None:
                continue
            key = f'{year}{month}'
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(content)


def gen_dates(b_date, days):
    day = datetime.timedelta(days=1)
    for i in range(days):
        yield b_date + day * i


def get_date_list(beginDate, endDate):
    start = datetime.datetime.strptime(beginDate, "%Y%m%d")
    end = datetime.datetime.strptime(endDate, "%Y%m%d")
    data = []
    for d in gen_dates(start, (end-start).days):
        data.append(d)
    return data


if __name__ == '__main__':
    print('---文章爬取系统---')
    beginDate = input('请输入开始日期(格式如20220706):')
    endDate = input('请输入结束日期(格式如20220706):')
    data = get_date_list(beginDate, endDate)
    destdir = "./copus_new"
    data_dict = {}

    for d in data:
        year = str(d.year)
        month = str(d.month) if d.month >= 10 else '0' + str(d.month)
        day = str(d.day) if d.day >= 10 else '0' + str(d.day)
        download_rmrb(year, month, day, destdir, data_dict)
        print(f'爬取文章时间为：{year}/{month}/{day}的文章已成功写入数据字典中！')

    for key, value in data_dict.items():
        year, month = key[:4], key[4:]
        filename = f'{year}-{month}.json'
        saveJsonFile(value, destdir, filename)

    print('---文章爬取系统---')
