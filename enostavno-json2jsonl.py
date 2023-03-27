import json
import os

import pandas as pd
import re

map_months = {
    'januar': 1,
    'februar': 2,
    'marec': 3,
    'april': 4,
    'maj': 5,
    'junij': 6,
    'julij': 7,
    'avgust': 8,
    'september': 9,
    'oktober': 10,
    'november': 11,
    'december': 12
}


def get_date(metadata):
    if metadata:
        date = metadata[0].split()[:3]
        date = [date[0][:-1], str(map_months[date[1]]), date[2]]  # day, month, year
        date = '-'.join(date)
        return date
    else:
        return None


def get_url(file):
    html_path = file.split('/')[-1].replace('.json', '.html')
    full_path = os.path.join('data/crawl-enostavno-27-02-2023/cache', html_path)
    with open(full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<link rel="canonical" href='):
                href_regex = r'href="([^"]+)"'
                match = re.search(href_regex, line)
                if match is not None:
                    href = match.group(1)
                    return href


docs = []
for file in os.scandir('data/crawl-enostavno-27-02-2023/cache_processed'):
    doc = json.load(open(file.path))

    url = get_url(file.path)
    doc['url'] = url

    date = get_date(doc['metadata'])
    body = " ".join(doc['article'])
    doc['date'] = date
    doc['body'] = body
    del doc['metadata']
    del doc['article']
    docs.append(doc)

df = pd.DataFrame(docs)
df.to_json('output/rtvslo-enostavno.jsonl', lines=True, force_ascii=False, orient='records')
