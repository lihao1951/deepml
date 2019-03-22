#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/15
"""
from urllib import request
from urllib import response
import urllib
import json
from golaxy_nlp.dataload import clean_sentence
SIZE = 500
START=10000
FINAL_NUM = 1100000
HOTPOT_URL = 'http://10.1.101.55:9200/base_hotpot/hotpot_type/_search?pretty'
DETAIL_URL = 'http://10.1.101.55:9200/base_hotpot_detail/detail_type/_search?pretty'
def get_detail(key):
    params = {"query":{"term":{"_key":key}},"_source":["title","cont"]}
    req = request.urlopen(urllib.request.Request(DETAIL_URL,data=json.dumps(params).encode('utf-8')))
    req_json = json.loads(req.read().decode('utf-8'))
    hotpot_array = req_json.get('hits').get('hits')
    hotpot=hotpot_array[0]
    source = hotpot.get('_source')
    title,cont = source.get('title'),source.get('cont')
    return title.strip(),clean_sentence(cont.strip())

out = open('./hotpot','a',encoding='utf-8')

while True:
    START += SIZE
    if START >= FINAL_NUM:
        break
    params = {"from":START,"size":SIZE,"sort":{"event_time":"asc"},"_source":["newsdocs","peakhv","normpeakhv"]}
    req = request.urlopen(urllib.request.Request(HOTPOT_URL,data=json.dumps(params).encode('utf-8')))
    req_json = json.loads(req.read().decode('utf-8'))
    hotpot_array = req_json.get('hits').get('hits')
    print('running with : ',START)
    for hotpot in hotpot_array:
        try:
            source = hotpot.get('_source')
            news,peakhv,normpeakhv = source.get('newsdocs')[0],source.get('peakhv'),source.get('normpeakhv')
            if peakhv <= 5:
                continue
            key = news[2:]
            title,cont = get_detail(key)
            result = '{}\t{}\t{}\t{}\t{}\n'.format(news,peakhv,normpeakhv,title,cont)
            out.write(result)
        except:
            continue
out.close()


