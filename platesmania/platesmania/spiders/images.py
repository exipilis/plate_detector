# -*- coding: utf-8 -*-
import csv
import os

import scrapy


class ImagesSpider(scrapy.Spider):
    name = 'images'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = csv.reader(open('dataset.csv'), delimiter='\t')

    def start_requests(self):
        h = {
            'referer': '{http://platesmania.com/us/',
            'cookie': '__cfduid=dc0b45b5a3bfd086bf70e8ea5e7a75fc11524670865; ' +
                      '__utmc=228771677; ' +
                      '__utmz=228771677.1524670866.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); '
                      'PHPSESSID=5kn2q77ca9pj5qp8i456lutkf7; '
                      '_ym_uid=1524678539533483902; lang=en; '
                      '__utma=228771677.2028882421.1524670866.1524922383.1524936816.11; '
                      '__lx211338_load_cnt=1919; __lx211338_load_tmr=1524938555694; '
                      '__lx211338_load_tmr_pre=1524938555780; __lx208261_load_cnt=469; '
                      '__lx208261_load_tmr=1524938537143; '
                      '__lx208261_load_tmr_pre=1524938556032'}
        for l in self.reader:
            url = l[2]
            fn = 'images/' + url.replace('http://', '').replace('https://', '')

            if not os.path.isfile(fn):
                d = os.path.dirname(fn)
                os.makedirs(d, exist_ok=True)
                yield scrapy.Request(url, headers=h, meta={'fn': fn})

            url = l[3]
            fn = 'images/' + url.replace('http://', '').replace('https://', '')

            if not os.path.isfile(fn):
                d = os.path.dirname(fn)
                os.makedirs(d, exist_ok=True)
                yield scrapy.Request(url, headers=h, meta={'fn': fn})

    def parse(self, response):
        with open(response.meta['fn'], 'wb') as f:
            f.write(response.body)
