# -*- coding: utf-8 -*-
import json

import scrapy


class PagesSpider(scrapy.Spider):
    name = 'pages'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.states = json.load(open('states.json'))
        self.makes = json.load(open('makes.json'))

    # all
    def start_requests1(self):
        for state_id in self.states.keys():
            for ctype in self.states[state_id]['ctype']:
                url = 'http://platesmania.com/us/gallery.php?region=%s&ctype=%s' % (state_id, ctype)
                meta = {'state_id': state_id, 'ctype': ctype, 'make_id': '', 'page': ''}
                yield scrapy.Request(url, meta=meta)

    # region x ctype with more than 1000 cars
    def start_requests(self):
        popular_pages = [
            {'ctype': 142, 'state_id': 7548},
            {'ctype': 14, 'state_id': 7543},
            {'ctype': 11, 'state_id': 7543},
            {'ctype': 71, 'state_id': 7535},
            {'ctype': 42, 'state_id': 7514},
            {'ctype': 3, 'state_id': 7505}
        ]
        for p in popular_pages:
            for make_id in self.makes.keys():
                url = 'http://platesmania.com/us/gallery.php?region=%s&ctype=%s&markaavto=%s' % \
                      (p['state_id'], p['ctype'], make_id)
                meta = {'state_id': p['state_id'], 'ctype': p['ctype'], 'make_id': make_id, 'page': ''}
                yield scrapy.Request(url, meta=meta)

    def parse(self, response):
        html = response.body.decode('utf-8')

        state_id = response.meta['state_id']
        ctype = response.meta['ctype']
        make_id = response.meta['make_id']
        page = response.meta['page']

        total_cars = response.css('h1 b::text').extract()
        total_cars = int(total_cars[0].replace('.', ''))
        if total_cars:
            with open('pages/%s_%s_%s_%s.html' % (state_id, ctype, make_id, page), 'w') as f:
                f.write(html)

            max_page = 2 + min(1000, total_cars) // 10
            for p in range(0, max_page):
                url = 'http://platesmania.com/us/gallery.php?region=%s&ctype=%s&markaavto=%s&start=%s' % \
                      (state_id, ctype, make_id, p)
                meta = {'state_id': state_id, 'ctype': ctype, 'make_id': make_id, 'page': p}
                yield response.follow(url, meta=meta)

            print('total cars %s, pages %s' % (total_cars, max_page))
