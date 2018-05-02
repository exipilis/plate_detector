import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from scrapy.selector import Selector
# import re

files = [s for s in os.listdir('pages') if s.endswith('.html')]

# delete files with 0 cars
# grep -l "License plates found <b>0</b>" * | xargs rm


def process(fn):
    res = []
    with open('pages/' + fn) as f:
        text = f.read()
    html = Selector(text=text)

    total_cars = html.css('h1 b::text').extract()
    if len(total_cars):
        total_cars = int(total_cars[0].replace('.', ''))
    else:
        total_cars = 0
        print('no total cars' + fn)
    if total_cars > 1000 and fn.count('_') > 2:
        print('total cars = %s %s ' % (total_cars, fn))

    panels = html.css('.panel.panel-grey')
    # print(len(panels))
    for panel in panels:
        photo_src = panel.css('img.center-block ::attr(src)').extract_first()
        photo_src = photo_src.replace('/m/', '/o/')
        plate = panel.css('.text-center img')
        license_no = plate.css('::attr(alt)').extract_first()
        plate_src = plate.css('::attr(src)').extract_first()
        license_no = license_no.replace(' ', '')
        model_make = panel.css('h4 a::text').extract_first()
        url = panel.css('h4 a::attr(href)').extract_first()
        res.append('%s\t%s\t%s\t%s\t%s\n' % (license_no, url, plate_src, photo_src, model_make))
    # for l in re.findall(r'(/us/nomer\d+)', text):
    #     res.append(l + '\n')
    print(fn)
    return res


pool = ThreadPool(cpu_count())
result = pool.map(process, files)

result = [item for sublist in result for item in sublist]
result = sorted(set(result))

with open('dataset.csv', 'w') as df:
    df.writelines(result)
