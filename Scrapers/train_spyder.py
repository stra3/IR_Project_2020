import scrapy
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import csv
from langdetect import detect
import pickle
import numpy as np

class QuotesSpider(scrapy.Spider):
    name = "text"
    start_urls = [
        'https://ah.nl',
        'https://websitevhjaar.nl',
        'https://thuisarts.nl',
        'https://startpagina.nl',
        'https://volkskrant.nl',
        'https://telegraaf.nl',
        'https://wrts.nl',
        'https://startpagina.nl',
        'https://sitedeals.nl',
        'https://startcenter.nl',
        'https://consumentenbond.nl'
    ]

    def parse(self, response):
        url = response.url

        check, count, url_part = check_url_list(url)

        if check:
            soup = BeautifulSoup(response.body, 'html.parser')

            [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
            visible_text = soup.getText().strip()

            lang, nl = find_lang(soup)
            if nl:
                data = [url, nl, lang]


                self.log(f'site: {url_part}, count:{count}')



                self.log(f'Saved url: {url}')
                filename = 'data_maxrep.csv'
                save_data(filename, data)

                le = LinkExtractor() # empty for getting everything, check different options on documentation
                for link in le.extract_links(response):
                    yield scrapy.Request(link.url, callback=self.parse)

def check_url_list(url):
    url = url.replace("www.", "")
    url_index = url.index('.',1)
    url_part = url[:url_index]
    name = "url_dict"

    dict = np.load(name +'.npy', allow_pickle = True).item()

    if url_part in dict:
        if dict[url_part] >= 100:
            return False, dict[url_part], url_part

        else:
            dict[url_part] = dict[url_part]+1
            np.save(name +'.npy', dict)
            return True, dict[url_part], url_part
    else:
        dict[url_part] = 1
        np.save(name +'.npy', dict)
        return True, dict[url_part], url_part



def find_lang(soup):
    if "lang=" in str(soup):
        lang = soup.html["lang"]
    else:
        lang = detect(visible_text)

    nl = "nl" in lang.lower()

    return lang, nl

def save_data(filename, data):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
