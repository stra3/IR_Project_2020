import scrapy
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import csv
import pickle
import numpy as np


class QuotesSpider(scrapy.Spider):
    name = "download"
    start_urls = [
        'https://startpagina.nl'
    ]

    def parse(self, response):
        save_count('download_nl_count')
        url = response.url

        soup = BeautifulSoup(response.body, 'html.parser')

        [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
        visible_text = soup.getText().strip()

        lang, nl = find_lang(soup)
        if nl:
            data = [url]

            self.log(f'Saved url: {url}')
            filename = 'download_nl.csv'
            save_data(filename, data)

            le = LinkExtractor() # empty for getting everything, check different options on documentation
            for link in le.extract_links(response):
                yield scrapy.Request(link.url, callback=self.parse)


def find_lang(soup):
    if "lang=" in str(soup):
        lang = soup.html["lang"]
    else:
        lang = 'unknown'

    nl = "nl" in lang.lower()

    return lang, nl

def save_data(filename, data):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def save_count(filename):
        f = open(filename, "r")
        count = int(f.read())

        f = open(filename, "w")
        f.write(str(count+1))
        f.close()
