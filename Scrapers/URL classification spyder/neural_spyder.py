import scrapy
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import csv
import pickle
import numpy as np


class QuotesSpider(scrapy.Spider):
    name = "neural"
    start_urls = [
        'https://startpagina.nl'
    ]

    def parse(self, response):
        save_count('neural_nl_count')
        url = response.url


        if nl:
            data = [url]

            self.log(f'Saved url: {url}')
            filename = 'neural_nl.csv'
            save_data(filename, data)

            le = LinkExtractor() # empty for getting everything, check different options on documentation
            for link in le.extract_links(response):
                yield scrapy.Request(link.url, callback=self.parse)


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
