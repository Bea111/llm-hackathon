# https://docs.scrapy.org/en/latest/intro/tutorial.html#creating-a-project
from pathlib import Path

import scrapy
import json


class GWSpider(scrapy.Spider):
    name = "gw" # gardeners world

 

    def start_requests(self):
        urls = [
            "https://www.gardenersworld.com/how-to/grow-plants/10-gardening-projects-for-kids/",
            "https://www.gardenersworld.com/how-to/how-to-grow-hyacinths/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        article_dict={"title":[], "text:":[]}

        page = response.url.split("/")[-2]
        filename = f"gardeners_world-{page}.html"
        title=response.xpath("//title/text()").get()
        text=response.xpath('//div/p/descendant-or-self::*/text()').extract()
        article_dict={"title":[], "text":[]}
        article_dict["title"].append(title)
        article_dict["text"].append(text)
        bytes_dict = json.dumps(article_dict).encode('utf-8')

        Path(filename).write_bytes(bytes_dict)
        self.log(f"Saved file {filename}")


