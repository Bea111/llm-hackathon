# https://docs.scrapy.org/en/latest/intro/tutorial.html#creating-a-project
from pathlib import Path

import scrapy


class QuotesSpider(scrapy.Spider):
    name = "gw" # gardeners world

    def start_requests(self):
        urls = [
            "https://www.gardenersworld.com/how-to/grow-plants/10-gardening-projects-for-kids/",
            "https://www.gardenersworld.com/how-to/how-to-grow-hyacinths/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"gardeners_world-{page}.html"
        Path(filename).write_bytes(response.body)
        title=response.xpath("//title/text()").get()
        text=response.css("div.post__content")
        self.log(f"Saved file {filename}")


