from urllib.request import urlopen

url_1 = "https://www.patchplants.com/gb/en/read/plant-care/how-to-care-for-houseplants-in-spring/"
url_2 = "https://www.botanopia.com/knowledge-base-articles/"
url_3 = "https://www.gardenersworld.com/how-to/"
page = urlopen(url_1)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html)
