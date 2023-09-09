from urllib.request import urlopen

url_1 = "https://www.patchplants.com/gb/en/read/plant-care/how-to-care-for-houseplants-in-spring/"
url_2='https://www.botanopia.com/knowledge-base-articles/'
url_3='https://www.gardenersworld.com/how-to/'
url_4='https://www.gardenersworld.com/how-to/grow-plants/10-gardening-projects-for-kids/'
page = urlopen(url_4)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html) 

