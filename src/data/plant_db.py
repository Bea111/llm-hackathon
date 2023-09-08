from urllib.request import urlopen

url = "https://www.patchplants.com/gb/en/read/plant-care/how-to-care-for-houseplants-in-spring/"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html) 

