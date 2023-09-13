from urllib.request import urlopen

url = "https://www.allrecipes.com/recipe/212498/easy-chicken-and-broccoli-alfredo/"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html)
