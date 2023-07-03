import requests

from bs4 import BeautifulSoup

import csv

url = 'http://www.ace.ucv.ro'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

excluded = ['.jpg', 'http']
def getLinksRec(link, allLinks, writer):
  lastLevel = ['http', '.pdf', '.tgz', '.rar', '.png', '.gif', '.doc', '.zip']

  if all([x not in link for x in lastLevel]):
    newUrl = url+'/'+link
    if(link.startswith('/')):
       newUrl = url+link
    pageRec = requests.get(newUrl)
    soupRec = BeautifulSoup(pageRec.content, 'html.parser')
    links = soupRec.find_all('a')
    if len(links) > 0:
      for linkVar in links:
        href = linkVar.get('href')
        if href is not None and href not in allLinks and all([x not in href for x in excluded]):
          print(href)
          writer.writerow([href])
          allLinks.append(href)
          getLinksRec(href, allLinks, writer)

header = ['link']
allLinks = []
with open('links.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for link in soup.find_all('a'):
        href = link.get('href')
        if all([x not in href for x in excluded]):
          print(href)
          writer.writerow([href])
          allLinks.append(href)
          getLinksRec(href, allLinks, writer)