import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
from selenium.webdriver.chrome.service import Service
import requests
from bs4 import BeautifulSoup

chrome_driver_path = "D:/Kyron/Downloads/chromedriver_win32/chromedriver.exe"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)
url =url = 'https://alternative.me/crypto/fear-and-greed-index'
driver.get('https://wikipedia.de')


url = 'https://alternative.me/crypto/fear-and-greed-index'
l1 = requests.get(url).content
soup = BeautifulSoup(l1, 'html.parser')

#//*[@id="main"]/section/div/div[3]/div[2]/div/div/div[1]/div[2]/div
element = soup.find('div', attrs={'id': 'main'})
result = soup.select_one(
    '#main section div:nth-of-type(3) div:nth-of-type(2) div div div div:nth-of-type(1) div:nth-of-type(2) div')

result = element.select_one('section > div:nth-of-type(3) > div:nth-of-type(2) > div > div > div > div:nth-of-type(1) > div:nth-of-type(2) > div')
result = element.select_one('section > div:nth-of-type(3) > div:nth-of-type(2) > div > div > div > div:nth-of-type(1) > div:nth-of-type(2) > div')
print(result)
title = soup.body.find(class="fng-circle").get_text()
print(title)
print(l1)