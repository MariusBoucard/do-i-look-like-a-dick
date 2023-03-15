import requests
from bs4 import BeautifulSoup
import os

import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# Configurer le driver Selenium
url = 'https://www.thisdickpicdoesnotexist.com/'

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
# Configurer le service ChromeDriver
# service = Service(executable_path=ChromeDriverManager().install())

# Configurer le driver Selenium

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('/home/marius/Documents/chromedriver/chromedriver',chrome_options=chrome_options)

# Ouvrir l'URL de la page à scraper
# driver.get(url)

# Trouver le div souhaité et accéder à son contenu
# div = driver.find_element(By.CSS_SELECTOR, 'div.my-class')
# content = div.get_attribute('innerHTML')
print(driver.get('https://www.google.nl/'))

# Fermer le navigateur web
driver.quit()


response = requests.get(url)
# print(response.text)
soup = BeautifulSoup(response.text, 'html.parser')
# print(soup)
div = soup.find_all('div', {'class': 'v-image'})
attr1 = div[0]['style']
# print(div)
for teub in div :

    try:
        style = teub['style']
        bybys = re.search('url\((.*?)\)', style).group(1)
        print(bybys)
    except:
       pass

# print(url)
# for img in img_tags:
#     img_url = img['src']
#     filename = os.path.join('images', os.path.basename(img_url))
#     img_data = requests.get(img_url).content
#     with open(filename, 'wb') as handler:
#         handler.write(img_data)
