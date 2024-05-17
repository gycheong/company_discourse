import bs4
from selenium import webdriver  # seems like one needs to pip install selenium to have it work in an IDE
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests

more_selector = "#nav-pdp-tab-header-13 > div.row.view-more-v2 > div > input"

# url = 'https://www.costco.com/.product.100797736.html'
driver = webdriver.Firefox()
driver.implicitly_wait(30)
driver.get(url)

timeout = 10
try:
    more_clickable = EC.element_to_be_clickable((By.CSS_SELECTOR, more_selector))
    WebDriverWait(driver, timeout).until(more_clickable)
except TimeoutException:
    print("Timed out waiting for page to load")

more_button = driver.find_element(By.CSS_SELECTOR, more_selector)
more_button.click()  # click load more button

# headers = {"User-Agent":
#            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0"
#            }
# response = requests.get(url='https://www.costco.com/.product.100797736.html', headers=headers, timeout=10)
#
# soup = BeautifulSoup(response.text, 'html.parser')

soup = BeautifulSoup(driver.page_source, 'html.parser')
reviews_container = soup.find_all("ol", attrs={"data-bv-v": "contentItemCollection:2"})[0]

# Getting authors
authors_container = reviews_container.find_all('button', attrs={'itemprop': 'author'})
authors = []
for i in range(len(authors_container)):
    authors.append(authors_container[i].find('span').text)

print(authors)


# Getting dates
dates_container = reviews_container.find_all('span', attrs={'class': 'bv-content-datetime-stamp'})
dates = []
for i in range(len(dates_container)):
    dates.append(dates_container[i].text)

print(dates)

# Getting scores
scores_container = reviews_container.find_all('span', attrs={'itemprop': 'reviewRating'})
scores = []
for i in range(len(scores_container)):
    scores.append(scores_container[i].find('meta', attrs={'itemprop': 'ratingValue'}).get('content',default=None))

print(scores)


# Getting review text
texts_container = reviews_container.find_all('div', attrs={'class': 'bv-content-summary-body-text'})
texts = []
for i in range(len(texts_container)):
    container = texts_container[i].find_all('p')
    t = ''
    for j in range(len(container)):
        t = t + container[j].text + ' '
    texts.append(t)

print(texts)


# Finding number of people who found review helpful
helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-yes'})
not_helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-no'})
helpful = []
not_helpful = []
for i in range(len(helpful_container)):
    helpful.append(helpful_container[i].find('span', attrs='bv-content-btn-count').text)

for i in range(len(not_helpful_container)):
    not_helpful.append(not_helpful_container[i].find('span', attrs='bv-content-btn-count').text)

print(helpful)
print(not_helpful)

# Finding page buttons
pagination = soup.find('ul', attrs={'class': 'bv-content-pagination-buttons'})
next_page = pagination.find_all('li')[1]

# while next_page.find('a'):
#     next_selector = '#BVRRContainer > div > div > div > div > div.bv-content-pagination > div > ul > li.bv-content-pagination-buttons-item.bv-content-pagination-buttons-item-next > a'
#     next_clickable = EC.element_to_be_clickable((By.CSS_SELECTOR, next_selector))
#     WebDriverWait(driver, timeout).until(next_clickable)
#     next_button = driver.find_element(By.CSS_SELECTOR, next_selector)
#     next_button.click()