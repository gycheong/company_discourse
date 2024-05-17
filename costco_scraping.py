from selenium import webdriver  # seems like one needs to pip install selenium to have it work in an IDE
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def get_reviews_from_soup(soup):
    reviews_container = soup.find_all("ol", attrs={"data-bv-v": "contentItemCollection:2"})[0]

    # authors_container = reviews_container.find_all('span', attrs={'class': 'bv-author'})
    # dates_container = reviews_container.find_all('span', attrs={'class': 'bv-content-datetime-stamp'})
    # scores_container = reviews_container.find_all('span', attrs={'itemprop': 'reviewRating'})
    # texts_container = reviews_container.find_all('div', attrs={'class': 'bv-content-summary-body-text'})
    # helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-yes'})
    # not_helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-no'})
    #
    # if not(len(authors_container) == len(dates_container) == len(scores_container) == len(texts_container) == len(helpful_container) == len(not_helpful_container)):
    #     print('Error:')
    #     print(str(len(authors_container)) + ' authors.')
    #     print(str(len(dates_container)) + ' dates.')
    #     print(str(len(scores_container)) + ' scores.')
    #     print(str(len(texts_container)) + ' texts.')
    #     print(str(len(helpful_container)) + ' helpful votes.')
    #     print(str(len(not_helpful_container)) + ' not helpful votes.')

    # n = len(authors_container)
    authors, dates, scores, texts, helpful, not_helpful = [], [], [], [], [], []

    reviews_list = reviews_container.find_all('li', recursive=False)
    for item in reviews_list:
        core = item.find('div', attrs={'class': 'bv-content-core'})
        feedback = item.find('div', attrs={'class': 'bv-active-feedback'})
        authors.append(core.find('div', attrs={'class': 'bv-content-meta'}).find('span', attrs={'class': 'bv-author'}).find('span', attrs={'itemprop': 'name'}).text)
        dates.append(core.find('div', attrs={'class': 'bv-content-meta'}).find('div', attrs={'class': 'bv-content-datetime'}).find('span', attrs={'class': 'bv-content-datetime-stamp'}).text.removesuffix(' \xa0'))
        scores.append(core.find('span', attrs={'class': 'bv-content-rating'}).find('meta', attrs={'itemprop': 'ratingValue'}).get('content', default=None))

        t_container = core.find('div', attrs={'class': 'bv-content-summary-body-text'}).find_all('p')
        t = ''
        for j in range(len(t_container)):
            t = t + t_container[j].text + ' '
        texts.append(t)

        helpful.append(feedback.find('button', attrs={'class': 'bv-content-btn-feedback-yes'}).find('span', attrs='bv-content-btn-count').text)
        not_helpful.append(feedback.find('button', attrs={'class': 'bv-content-btn-feedback-no'}).find('span', attrs='bv-content-btn-count').text)

    # for i in range(n):
    #     authors.append(authors_container[i].find('span', attrs={'itemprop': 'name'}).text)
    #     dates.append(dates_container[i].text)
    #     scores.append(scores_container[i].find('meta', attrs={'itemprop': 'ratingValue'}).get('content', default=None))
    #
    #     container = texts_container[i].find_all('p')
    #     t = ''
    #     for j in range(len(container)):
    #         t = t + container[j].text + ' '
    #     texts.append(t)
    #
    #     helpful.append(helpful_container[i].find('span', attrs='bv-content-btn-count').text)
    #     not_helpful.append(not_helpful_container[i].find('span', attrs='bv-content-btn-count').text)

    return authors, dates, scores, texts, helpful, not_helpful


def get_reviews_from_url(url):
    all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful = [], [], [], [], [], []

    more_selector = "#nav-pdp-tab-header-13 > div.row.view-more-v2 > div > input"

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

    page_exists = True

    while page_exists:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        authors, dates, scores, texts, helpful, not_helpful = get_reviews_from_soup(soup)

        all_authors = all_authors + authors
        all_dates = all_dates + dates
        all_scores = all_scores + scores
        all_texts = all_texts + texts
        all_helpful = all_helpful + helpful
        all_not_helpful = all_not_helpful + not_helpful

        # Finding page buttons
        pagination = soup.find('ul', attrs={'class': 'bv-content-pagination-buttons'})
        if pagination:
            next_page = pagination.find_all('li')[1]
            if next_page.find('a'):
                next_selector = '#BVRRContainer > div > div > div > div > div.bv-content-pagination > div > ul > li.bv-content-pagination-buttons-item.bv-content-pagination-buttons-item-next > a'
                next_clickable = EC.element_to_be_clickable((By.CSS_SELECTOR, next_selector))
                WebDriverWait(driver, timeout).until(next_clickable)
                next_button = driver.find_element(By.CSS_SELECTOR, next_selector)
                next_button.click()
            else:
                page_exists = False
        else:
            page_exists = False

    return all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful

url = 'https://www.costco.com/macbook-air-laptop-13.6-inch---apple-m2-chip%2c-8-core-cpu%2c-8-core-gpu%2c-8gb-memory%2c-256gb-ssd-storage.product.100713212.html'
all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful = get_reviews_from_url(url)