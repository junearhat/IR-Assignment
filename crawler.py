import json
import os
import random
import time
from collections import deque
from urllib.parse import urljoin
from urllib import robotparser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, Tag

# Suppress unnecessary logging from Selenium and ChromeDriver
import logging
logging.getLogger('selenium').setLevel(logging.WARNING)
os.environ['WDM_LOG_LEVEL'] = '0'

# Configuration for th web crawler
COVENTRY_PUREPORTAL_URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/publications"
BASE_URL = "https://pureportal.coventry.ac.uk"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
MAX_RETRIES = 3
PAGE_TIMEOUT = 30  # seconds
CRAWLED_DATA_FILE = "coventry_publications.json"
POLITE_DELAY = 3  # Reduced default delay, will be overridden by robots.txt if needed

# function to Fetch author name and profile url from publication url
def fetch_publication_author(soup, base_url):  
    authors_data = []
     # Find the paragraph element that holds author information
    persons_p = soup.select_one('p.relations.persons')
    if not persons_p:
        return []
    # Iterate through the elements within the authors' paragraph
    for element in persons_p.contents:
        if isinstance(element, Tag) and element.name == 'a':
            # If the element is an anchor tag, it's a linked author
            name = element.get_text(strip=True)
            url = urljoin(base_url, str(element.get('href', '')))
            if name:
                authors_data.append({'name': name, 'url': url})
        elif isinstance(element, str):
            # If it's just text, it might contain unlinked author names (e.g., external collaborators)
            potential_names = element.split(',')
            for name_part in potential_names:
                clean_name = name_part.strip(' ,')
                if clean_name:
                    authors_data.append({'name': clean_name, 'url': None})
    return authors_data

# Function to Fetch abstract content from publication url
def fetch_publication_abstract(soup):
    # Locate the specific div element containing the abstract content
    abstract_div = soup.find('div', class_='rendering_researchoutput_abstractportal')
    if abstract_div:
        # The actual text is nested inside a 'textblock' div
        text_block = abstract_div.find('div', class_='textblock')
        if text_block:
            return text_block.get_text(strip=True)
    return ''  # Return empty string if abstract is not found

# Main Crawler Function (synchronous using Selenium)
def crawl():
    print("Starting crawler with Selenium...")

    # Setup Chrome driver options for headless and silent operation
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"user-agent={USER_AGENT}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")  # Suppress console logging
    options.add_argument("--silent")
    # Suppress automation messages and logging from Selenium
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)

    # Configure service to suppress driver logs
    service = Service(log_output=os.devnull)
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, PAGE_TIMEOUT)

    rp = robotparser.RobotFileParser()
    robots_url = urljoin(BASE_URL, 'robots.txt')
    print(f"Fetching robots.txt from: {robots_url}")
    
    try:
        driver.get(robots_url)
        robots_page_source = driver.page_source
        soup_robots = BeautifulSoup(robots_page_source, 'html.parser')
        robots_text_value = soup_robots.get_text()

        print("\n--- robots.txt content ---")
        print(robots_text_value)
        print("----------------------------------------------\n")
        
        rp.parse(robots_text_value.splitlines())
        print("robots.txt parsed successfully.")
    except WebDriverException as e:
        print(f"Warning: Could not fetch or parse robots.txt with Selenium. Proceeding with default settings. Error: {e}")
    
    crawl_delay_duration = rp.crawl_delay(USER_AGENT)
    robots_delay_duration = int(crawl_delay_duration) if crawl_delay_duration else None
    user_minimum_delay = POLITE_DELAY
    
    if robots_delay_duration and robots_delay_duration > user_minimum_delay:
        min_effective_delay = robots_delay_duration
        print(f"Using Crawl-Delay from robots.txt: {min_effective_delay} seconds.")
    else:
        min_effective_delay = user_minimum_delay
        print(f"Using default delay: {min_effective_delay} seconds.")

    # Ensure the output directory exists before saving the file
    data_dir = os.path.dirname(CRAWLED_DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    try:
        # PHASE 1: DISCOVER ALL PUBLICATION URLS
        # This phase navigates through all paginated list pages to collect every publication's URL.
        print("\n--- Phase 1: Discovering all publication URLs ---")
        publications_to_scrape = []
        queue = deque([COVENTRY_PUREPORTAL_URL])
        visited_urls = {COVENTRY_PUREPORTAL_URL}

        print("Scanning Pages...")

        while queue:
            current_url = queue.popleft()
            
            if not rp.can_fetch(USER_AGENT, current_url):
                print(f"\nSkipping disallowed URL (from robots.txt): {current_url}")
                continue
            
            success = False
            
            print(f"Visiting: {current_url}")

            for attempt in range(MAX_RETRIES):
                try:
                    driver.get(current_url)

                    # Handle the cookie consent pop-up on the first visit
                    if len(visited_urls) == 1:
                        try:
                            WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
                            ).click()
                        except Exception:
                            pass

                    # Wait for the main publication list to load on the page
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.list-result-item")))
                    success = True
                    break
                except (TimeoutException, WebDriverException) as e:
                    # Retry if the page fails to load or times out
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for {current_url}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

            if not success:
                print(f"All retries failed for {current_url}. Skipping page.")
                continue
            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")
            # Find all publication items on the current page
            pub_list = soup.find_all("li", class_="list-result-item")
            
            print(f"Found {len(pub_list)} publications on this page.")
            for pub_item in pub_list:
                if isinstance(pub_item, Tag):
                    title_tag = pub_item.find("h3", class_="title")
                    if isinstance(title_tag, Tag) and title_tag.a:
                        title = title_tag.get_text(strip=True)
                        pub_url = urljoin(BASE_URL, str(title_tag.a["href"]))
                        date_tag = pub_item.find("span", class_="date")
                        date = date_tag.get_text(strip=True) if date_tag else "N/A"
                        publications_to_scrape.append({"title": title, "url": pub_url, "date": date})
                        print(f"Found publication: {title} ({date}) - {pub_url}")

            # Find and add the next page link to the queue if it exists            
            next_page_tag = soup.find("a", class_="nextLink")
            print(f"Next page link found: {next_page_tag is not None}")
            if isinstance(next_page_tag, Tag) and "href" in next_page_tag.attrs:
                next_page_url = urljoin(BASE_URL, str(next_page_tag["href"]))
                if next_page_url not in visited_urls:
                    visited_urls.add(next_page_url)
                    queue.append(next_page_url)

            # Polite crawling: # Pause to prevent overloading the server
            time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

        print(f"--- Discovery complete. Found {len(publications_to_scrape)} publications to scrape. ---")

        # PHASE 2: SCRAPE AUTHOR DETAILS AND ABSTRACT FOR EACH PUBLICATION
        # This phase visits each publication's detail page to get more information.
        print("\n--- Phase 2: Scraping author details and abstract for each publication ---")
        final_publications = []
        for pub_data in publications_to_scrape:
            if not rp.can_fetch(USER_AGENT, pub_data['url']):
                print(f"\nSkipping disallowed URL (from robots.txt): {pub_data['url']}")
                continue
            
            success = False
            print(f"\nProcessing publication: {pub_data['url']}")
            for attempt in range(MAX_RETRIES):
                try:
                    driver.get(pub_data["url"])
                    # Wait for a key element (authors' paragraph) to ensure the page has loaded
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "p.relations.persons")))
                    detail_soup = BeautifulSoup(driver.page_source, "html.parser")
                    # Fetch the authors and abstract using the helper functions
                    pub_data["authors"] = fetch_publication_author(detail_soup, BASE_URL)
                    pub_data["abstract"] = fetch_publication_abstract(detail_soup)
                    success = True
                    print(f"Successfully scraped publication: {pub_data['title']}")
                    break
                except (TimeoutException, WebDriverException) as e:
                    # Retry if loading the detail page fails
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for {pub_data['url']}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

            if not success:
                # If all retries fail, save the data as is and move on
                print(f"All retries failed for {pub_data['url']}. Saving without author details and abstract.")
                pub_data["authors"] = []
                pub_data["abstract"] = ""

            final_publications.append(pub_data)
            # Add a small, randomized delay between detail page scrapes
            time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

    finally:
        # Always quit the driver to free up system resources
        print("\nClosing Selenium driver.")
        driver.quit()

    # Save the extracted data to the specified JSON file
    with open(CRAWLED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(final_publications, f, indent=4, ensure_ascii=False)

    print(f"\nCrawling complete. Found {len(final_publications)} publications.")
    print(f"Data saved to {CRAWLED_DATA_FILE}")

if __name__ == '__main__':
    crawl()