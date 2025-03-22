import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Optional


def extract_reviews(page_num: int) -> Optional[List[Dict]]:
    """
    Extract reviews from a given page number.

    Args:
        page_num (int): The page number to scrape.

    Returns:
        Optional[List[Dict]]: A list of reviews if available, otherwise None if no reviews or page is 404.
    """
    url = f'https://www.trustpilot.com/review/www.keytradebank.be?languages=all&page={page_num}'
    response = requests.get(url)

    if response.status_code == 404:
        return None

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})

    if script_tag:
        json_data_str = script_tag.string
        try:
            json_data = json.loads(json_data_str)
            reviews = json_data.get('props', {}).get('pageProps', {}).get('reviews', [])
            if not reviews:
                return None
            return reviews
        except json.JSONDecodeError:
            return []
    return []


def save_reviews_to_json(reviews: List[Dict], filename: str = 'reviews.json') -> None:
    """
    Save the extracted reviews to a JSON file.

    Args:
        reviews (List[Dict]): List of reviews to save.
        filename (str): The filename where reviews are stored.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_reviews = json.load(f)
    else:
        existing_reviews = []

    existing_reviews.extend(reviews)

    with open(filename, 'w') as f:
        json.dump(existing_reviews, f, indent=4)


def scrape_and_save_reviews(page_num: int) -> bool:
    """
    Scrape reviews from a given page number and save them to a JSON file.

    Args:
        page_num (int): The page number to scrape.
    """
    reviews = extract_reviews(page_num)
    if reviews is None:
        print(f"404 encountered at page {page_num}. Stopping scraping.")
        return False
    save_reviews_to_json(reviews)
    return True


def scrape_reviews() -> None:
    """
    Scrape reviews from sequential pages and save them to a JSON file until a 404 page is encountered.
    """
    page_num = 1
    while True:
        print(f"Scraping page {page_num}...")
        if not scrape_and_save_reviews(page_num):
            break
        page_num += 1

    print("Scraping process completed.")


if __name__ == '__main__':
    scrape_reviews()
