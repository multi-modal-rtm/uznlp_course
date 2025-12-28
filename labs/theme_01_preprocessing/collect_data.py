import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

CATEGORIES = {
    "education": "https://kun.uz/news/category/talim",
    "finance": "https://kun.uz/news/category/moliya",
    "auto": "https://kun.uz/news/category/avto"
}
LIMIT_PER_CATEGORY = 20
OUTPUT_PATH = "data/raw/uzbek_news.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

def get_article_links_robust(category_url, limit=20):
    print(f"Scanning category: {category_url}...")
    links = set()
    try:
        response = requests.get(category_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")
        
        all_tags = soup.find_all("a", href=True)
        for tag in all_tags:
            href = tag['href']
            
            if not href.startswith("http"):
                href = "https://kun.uz" + href if href.startswith("/") else href

            if re.search(r'/news/\d{4}/\d{2}/\d{2}/', href):
                links.add(href)
                
            if len(links) >= limit:
                break
    except Exception as e:
        print(f"Error scanning: {e}")
    return list(links)

def parse_article_smart(url, category):
    """
    Smart extraction: If specific classes fail, it finds the block with the most text.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.content, "html.parser")

        title = "No Title"
        title_selectors = ["div.single-header__title", "h1", ".news-title"]
        for selector in title_selectors:
            tag = soup.select_one(selector)
            if tag:
                title = tag.get_text(strip=True)
                break

        body_text = ""

        content_div = soup.select_one("div.single-content")

        if not content_div:
            max_p_count = 0
            candidate_div = None
            for div in soup.find_all("div"):
                p_count = len(div.find_all("p", recursive=False))
                if p_count > max_p_count:
                    max_p_count = p_count
                    candidate_div = div
            
            if candidate_div and max_p_count > 3:
                content_div = candidate_div

        if content_div:
            for junk in content_div(["script", "style", "iframe", "div"]):
                junk.decompose()
            body_text = content_div.get_text(separator=" ", strip=True)
        
        if len(body_text) < 100:
            return None

        return {
            "url": url,
            "category": category,
            "title": title,
            "body": body_text
        }

    except Exception as e:
        return None

def main():
    print("Starting FINAL Data Collection...")
    all_articles = []
    
    for cat_name, cat_url in CATEGORIES.items():
        links = get_article_links_robust(cat_url, limit=LIMIT_PER_CATEGORY)
        print(f"   Found {len(links)} VALID articles for '{cat_name}'. Downloading...")
        
        for i, link in enumerate(links):
            article = parse_article_smart(link, cat_name)
            if article:
                all_articles.append(article)
                print(f"     [{i+1}] Parsed: {article['title'][:40]}...")
            
            time.sleep(random.uniform(0.5, 1.0))
            
    if all_articles:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df = pd.DataFrame(all_articles)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSuccess! Saved {len(df)} articles to {OUTPUT_PATH}")
    else:
        print("\nFailed. The website structure might be completely blocking scripts.")

if __name__ == "__main__":
    main()