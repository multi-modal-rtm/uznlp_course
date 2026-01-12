import sys
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# --- SOZLAMALAR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
data_path = os.path.join(project_root, 'data/raw/uzbek_news.csv')

SOURCES = [
    {
        "domain": "kun.uz",
        "categories": {
            "education": "https://kun.uz/news/category/talim",
            "finance": "https://kun.uz/news/category/moliya",
            "auto": "https://kun.uz/news/category/avto",
            "sport": "https://kun.uz/news/category/sport",
            "technology": "https://kun.uz/news/category/texnologiya"
        }
    },
    {
        "domain": "daryo.uz",
        "categories": {
            "education": "https://daryo.uz/category/social",
            "finance": "https://daryo.uz/category/business",
            "auto": "https://daryo.uz/category/avto",
            "sport": "https://daryo.uz/category/sport",
            "technology": "https://daryo.uz/category/texnologiya"
        }
    },
    {
        "domain": "xabar.uz",
        "categories": {
            "education": "https://xabar.uz/uz/ta-lim",
            "finance": "https://xabar.uz/uz/iqtisodiyot",
            "auto": "https://xabar.uz/uz/avto",
            "sport": "https://xabar.uz/uz/sport",
            "technology": "https://xabar.uz/uz/texnologiya"
        }
    }
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def get_links(url, domain):
    links = set()
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup.find_all("a", href=True):
            href = tag['href']
            if not href.startswith("http"):
                base = f"https://{domain}"
                href = base + href if href.startswith("/") else href
            
            # Har bir sayt uchun o'ziga xos URL pattern
            is_valid = False
            if domain == "kun.uz" and re.search(r'/news/\d{4}/\d{2}/\d{2}/', href): is_valid = True
            elif domain == "daryo.uz" and re.search(r'/\d{4}/\d{2}/\d{2}/', href): is_valid = True
            elif domain == "xabar.uz" and "/uz/" in href and href.count('/') > 4: is_valid = True

            if is_valid: links.add(href)
    except Exception as e:
        print(f"   ⚠️ Link olishda xato ({domain}): {e}")
        
    return list(links)

def find_best_content_div(soup):
    """
    Sahifadagi eng ko'p <p> tegi bor divni topadi.
    Bu funksiya har qanday sayt uchun "Universal kalit" hisoblanadi.
    """
    max_p = 0
    best_div = None
    
    # Barcha divlarni tekshiramiz
    for div in soup.find_all('div'):
        # Footer, header, sidebarlarni o'tkazib yuboramiz
        classes = str(div.get('class', [])).lower()
        if any(x in classes for x in ['footer', 'header', 'menu', 'sidebar', 'widget', 'related', 'banner', 'comment']):
            continue
            
        # Paragraflarni sanaymiz (Recursive=True -> ichma-ich hamma p larni sanaydi)
        p_count = len(div.find_all('p'))
        
        # Eng kami 3 ta paragraf bo'lishi kerak (Popup va reklamalarni elash uchun)
        if p_count > max_p and p_count >= 3:
            max_p = p_count
            best_div = div
            
    return best_div

def parse_article(url, category, domain):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.content, 'html.parser')

        # TITLE TOPISH
        title = ""
        h1 = soup.find('h1')
        if h1: title = h1.get_text(strip=True)
        
        # BODY TOPISH (AQLLI USUL)
        body_div = find_best_content_div(soup)
        
        if not body_div: return None

        # Tozalash
        for junk in body_div(["script", "style", "iframe", "aside", "a", "button", "figure", "div.related-news"]):
            junk.decompose()
            
        body = body_div.get_text(separator=' ', strip=True)

        # FINAL FILTRLAR
        if len(body) < 100: return None # Juda qisqa
        if "imloviy xato" in body.lower() and len(body) < 300: return None # Popup
        
        return {
            'url': url,
            'category': category,
            'source': domain,
            'title': title,
            'body': body
        }

    except Exception:
        return None

def main():
    all_data = []
    print("🚀 3-MANBALI FINAL SCRAPER ISHGA TUSHDI...")

    for source in SOURCES:
        domain = source['domain']
        print(f"\n🌍 MANBA: {domain.upper()}")
        
        for cat_name, cat_url in source['categories'].items():
            print(f"   📂 {cat_name.upper()}...", end="")
            collected_links = set()
            
            # 5 ta sahifa
            for page in range(1, 6):
                # Xabar.uz sahifalash tizimi boshqacha (/page/2)
                if domain == "xabar.uz":
                    page_url = f"{cat_url}/page/{page}" if page > 1 else cat_url
                else:
                    connector = "&" if "?" in cat_url else "?"
                    page_url = f"{cat_url}{connector}page={page}" if page > 1 else cat_url
                
                # Linklarni yig'ish
                new_links = get_links(page_url, domain)
                collected_links.update(new_links)
                time.sleep(0.3) # Serverni og'irlashtirmaslik uchun

            print(f" Linklar: {len(collected_links)} ta. Yuklanmoqda...", end="")
            
            # Maqolalarni yuklash
            count = 0
            for link in collected_links:
                article = parse_article(link, cat_name, domain)
                if article:
                    all_data.append(article)
                    count += 1
                time.sleep(0.1)
            print(f" ✅ {count} ta saqlandi")

    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # Faylni yangilash (Append emas, Write)
        df.to_csv(data_path, index=False)
        print(f"\n🎉 JAMI NATIJA: {len(df)} ta maqola!")
        print(df['category'].value_counts())
    else:
        print("❌ Hech qanday ma'lumot yig'ilmadi. Internetni tekshiring.")

if __name__ == "__main__":
    main()
















# import sys
# import os
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# import re

# # --- SOZLAMALAR ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# data_path = os.path.join(project_root, 'data/raw/uzbek_news.csv')

# # 3 TA MANBA
# SOURCES = [
#     {
#         "domain": "kun.uz",
#         "categories": {
#             "education": "https://kun.uz/news/category/talim",
#             "finance": "https://kun.uz/news/category/moliya",
#             "auto": "https://kun.uz/news/category/avto",
#             "sport": "https://kun.uz/news/category/sport",
#             "technology": "https://kun.uz/news/category/texnologiya"
#         }
#     },
#     {
#         "domain": "daryo.uz",
#         "categories": {
#             "education": "https://daryo.uz/category/social",
#             "finance": "https://daryo.uz/category/business",
#             "auto": "https://daryo.uz/category/avto",
#             "sport": "https://daryo.uz/category/sport",
#             "technology": "https://daryo.uz/category/texnologiya"
#         }
#     },
#     {
#         "domain": "xabar.uz",
#         "categories": {
#             "education": "https://xabar.uz/uz/ta-lim",
#             "finance": "https://xabar.uz/uz/iqtisodiyot",
#             "auto": "https://xabar.uz/uz/avto",
#             "sport": "https://xabar.uz/uz/sport",
#             "technology": "https://xabar.uz/uz/texnologiya"
#         }
#     }
# ]

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# }

# def get_links(url, domain):
#     links = set()
#     try:
#         response = requests.get(url, headers=HEADERS, timeout=10)
#         if response.status_code != 200: return []
        
#         soup = BeautifulSoup(response.content, 'html.parser')
#         all_tags = soup.find_all("a", href=True)
        
#         for tag in all_tags:
#             href = tag['href']
            
#             # Domen nomini to'g'irlash
#             if not href.startswith("http"):
#                 base = f"https://{domain}"
#                 href = base + href if href.startswith("/") else href
            
#             # Har bir sayt uchun o'ziga xos URL pattern
#             is_valid = False
#             if domain == "kun.uz" and re.search(r'/news/\d{4}/\d{2}/\d{2}/', href):
#                 is_valid = True
#             elif domain == "daryo.uz" and re.search(r'/\d{4}/\d{2}/\d{2}/', href):
#                 is_valid = True
#             elif domain == "xabar.uz" and ("/uz/" in href) and (href.count('/') > 4):
#                 is_valid = True

#             if is_valid:
#                 links.add(href)
                
#     except Exception:
#         pass
#     return list(links)

# def parse_article(url, category, domain):
#     try:
#         response = requests.get(url, headers=HEADERS, timeout=10)
#         if response.status_code != 200: return None
        
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # --- 1. TITLE TOPISH ---
#         title = ""
#         h1 = soup.find('h1')
#         if h1:
#             title = h1.get_text(strip=True)
#         else:
#             # Fallback
#             t_div = soup.find('div', class_=re.compile(r'(title|header)'))
#             if t_div: title = t_div.get_text(strip=True)

#         # --- 2. BODY TOPISH (GIBRID USUL) ---
#         body = ""
#         content_div = None

#         # A) KUN.UZ va XABAR.UZ uchun "Eng ko'p <p>" usuli yaxshi ishlaydi
#         if domain in ["kun.uz", "xabar.uz"]:
#             max_p_count = 0
#             for div in soup.find_all('div'):
#                 classes = str(div.get('class', [])).lower()
#                 if 'footer' in classes or 'header' in classes or 'related' in classes: continue
                
#                 # Recursive=False muhim, aks holda butun saytni olib ketadi
#                 p_tags = div.find_all('p', recursive=False)
#                 if len(p_tags) > max_p_count:
#                     max_p_count = len(p_tags)
#                     content_div = div

#         # B) DARYO.UZ uchun maxsus klass qidiramiz
#         elif domain == "daryo.uz":
#             content_div = soup.find('div', class_='article-content') or \
#                           soup.find('div', class_='main-content') or \
#                           soup.find('div', class_='post-content')

#         # Agar topilmasa, Daryo uchun ham "Universal" usulni qo'llab ko'ramiz
#         if not content_div:
#              max_p_count = 0
#              for div in soup.find_all('div'):
#                 p_tags = div.find_all('p') # Recursive True
#                 if len(p_tags) > max_p_count:
#                     max_p_count = len(p_tags)
#                     content_div = div

#         if content_div:
#             # Tozalash
#             for s in content_div(["script", "style", "button", "a", "iframe"]): 
#                 s.decompose()
#             body = content_div.get_text(separator=' ', strip=True)

#         if len(body) < 100 or len(title) < 5:
#             return None

#         return {
#             'url': url,
#             'category': category,
#             'source': domain,
#             'title': title,
#             'body': body
#         }

#     except Exception:
#         return None

# def main():
#     all_data = []
#     print("🚀 3 TA MANBA: KUN.UZ + DARYO.UZ + XABAR.UZ ...")

#     for source in SOURCES:
#         domain = source['domain']
#         print(f"\n🌍 MANBA: {domain.upper()}")
        
#         for cat_name, cat_url in source['categories'].items():
#             print(f"   📂 Kategoriya: {cat_name.upper()}")
            
#             collected_links = set()
#             # 5 ta sahifa
#             for page in range(1, 6):
#                 if domain == "kun.uz":
#                     connector = "&" if "?" in cat_url else "?"
#                     page_url = f"{cat_url}{connector}page={page}" if page > 1 else cat_url
#                 elif domain == "daryo.uz":
#                     connector = "&" if "?" in cat_url else "?"
#                     page_url = f"{cat_url}{connector}page={page}" if page > 1 else cat_url
#                 elif domain == "xabar.uz":
#                     # Xabar.uz sahifalash: /page/2
#                     page_url = f"{cat_url}/page/{page}" if page > 1 else cat_url

#                 new_links = get_links(page_url, domain)
#                 collected_links.update(new_links)
#                 print(f"      Sahifa {page}: {len(new_links)} link.")
#                 time.sleep(0.5)

#             print(f"      📥 {len(collected_links)} ta maqola yuklanmoqda...")
            
#             saved_count = 0
#             for link in collected_links:
#                 article = parse_article(link, cat_name, domain)
#                 if article:
#                     all_data.append(article)
#                     saved_count += 1
#                 time.sleep(0.2)
            
#             print(f"      ✅ {saved_count} ta saqlandi.")

#     # CSV SAQLASH
#     if all_data:
#         df = pd.DataFrame(all_data)
#         os.makedirs(os.path.dirname(data_path), exist_ok=True)
#         df.to_csv(data_path, index=False)
#         print(f"\n🎉 JAMI: {len(df)} ta maqola!")
#         print(f"Manzil: {data_path}")
#         print(df['category'].value_counts())
#     else:
#         print("❌ Xatolik: Ma'lumot yig'ilmadi.")

# if __name__ == "__main__":
#     main()













# # import sys
# # import os
# # import requests
# # from bs4 import BeautifulSoup
# # import pandas as pd
# # import time
# # import re

# # # --- SETUP ---
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# # data_path = os.path.join(project_root, 'data/raw/uzbek_news.csv')

# # # Categories
# # CATEGORIES = {
# #     "education": "https://kun.uz/news/category/talim",
# #     "finance": "https://kun.uz/news/category/moliya",
# #     "auto": "https://kun.uz/news/category/avto",
# #     "sport": "https://kun.uz/news/category/sport",
# #     "technology": "https://kun.uz/news/category/texnologiya"
# # }

# # HEADERS = {
# #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# # }

# # def get_links_from_page(url):
# #     """
# #     Extracts article URLs using Regex (looking for /news/yyyy/mm/dd/ pattern).
# #     This is much more robust than looking for 'class' names.
# #     """
# #     links = set()
# #     try:
# #         response = requests.get(url, headers=HEADERS, timeout=10)
# #         if response.status_code != 200:
# #             return []
            
# #         soup = BeautifulSoup(response.content, 'html.parser')
        
# #         # Find ALL links on the page
# #         all_tags = soup.find_all("a", href=True)
        
# #         for tag in all_tags:
# #             href = tag['href']
            
# #             # Fix relative URLs
# #             if not href.startswith("http"):
# #                 href = "https://kun.uz" + href if href.startswith("/") else href
            
# #             # THE MAGIC FIX: Look for the /news/DATE pattern
# #             # Example: https://kun.uz/news/2024/01/01/article-slug
# #             if re.search(r'/news/\d{4}/\d{2}/\d{2}/', href):
# #                 links.add(href)
                
# #     except Exception as e:
# #         print(f"Error fetching links: {e}")
        
# #     return list(links)

# # def parse_article(url, category):
# #     """
# #     Visits the article URL and extracts Title and Body.
# #     """
# #     try:
# #         response = requests.get(url, headers=HEADERS, timeout=10)
# #         soup = BeautifulSoup(response.content, 'html.parser')

# #         # 1. Extract Title (Try multiple common tags)
# #         title = ""
# #         title_tag = soup.find('h1') # Standard H1
# #         if not title_tag:
# #             title_tag = soup.find('div', class_='single-header__title')
        
# #         if title_tag:
# #             title = title_tag.get_text(strip=True)
# #         else:
# #             return None # Skip if no title found

# #         # 2. Extract Body (Try finding the content div)
# #         body = ""
# #         content_div = soup.find('div', class_='single-content')
        
# #         if not content_div:
# #             # Fallback: Find the div with the most <p> tags
# #             max_p = 0
# #             for div in soup.find_all('div'):
# #                 p_count = len(div.find_all('p', recursive=False))
# #                 if p_count > max_p:
# #                     max_p = p_count
# #                     content_div = div

# #         if content_div:
# #             # Clean up scripts and styles inside the text
# #             for script in content_div(["script", "style"]):
# #                 script.decompose()
# #             body = content_div.get_text(separator=' ', strip=True)

# #         if len(body) < 50: # Skip very short/empty articles
# #             return None

# #         return {
# #             'url': url,
# #             'category': category,
# #             'title': title,
# #             'body': body
# #         }

# #     except Exception as e:
# #         return None

# # def main():
# #     all_data = []
    
# #     print("🚀 Starting Data Collection (Robust Mode)...")

# #     for cat_name, cat_url in CATEGORIES.items():
# #         print(f"\n🔍 Processing Category: {cat_name.upper()}")
        
# #         category_links = set()
        
# #         # Pagination Loop (Page 1 to 5)
# #         for page in range(1, 6):
# #             if page == 1:
# #                 page_url = cat_url
# #             else:
# #                 # Kun.uz uses ?q=...&page=X structure usually
# #                 page_url = f"{cat_url}?page={page}"
                
# #             print(f"   Scanning Page {page}...")
# #             new_links = get_links_from_page(page_url)
            
# #             # Add new links to our set
# #             before_count = len(category_links)
# #             category_links.update(new_links)
# #             added_count = len(category_links) - before_count
            
# #             print(f"     Found {len(new_links)} links. (New unique: {added_count})")
            
# #             time.sleep(1) # Be polite to the server

# #         print(f"   Downloading {len(category_links)} articles for {cat_name}...")
        
# #         # Download content for each link
# #         for i, link in enumerate(category_links):
# #             article = parse_article(link, cat_name)
# #             if article:
# #                 all_data.append(article)
# #                 # Print progress every 10 articles
# #                 if (i+1) % 10 == 0:
# #                     print(f"     Parsed {i+1}/{len(category_links)}: {article['title'][:30]}...")
            
# #             time.sleep(0.5)

# #     # Save Data
# #     if all_data:
# #         df = pd.DataFrame(all_data)
# #         os.makedirs(os.path.dirname(data_path), exist_ok=True)
# #         df.to_csv(data_path, index=False)
# #         print(f"\n✅ SUCCESS! Collected {len(df)} articles.")
# #         print(f"💾 Saved to: {data_path}")
# #     else:
# #         print("\n❌ Failed to collect any data.")

# # if __name__ == "__main__":
# #     main()













# # # import requests
# # # from bs4 import BeautifulSoup
# # # import pandas as pd
# # # import time
# # # import random
# # # import os
# # # import sys
# # # import re

# # # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# # # CATEGORIES = {
# # #     "education": "https://kun.uz/news/category/talim",
# # #     "finance": "https://kun.uz/news/category/moliya",
# # #     "auto": "https://kun.uz/news/category/avto",
# # #     "sport": "https://kun.uz/news/category/sport",
# # #     "technology": "https://kun.uz/news/category/texnologiya"
# # # }
# # # LIMIT_PER_CATEGORY = 20
# # # OUTPUT_PATH = "data/raw/uzbek_news.csv"

# # # HEADERS = {
# # #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
# # #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
# # # }

# # # def get_article_links_robust(category_url, limit=20):
# # #     print(f"Scanning category: {category_url}...")
# # #     links = set()
# # #     try:
# # #         response = requests.get(category_url, headers=HEADERS, timeout=15)
# # #         soup = BeautifulSoup(response.content, "html.parser")
        
# # #         all_tags = soup.find_all("a", href=True)
# # #         for tag in all_tags:
# # #             href = tag['href']
            
# # #             if not href.startswith("http"):
# # #                 href = "https://kun.uz" + href if href.startswith("/") else href

# # #             if re.search(r'/news/\d{4}/\d{2}/\d{2}/', href):
# # #                 links.add(href)
                
# # #             if len(links) >= limit:
# # #                 break
# # #     except Exception as e:
# # #         print(f"Error scanning: {e}")
# # #     return list(links)

# # # def parse_article_smart(url, category):
# # #     """
# # #     Smart extraction: If specific classes fail, it finds the block with the most text.
# # #     """
# # #     try:
# # #         response = requests.get(url, headers=HEADERS, timeout=10)
# # #         if response.status_code != 200: return None
# # #         soup = BeautifulSoup(response.content, "html.parser")

# # #         title = "No Title"
# # #         title_selectors = ["div.single-header__title", "h1", ".news-title"]
# # #         for selector in title_selectors:
# # #             tag = soup.select_one(selector)
# # #             if tag:
# # #                 title = tag.get_text(strip=True)
# # #                 break

# # #         body_text = ""

# # #         content_div = soup.select_one("div.single-content")

# # #         if not content_div:
# # #             max_p_count = 0
# # #             candidate_div = None
# # #             for div in soup.find_all("div"):
# # #                 p_count = len(div.find_all("p", recursive=False))
# # #                 if p_count > max_p_count:
# # #                     max_p_count = p_count
# # #                     candidate_div = div
            
# # #             if candidate_div and max_p_count > 3:
# # #                 content_div = candidate_div

# # #         if content_div:
# # #             for junk in content_div(["script", "style", "iframe", "div"]):
# # #                 junk.decompose()
# # #             body_text = content_div.get_text(separator=" ", strip=True)
        
# # #         if len(body_text) < 100:
# # #             return None

# # #         return {
# # #             "url": url,
# # #             "category": category,
# # #             "title": title,
# # #             "body": body_text
# # #         }

# # #     except Exception as e:
# # #         return None

# # # def main():
# # #     print("Starting FINAL Data Collection...")
# # #     all_articles = []
    
# # #     for cat_name, cat_url in CATEGORIES.items():
# # #         links = get_article_links_robust(cat_url, limit=LIMIT_PER_CATEGORY)
# # #         print(f"   Found {len(links)} VALID articles for '{cat_name}'. Downloading...")
        
# # #         for i, link in enumerate(links):
# # #             article = parse_article_smart(link, cat_name)
# # #             if article:
# # #                 all_articles.append(article)
# # #                 print(f"     [{i+1}] Parsed: {article['title'][:40]}...")
            
# # #             time.sleep(random.uniform(0.5, 1.0))
            
# # #     if all_articles:
# # #         os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
# # #         df = pd.DataFrame(all_articles)
# # #         df.to_csv(OUTPUT_PATH, index=False)
# # #         print(f"\nSuccess! Saved {len(df)} articles to {OUTPUT_PATH}")
# # #     else:
# # #         print("\nFailed. The website structure might be completely blocking scripts.")

# # # if __name__ == "__main__":
# # #     main()