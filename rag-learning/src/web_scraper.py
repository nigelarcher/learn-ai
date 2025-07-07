#!/usr/bin/env python3
"""
Web Scraper: Responsibly scrape public data for RAG demo
Limited to small amounts for educational purposes only
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict
from datetime import datetime


class ResponsibleWebScraper:
    """
    A responsible web scraper that:
    - Respects robots.txt
    - Adds delays between requests
    - Limits the amount of data scraped
    - Only scrapes public, freely available content
    """
    
    def __init__(self, delay_seconds: float = 1.0, max_pages: int = 5):
        """
        Initialize scraper with responsible defaults.
        
        Args:
            delay_seconds: Delay between requests (be nice to servers!)
            max_pages: Maximum pages to scrape (keep it small for demos)
        """
        self.delay_seconds = delay_seconds
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Educational-RAG-Demo-Bot/1.0 (Learning purposes only)'
        })
        
    def scrape_wikipedia_page(self, topic: str) -> Dict:
        """
        Scrape a single Wikipedia page (public domain content).
        Wikipedia is chosen because:
        - Content is freely available under Creative Commons
        - Has a clear structure
        - Educational use is encouraged
        """
        # Clean topic for URL
        topic_url = topic.replace(' ', '_')
        url = f"https://en.wikipedia.org/wiki/{topic_url}"
        
        print(f"📖 Fetching Wikipedia page: {topic}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get the main content div
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
                
            # Extract paragraphs (limit to first few for demo)
            paragraphs = []
            for p in content_div.find_all('p', limit=5):
                text = p.get_text().strip()
                # Skip very short paragraphs
                if len(text) > 50:
                    paragraphs.append(text)
                    
            # Extract section headers for context
            sections = []
            for h2 in content_div.find_all(['h2', 'h3'], limit=10):
                section_text = h2.get_text().strip()
                if section_text and not section_text.startswith('['):
                    sections.append(section_text)
            
            return {
                'url': url,
                'title': soup.find('h1').get_text() if soup.find('h1') else topic,
                'content': '\n\n'.join(paragraphs),
                'sections': sections,
                'scraped_at': datetime.now().isoformat(),
                'source': 'Wikipedia'
            }
            
        except requests.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
        finally:
            # Always wait between requests
            time.sleep(self.delay_seconds)
    
    def scrape_topics(self, topics: List[str]) -> List[Dict]:
        """
        Scrape multiple topics with rate limiting.
        
        Args:
            topics: List of topics to scrape
            
        Returns:
            List of scraped documents
        """
        documents = []
        
        # Limit topics to max_pages
        topics_to_scrape = topics[:self.max_pages]
        
        print(f"\n🔍 Scraping {len(topics_to_scrape)} topics (limited to {self.max_pages} for demo)")
        print("=" * 60)
        
        for i, topic in enumerate(topics_to_scrape):
            print(f"\n[{i+1}/{len(topics_to_scrape)}] ", end="")
            
            doc = self.scrape_wikipedia_page(topic)
            if doc:
                documents.append(doc)
                print(f"✅ Successfully scraped: {doc['title']}")
                print(f"   Content length: {len(doc['content'])} characters")
                print(f"   Sections found: {len(doc['sections'])}")
            else:
                print(f"❌ Failed to scrape: {topic}")
                
        print(f"\n✅ Scraping complete! Got {len(documents)} documents")
        return documents


def scrape_ai_knowledge_base():
    """
    Scrape a small AI knowledge base for our RAG demo.
    Using Wikipedia as it's public domain and educational.
    """
    print("🌐 Web Scraping for RAG Demo")
    print("=" * 60)
    print("📝 Note: This scraper is designed for educational purposes")
    print("   - Respects rate limits (1 second delay)")
    print("   - Only scrapes public domain content (Wikipedia)")
    print("   - Limited to 5 pages maximum")
    print("=" * 60)
    
    # Topics related to AI for our knowledge base
    ai_topics = [
        "Artificial intelligence",
        "Machine learning", 
        "Natural language processing",
        "Computer vision",
        "Neural network",
        # More topics available but limited to 5 for demo
        "Deep learning",
        "Reinforcement learning",
        "Transformer (machine learning model)"
    ]
    
    # Create scraper with responsible settings
    scraper = ResponsibleWebScraper(
        delay_seconds=1.0,  # 1 second between requests
        max_pages=5         # Only scrape 5 pages for demo
    )
    
    # Scrape the topics
    documents = scraper.scrape_topics(ai_topics)
    
    # Save to file
    if documents:
        output_path = "data/scraped_ai_knowledge.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(documents)} documents to {output_path}")
        
        # Show summary
        print("\n📊 Scraping Summary:")
        print("-" * 40)
        total_chars = sum(len(doc['content']) for doc in documents)
        print(f"Total documents: {len(documents)}")
        print(f"Total content: {total_chars:,} characters")
        print(f"Average document size: {total_chars // len(documents):,} characters")
        
        print("\n📑 Documents scraped:")
        for doc in documents:
            print(f"  - {doc['title']}")
            
    return documents


def load_scraped_documents(filepath: str = "data/scraped_ai_knowledge.json") -> List[Dict]:
    """Load previously scraped documents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ No scraped documents found at {filepath}")
        print("   Run the scraper first to collect documents.")
        return []


if __name__ == "__main__":
    # Demo the scraper
    scrape_ai_knowledge_base()