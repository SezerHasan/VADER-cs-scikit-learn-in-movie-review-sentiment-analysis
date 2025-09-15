#!/usr/bin/env python3
"""
Data Collection Script for TOEFL/IELTS AI Teaching Platform
Collects training data from various free sources
"""

import os
import requests
import json
import csv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def create_directories(self):
        """Create necessary directories for data storage"""
        dirs = ['grammar', 'reading', 'writing', 'speaking']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)
        logger.info("Created data directories")
    
    def collect_grammar_data(self):
        """Collect grammar exercises and corrections"""
        logger.info("Starting grammar data collection...")
        
        # Sample grammar data structure
        grammar_data = [
            {
                "id": "gram_001",
                "type": "error_correction",
                "difficulty": "beginner",
                "sentence": "I am go to school yesterday.",
                "corrected": "I went to school yesterday.",
                "explanation": "Past tense should be used for actions completed in the past.",
                "grammar_point": "past_tense"
            },
            {
                "id": "gram_002", 
                "type": "multiple_choice",
                "difficulty": "intermediate",
                "question": "Choose the correct form:",
                "options": ["She have been", "She has been", "She is been", "She was been"],
                "correct": 1,
                "explanation": "Present perfect tense uses 'has' with third person singular.",
                "grammar_point": "present_perfect"
            }
        ]
        
        # Save grammar data
        with open(os.path.join(self.base_dir, 'grammar', 'grammar_exercises.json'), 'w') as f:
            json.dump(grammar_data, f, indent=2)
        
        logger.info(f"Collected {len(grammar_data)} grammar exercises")
    
    def collect_reading_data(self):
        """Collect reading comprehension passages and questions"""
        logger.info("Starting reading data collection...")
        
        reading_data = [
            {
                "id": "read_001",
                "title": "Climate Change and Its Effects",
                "passage": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities have been the main driver of climate change since the 1800s, primarily due to burning fossil fuels...",
                "questions": [
                    {
                        "question": "What is the main cause of climate change since the 1800s?",
                        "options": ["Natural variations", "Human activities", "Solar radiation", "Ocean currents"],
                        "correct": 1,
                        "explanation": "The passage states that human activities have been the main driver of climate change since the 1800s."
                    }
                ],
                "difficulty": "intermediate",
                "word_count": 250
            }
        ]
        
        with open(os.path.join(self.base_dir, 'reading', 'reading_passages.json'), 'w') as f:
            json.dump(reading_data, f, indent=2)
        
        logger.info(f"Collected {len(reading_data)} reading passages")
    
    def collect_writing_data(self):
        """Collect writing prompts and sample essays"""
        logger.info("Starting writing data collection...")
        
        writing_data = [
            {
                "id": "write_001",
                "type": "essay",
                "topic": "The advantages and disadvantages of social media",
                "prompt": "Some people believe that social media has a positive impact on society, while others argue that it has negative effects. Discuss both views and give your opinion.",
                "sample_essay": "Social media has become an integral part of modern life, connecting people across the globe...",
                "band_score": 7.0,
                "criteria": {
                    "task_achievement": 7,
                    "coherence_cohesion": 7,
                    "lexical_resource": 7,
                    "grammatical_range_accuracy": 7
                },
                "difficulty": "intermediate"
            }
        ]
        
        with open(os.path.join(self.base_dir, 'writing', 'writing_prompts.json'), 'w') as f:
            json.dump(writing_data, f, indent=2)
        
        logger.info(f"Collected {len(writing_data)} writing prompts")
    
    def collect_speaking_data(self):
        """Collect speaking practice materials and sample responses"""
        logger.info("Starting speaking data collection...")
        
        speaking_data = [
            {
                "id": "speak_001",
                "part": 1,
                "topic": "Hometown",
                "questions": [
                    "Where are you from?",
                    "What do you like about your hometown?",
                    "Has your hometown changed much over the years?"
                ],
                "sample_responses": [
                    "I'm from a small city in the north of the country...",
                    "What I like most about my hometown is the sense of community...",
                    "Yes, my hometown has changed quite significantly..."
                ],
                "difficulty": "beginner"
            }
        ]
        
        with open(os.path.join(self.base_dir, 'speaking', 'speaking_topics.json'), 'w') as f:
            json.dump(speaking_data, f, indent=2)
        
        logger.info(f"Collected {len(speaking_data)} speaking topics")
    
    def scrape_ielts_data(self):
        """Scrape data from IELTS preparation websites"""
        logger.info("Starting web scraping for IELTS data...")
        
        # URLs to scrape (add more as needed)
        urls = [
            "https://ieltsdata.org/",
            "https://www.ielts.org/about-the-test/sample-test-questions"
        ]
        
        scraped_data = []
        
        for url in urls:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract relevant content (customize based on website structure)
                # This is a basic example - you'll need to adapt for each site
                content = {
                    "url": url,
                    "title": soup.find('title').text if soup.find('title') else "",
                    "text": soup.get_text()[:1000]  # First 1000 characters
                }
                
                scraped_data.append(content)
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        # Save scraped data
        with open(os.path.join(self.base_dir, 'scraped_data.json'), 'w') as f:
            json.dump(scraped_data, f, indent=2)
        
        logger.info(f"Scraped data from {len(scraped_data)} websites")
    
    def run_collection(self):
        """Run the complete data collection process"""
        logger.info("Starting data collection process...")
        
        # Create directories
        self.create_directories()
        
        # Collect data for each skill
        self.collect_grammar_data()
        self.collect_reading_data()
        self.collect_writing_data()
        self.collect_speaking_data()
        
        # Scrape additional data
        self.scrape_ielts_data()
        
        logger.info("Data collection completed successfully!")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_collection()