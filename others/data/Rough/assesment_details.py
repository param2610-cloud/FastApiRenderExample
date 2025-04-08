import os
import csv
import requests
from bs4 import BeautifulSoup
import time
import random
import re

def fetch_assessment_details(url):
    """Fetch the assessment details page and return the HTML content"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_assessment_details(html_content):
    """Parse the HTML content to extract structured data"""
    if not html_content:
        return {}
    
    data = {}
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract description
    desc_divs = soup.select('div.product-catalogue-training-calendar__row.typ')
    for div in desc_divs:
        if div.find('h4') and div.find('h4').text.strip() == 'Description':
            description = div.find('p').text.strip() if div.find('p') else ""
            data['description'] = description
            break
    
    # Extract job levels
    job_levels_divs = soup.select('div.product-catalogue-training-calendar__row.typ')
    for div in job_levels_divs:
        if div.find('h4') and div.find('h4').text.strip() == 'Job levels':
            job_levels_text = div.find('p').text.strip() if div.find('p') else ""
            # Split by comma and remove trailing comma/spaces
            job_levels = [level.strip() for level in job_levels_text.split(',') if level.strip()]
            data['job_levels'] = job_levels
            break
    
    # Extract languages
    languages_divs = soup.select('div.product-catalogue-training-calendar__row.typ')
    for div in languages_divs:
        if div.find('h4') and div.find('h4').text.strip() == 'Languages':
            languages_text = div.find('p').text.strip() if div.find('p') else ""
            languages = [lang.strip() for lang in languages_text.split(',') if lang.strip()]
            data['languages'] = languages
            break
    
    # Extract assessment length/duration and standardize to minutes
    length_divs = soup.select('div.product-catalogue-training-calendar__row.typ')
    for div in length_divs:
        if div.find('h4') and any(term in div.find('h4').text.strip() for term in ['Assessment length', 'Duration', 'Completion Time']):
            length_text = div.find('p').text.strip() if div.find('p') else ""
            
            # First, try to find minutes
            min_match = re.search(r'(\d+)\s*(?:min|minutes|mins)', length_text, re.IGNORECASE)
            if min_match:
                data['duration'] = int(min_match.group(1))
                continue
            
            # Check for hours and minutes format (e.g., "1 hour 30 min" or "1.5 hours")
            hour_min_match = re.search(r'(\d+)\s*hour(?:s)?\s*(?:and\s*)?(\d+)\s*min(?:ute)?(?:s)?', length_text, re.IGNORECASE)
            if hour_min_match:
                hours = int(hour_min_match.group(1))
                minutes = int(hour_min_match.group(2))
                data['duration'] = hours * 60 + minutes
                continue
            
            # Check for decimal hours (e.g., "1.5 hours")
            decimal_hour_match = re.search(r'(\d+(?:\.\d+)?)\s*hour(?:s)?', length_text, re.IGNORECASE)
            if decimal_hour_match:
                hours = float(decimal_hour_match.group(1))
                data['duration'] = int(hours * 60)
                continue
            
            # If no specific time format is found, but there's a number, assume it's minutes
            generic_number_match = re.search(r'(\d+)', length_text)
            if generic_number_match:
                data['duration'] = int(generic_number_match.group(1))
                continue
                
            # If no duration found, set to None
            data['duration'] = None
            
            break
    
    return data

def load_assessments(csv_path):
    """Load assessment data from CSV file"""
    assessments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assessments = list(reader)
    return assessments

def save_to_csv(data, output_path):
    """Save assessment details to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["name", "url", "description", "job_levels", "languages", "duration", 
                     "test_type", "remote_testing", "adaptive_irt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            # Convert lists to strings for CSV
            item_copy = item.copy()
            for field in ['job_levels', 'languages']:
                if isinstance(item_copy.get(field), list):
                    item_copy[field] = ", ".join(item_copy[field])
            writer.writerow(item_copy)

def main():
    input_csv = 'assessment_data.csv'  # Path to input CSV with basic assessment data
    output_csv = 'assessment_details.csv'  # Path for output CSV with detailed data
    
    # Load assessment data
    assessments = load_assessments(input_csv)
    print(f"Loaded {len(assessments)} assessments from {input_csv}")
    
    # Process each assessment
    detailed_assessments = []
    for i, assessment in enumerate(assessments):
        print(f"Processing {i+1}/{len(assessments)}: {assessment['name']}")
        
        # Construct full URL
        url = f"https://www.shl.com{assessment['url']}"
        
        # Fetch HTML content
        html_content = fetch_assessment_details(url)
        
        # Parse details
        details = parse_assessment_details(html_content)
        
        # Create result
        result = {
            "name": assessment["name"],
            "url": url,
            "description": details.get("description", ""),
            "job_levels": details.get("job_levels", []),
            "languages": details.get("languages", []),
            "duration": details.get("duration"),
            "test_type": assessment["test_type"],
            "remote_testing": assessment["remote_testing"] == "Yes",
            "adaptive_irt": assessment["adaptive_irt"] == "Yes"
        }
        
        detailed_assessments.append(result)
        
        # Add delay to be respectful to the server
        time.sleep(random.uniform(1, 2))
    
    # Save to CSV
    save_to_csv(detailed_assessments, output_csv)
    print(f"Saved details for {len(detailed_assessments)} assessments to {output_csv}")

if __name__ == "__main__":
    main()