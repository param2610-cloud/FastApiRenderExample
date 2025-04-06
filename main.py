# app.py - Main Application File
import os
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Union
# import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# Update these imports to use the new community packages
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Add these imports for Google Gemini
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # Remove the hard-coded key

# Configure Google Gemini if API key is available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
DATA_DIR = "data"
VECTORDB_DIR = "vectordb"
"""Create a FastAPI application."""
from fastapi import FastAPI, Query
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="SHL Assessment Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryItem(BaseModel):
    text: str
    is_url: bool = False
    max_results: int = 10
# Sample assessment structure (for demo purposes)
SAMPLE_ASSESSMENTS = [
    {
        "name": "Account Manager Solution",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/account-manager-solution/",
        "description": "The Account Manager solution is an assessment used for job candidates applying to mid-level leadership positions that tend to manage the day-to-day operations and activities of client accounts. Sample tasks for these jobs include, but are not limited to: communicating with clients about project status, developing and maintaining project plans, coordinating internally with appropriate project personnel, and ensuring client expectations are being met.",
        "job_levels": ["Mid-Professional"],
        "languages": ["English (USA)"],
        "duration": 49,
        "test_type": "CPAB",  # Cognitive, Personality, Aptitude, Behavioral
        "remote_testing": True,
        "adaptive_irt": False  # Assuming this based on typical patterns
    },
    {
        "name": "Python Developer Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/python-developer/",
        "description": "Comprehensive assessment for evaluating Python programming skills, problem-solving abilities, and coding practices. Tests candidates on core Python concepts, data structures, algorithms, and practical application development.",
        "job_levels": ["Entry-Level", "Mid-Professional"],
        "languages": ["English (USA)", "English (UK)"],
        "duration": 60,
        "test_type": "CB",  # Cognitive, Behavioral
        "remote_testing": True,
        "adaptive_irt": True
    },
    {
        "name": "Business Analyst Skills Test",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/business-analyst/",
        "description": "Evaluates key business analysis competencies including requirements gathering, stakeholder management, process modeling, and data analysis skills. Suitable for candidates who will work at the intersection of business and technology.",
        "job_levels": ["Mid-Professional", "Senior"],
        "languages": ["English (USA)", "Spanish"],
        "duration": 45,
        "test_type": "CPB",  # Cognitive, Personality, Behavioral
        "remote_testing": True,
        "adaptive_irt": False
    },
    {
        "name": "Java Developer Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/java-developer/",
        "description": "Comprehensive Java programming assessment testing core Java concepts, object-oriented principles, troubleshooting abilities, and software development practices. Includes practical coding exercises and problem-solving tasks.",
        "job_levels": ["Entry-Level", "Mid-Professional"],
        "languages": ["English (USA)", "English (UK)"],
        "duration": 55,
        "test_type": "CA",  # Cognitive, Aptitude
        "remote_testing": True,
        "adaptive_irt": True
    },
    {
        "name": "Leadership Potential Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/leadership-potential/",
        "description": "Measures key leadership competencies including strategic thinking, team management, decision making, and emotional intelligence. Designed to identify candidates with high potential for leadership roles.",
        "job_levels": ["Mid-Professional", "Senior"],
        "languages": ["English (USA)", "French", "German"],
        "duration": 35,
        "test_type": "PA",  # Personality, Aptitude
        "remote_testing": True,
        "adaptive_irt": True
    },
    {
        "name": "Data Analyst Skills Test",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/data-analyst/",
        "description": "Comprehensive assessment of data analysis capabilities including SQL querying, data manipulation, statistical analysis, and data visualization skills. Tests both technical knowledge and analytical thinking.",
        "job_levels": ["Entry-Level", "Mid-Professional"],
        "languages": ["English (USA)"],
        "duration": 40,
        "test_type": "CA",  # Cognitive, Aptitude
        "remote_testing": True,
        "adaptive_irt": False
    },
    {
        "name": "Customer Service Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-service/",
        "description": "Evaluates key customer service competencies including communication skills, problem-solving abilities, empathy, and handling difficult situations. Designed for roles with direct customer interaction.",
        "job_levels": ["Entry-Level"],
        "languages": ["English (USA)", "Spanish", "French"],
        "duration": 30,
        "test_type": "PB",  # Personality, Behavioral
        "remote_testing": True,
        "adaptive_irt": False
    }
]

class SHLScraper:
    """Scraper for SHL assessment catalog and detail pages."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.catalog_urls = []
        
    def scrape_catalog(self) -> List[str]:
        """Scrape the main catalog page to get all assessment URLs."""
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract assessment links - this selector would need to be updated based on actual page structure
                links = soup.select('a.assessment-link')  # Update with actual CSS selector
                self.catalog_urls = [link['href'] for link in links]
                return self.catalog_urls
            else:
                print(f"Failed to fetch catalog. Status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error scraping catalog: {e}")
            return []
    
    def scrape_assessment_page(self, url: str) -> Dict[str, Any]:
        """Scrape individual assessment detail page."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data - selectors would need to be updated based on actual page structure
                name = soup.select_one('h1').text.strip() if soup.select_one('h1') else ""
                description = soup.select_one('div.product-catalogue-training-calendar__row p').text.strip() if soup.select_one('div.product-catalogue-training-calendar__row p') else ""
                
                # Job levels
                job_levels_el = soup.select_one('h4:contains("Job levels") + p')
                job_levels = [level.strip() for level in job_levels_el.text.split(',')] if job_levels_el else []
                
                # Languages
                languages_el = soup.select_one('h4:contains("Languages") + p')
                languages = [lang.strip() for lang in languages_el.text.split(',')] if languages_el else []
                
                # Duration
                duration_text = soup.select_one('h4:contains("Assessment length") + p').text if soup.select_one('h4:contains("Assessment length") + p') else ""
                duration = int(''.join(filter(str.isdigit, duration_text))) if duration_text else 0
                
                # Test type
                test_type_el = soup.select_one('span.d-flex ms-2')
                test_type = ''.join([span.text.strip() for span in test_type_el.select('span.product-catalogue__key')]) if test_type_el else ""
                
                # Remote testing
                remote_testing = bool(soup.select_one('span.catalogue__circle.-yes')) if soup.select_one('span.catalogue__circle.-yes') else False
                
                # We don't have explicit info for adaptive_irt in the HTML, so we'll assume False as default
                adaptive_irt = False
                
                return {
                    "name": name,
                    "url": url,
                    "description": description,
                    "job_levels": job_levels,
                    "languages": languages,
                    "duration": duration,
                    "test_type": test_type,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt
                }
            else:
                print(f"Failed to fetch assessment page. Status code: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error scraping assessment page: {e}")
            return {}
    
    def scrape_all_assessments(self) -> List[Dict[str, Any]]:
        """Scrape all assessments from the catalog."""
        if not self.catalog_urls:
            self.scrape_catalog()
        
        assessments = []
        for url in self.catalog_urls:
            assessment = self.scrape_assessment_page(url)
            if assessment:
                assessments.append(assessment)
        
        return assessments
    
    def save_assessments(self, output_file: str = os.path.join(DATA_DIR, "assessments.json")):
        """Save scraped assessments to a JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        assessments = self.scrape_all_assessments()
        with open(output_file, 'w') as f:
            json.dump(assessments, f, indent=2)
        return assessments

class SHLRecommender:
    """Recommender system for SHL assessments."""
    
    def __init__(self, use_sample_data: bool = True, use_gemini: bool = True):
        """Initialize the recommender system."""
        self.assessments = []
        self.vectorstore = None
        self.embeddings = None
        
        # Initialize embeddings based on available API keys and preference
        if use_gemini and GOOGLE_API_KEY:
            try:
                print("Using Google Gemini embeddings")
                # Use the correct model name and add task_type for document embeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",  # Correct model name
                    task_type="retrieval_document",  # Optimized for document retrieval
                    google_api_key=GOOGLE_API_KEY
                )
                # Test the embeddings with a simple query to verify it works
                test_result = self.embeddings.embed_query("Test embedding")
                print(f"Embedding test successful: vector length = {len(test_result)}")
            except Exception as e:
                print(f"Error initializing Google Gemini embeddings: {e}")
                self.embeddings = None
        # Fall back to OpenAI if Gemini is not available or not preferred
        elif OPENAI_API_KEY:
            try:
                print("Using OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings()
            except Exception as e:
                print(f"Error initializing OpenAI embeddings: {e}")
                self.embeddings = None
        
        if use_sample_data:
            self.assessments = SAMPLE_ASSESSMENTS
            self._create_vector_store()
    
    def _create_documents(self) -> List[Document]:
        """Create document objects for the vector store."""
        documents = []
        for assessment in self.assessments:
            # Combine relevant fields for the primary content
            content = f"{assessment['name']}\n{assessment['description']}"
            
            # Create metadata for filtering - convert lists to strings to make it compatible with Chroma
            metadata = {
                "name": assessment["name"],
                "url": assessment["url"],
                "job_levels": ", ".join(assessment["job_levels"]),
                "languages": ", ".join(assessment["languages"]),
                "duration": assessment["duration"],
                "test_type": assessment["test_type"],
                "remote_testing": assessment["remote_testing"],
                "adaptive_irt": assessment["adaptive_irt"]
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _create_vector_store(self):
        """Create and persist the vector store."""
        if not self.embeddings:
            print("Embeddings not available. Using simple keyword search instead.")
            return
        
        documents = self._create_documents()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # Create and persist vector store
        os.makedirs(VECTORDB_DIR, exist_ok=True)
        try:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=VECTORDB_DIR
            )
            # self.vectorstore.persist()
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vectorstore = None
    
    def load_data(self, file_path: str = os.path.join(DATA_DIR, "assessments.json")):
        """Load assessment data from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                self.assessments = json.load(f)
            self._create_vector_store()
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def extract_requirements(self, query: str) -> Dict[str, Any]:
        """Extract requirements from the query using LLM."""
        if not OPENAI_API_KEY:
            # Simple keyword extraction for demo
            requirements = {}
            
            # Extract duration requirement
            if "minutes" in query or "mins" in query:
                duration_words = query.split()
                for i, word in enumerate(duration_words):
                    if word.isdigit() and i > 0 and (duration_words[i-1] == "than" or duration_words[i+1] in ["minutes", "mins"]):
                        requirements["max_duration"] = int(word)
                        break
            
            # Extract job level
            if "entry" in query.lower():
                requirements["job_level"] = "Entry-Level"
            elif "senior" in query.lower():
                requirements["job_level"] = "Senior"
            elif "mid" in query.lower():
                requirements["job_level"] = "Mid-Professional"
            
            # Extract skills/programming languages
            programming_languages = ["java", "python", "javascript", "sql", "c++", "c#"]
            requirements["skills"] = [lang for lang in programming_languages if lang.lower() in query.lower()]
            
            return requirements
        else:
            # Use LLM for requirement extraction
            try:
                llm = ChatOpenAI(temperature=0)
                prompt = ChatPromptTemplate.from_template("""
                Extract the key requirements from the following job query:
                
                "{query}"
                
                Return a JSON with the following fields if mentioned:
                - max_duration: Maximum assessment duration in minutes (integer)
                - job_level: Job level (Entry-Level, Mid-Professional, Senior)
                - skills: List of required skills or programming languages
                - test_types: Types of assessments needed (Cognitive, Personality, Aptitude, Behavioral)
                - languages: Preferred languages for the assessment
                
                If a field is not mentioned, omit it from the JSON.
                """)
                
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"query": query})
                
                try:
                    return json.loads(result)
                except:
                    print("Error parsing LLM output as JSON")
                    return {}
                
            except Exception as e:
                print(f"Error extracting requirements with LLM: {e}")
                return {}
    
    def search(self, query: str, is_url: bool = False, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant assessments based on a query or job description URL."""
        if is_url:
            try:
                # Load content from URL
                loader = WebBaseLoader(query)
                docs = loader.load()
                query_text = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Error loading URL: {e}")
                return []
        else:
            query_text = query
        
        # Extract requirements from query
        requirements = self.extract_requirements(query_text)
        
        # Search using vector store if available
        if self.vectorstore:
            try:
                # Set up filter based on requirements
                filter_dict = {}
                
                if "max_duration" in requirements:
                    # Filter for assessments with duration <= max_duration
                    max_duration = requirements["max_duration"]
                    docs_with_metadata = self.vectorstore.similarity_search_with_score(
                        query_text, k=20
                    )
                    
                    # Post-process to apply duration filter
                    filtered_docs = [
                        doc for doc, _ in docs_with_metadata 
                        if doc.metadata.get("duration", float("inf")) <= max_duration
                    ]
                    
                    # Further filter by job level if specified
                    if "job_level" in requirements:
                        filtered_docs = [
                            doc for doc in filtered_docs
                            if requirements["job_level"] in doc.metadata.get("job_levels", "").split(", ")
                        ]
                    
                    # Get assessment objects corresponding to the filtered documents
                    results = []
                    seen_urls = set()
                    for doc in filtered_docs[:max_results]:
                        url = doc.metadata.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            for assessment in self.assessments:
                                if assessment["url"] == url:
                                    results.append(assessment)
                                    break
                    
                    return results
                else:
                    # Regular similarity search without duration filter
                    docs = self.vectorstore.similarity_search(query_text, k=max_results)
                    
                    # Get assessment objects corresponding to the documents
                    results = []
                    seen_urls = set()
                    for doc in docs:
                        url = doc.metadata.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            for assessment in self.assessments:
                                if assessment["url"] == url:
                                    results.append(assessment)
                                    break
                    
                    return results
            
            except Exception as e:
                print(f"Error searching with vector store: {e}")
                # Fall back to keyword search
        
        # Simple keyword-based search as fallback
        keyword_results = self._keyword_search(query_text, requirements, max_results)
        return keyword_results
    
    def _keyword_search(self, query: str, requirements: Dict[str, Any], max_results: int = 10) -> List[Dict[str, Any]]:
        """Simple keyword-based search as fallback."""
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Score each assessment based on keyword matches
        scored_assessments = []
        for assessment in self.assessments:
            score = 0
            
            # Check name and description for keywords
            if any(keyword in assessment["name"].lower() for keyword in query_lower.split()):
                score += 5
            if any(keyword in assessment["description"].lower() for keyword in query_lower.split()):
                score += 3
            
            # Check for skill matches
            if "skills" in requirements:
                for skill in requirements["skills"]:
                    if skill.lower() in assessment["description"].lower():
                        score += 4
            
            # Check for job level match
            if "job_level" in requirements and requirements["job_level"] in assessment["job_levels"]:
                score += 3
            
            # Check for duration requirement
            if "max_duration" in requirements and assessment["duration"] <= requirements["max_duration"]:
                score += 2
            elif "max_duration" in requirements:
                score -= 5  # Penalize if exceeds max duration
            
            # Only include if score is positive
            if score > 0:
                scored_assessments.append((assessment, score))
        
        # Sort by score (descending) and return top results
        scored_assessments.sort(key=lambda x: x[1], reverse=True)
        return [assessment for assessment, _ in scored_assessments[:max_results]]

# API Functions
# def create_api():

recommender = SHLRecommender(use_sample_data=True, use_gemini=True)


@app.get("/")
def read_root():
    return {"message": "Welcome to SHL Assessment Recommender API"}

@app.get("/search")
def search_api(
    query: str = Query(..., description="Search query or job description URL"),
    is_url: bool = Query(False, description="Whether the query is a URL"),
    max_results: int = Query(10, description="Maximum number of results to return")
):
    results = recommender.search(query, is_url, max_results)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }

@app.post("/search")
def search_post(item: QueryItem):
    results = recommender.search(item.text, item.is_url, item.max_results)
    return {
        "query": item.text,
        "count": len(results),
        "results": results
    }


    # return app

# Create app at module level
# app = create_api()

# if __name__ == "__main__":
#     # Add command line argument for embedding model choice
#     import argparse
#     parser = argparse.ArgumentParser(description='Run SHL Assessment Recommender')
#     parser.add_argument('--embedding-model', choices=['openai', 'gemini'], default='gemini',
#                         help='Embedding model to use (default: gemini)')
#     args = parser.parse_args()
    
#     use_gemini = args.embedding_model == 'gemini'
    
#     import uvicorn
#     # For local development
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)