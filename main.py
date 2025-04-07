from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import dotenv


dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions or natural language queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure Gemini API (make sure to set GOOGLE_API_KEY in environment variables)
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {str(e)}")
    raise

# Embeddings setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector store path
VECTOR_STORE_PATH = "/data/shl_optimized_vector_db_new"

# Response models
class AssessmentResult(BaseModel):
    name: str
    url: str
    duration: int
    test_types: List[str]
    description: str
    remote_testing: bool
    adaptive_irt: bool

class SearchResponse(BaseModel):
    query: str
    processed_query: Optional[str] = None
    job_url: Optional[str] = None
    results: List[AssessmentResult]

def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """Extract metadata filters from a natural language query."""
    logger.info(f"Extracting filters from query: '{query}'")
    
    filter_conditions = []
    
    # Job level detection
    job_levels = [
        "analyst", "director", "entry-level", "executive", "front line manager",
        "general population", "graduate", "manager", "mid-professional", 
        "professional individual contributor", "supervisor"
    ]
    
    job_level_filters = []
    for level in job_levels:
        if re.search(r'\b' + re.escape(level) + r'\b', query.lower()):
            logger.debug(f"Detected job level: {level}")
            job_level_filters.append(
                {"job_level_{0}".format(level.replace(' ', '_').replace('-', '_')): True}
            )
    
    # Add job levels as OR condition (any of these levels)
    if job_level_filters:
        if len(job_level_filters) == 1:
            logger.debug(f"Adding single job level filter: {job_level_filters[0]}")
            filter_conditions.append(job_level_filters[0])
        else:
            logger.debug(f"Adding OR condition for job levels: {job_level_filters}")
            filter_conditions.append({"$or": job_level_filters})
    
    # Test type categories detection - add each as a separate condition
    if re.search(r'\b(cognitive|ability|aptitude)\b', query.lower()):
        logger.debug("Detected cognitive assessment requirement")
        filter_conditions.append({"contains_cognitive": True})
    
    if re.search(r'\b(personality|behavior|behaviour)\b', query.lower()):
        logger.debug("Detected personality assessment requirement")
        filter_conditions.append({"contains_personality": True})
        
    if re.search(r'\b(technical|knowledge|skill)\b', query.lower()):
        logger.debug("Detected technical assessment requirement")
        filter_conditions.append({"contains_technical": True})
        
    if re.search(r'\b(soft skill|competenc|situational|judgment)\b', query.lower()):
        logger.debug("Detected soft skill assessment requirement")
        filter_conditions.append({"contains_soft_skill": True})
    
    # Duration detection
    duration_match = re.search(r'(\d+)\s*(min|mins|minutes)', query.lower())
    if duration_match:
        minutes = int(duration_match.group(1))
        logger.debug(f"Detected duration constraint: {minutes} minutes")
        if minutes <= 15:
            logger.debug("Adding very_short duration filter")
            filter_conditions.append({"duration_range": "very_short"})
        elif minutes <= 30:
            logger.debug("Adding short duration filter")
            filter_conditions.append({"duration_range": "short"})
        elif minutes <= 45:
            logger.debug("Adding duration_under_45 filter")
            filter_conditions.append({"duration_under_45": True})
        elif minutes <= 60:
            logger.debug("Adding duration_under_60 filter")
            filter_conditions.append({"duration_under_60": True})
        
    # Language detection
    languages = [
        "arabic", "chinese simplified", "chinese traditional", "czech", "danish",
        "dutch", "english", "estonian", "finnish", "flemish", "french", "german",
        "greek", "hungarian", "icelandic", "indonesian", "italian", "japanese",
        "korean", "latvian", "lithuanian", "malay", "norwegian", "polish",
        "portuguese", "romanian", "russian", "serbian", "slovak", "spanish",
        "swedish", "thai", "turkish", "vietnamese"
    ]
    
    language_filters = []
    for lang in languages:
        if re.search(r'\b' + re.escape(lang) + r'\b', query.lower()):
            logger.debug(f"Detected language requirement: {lang}")
            clean_lang = lang.replace(' ', '_').replace('-', '_')
            language_filters.append({f"language_{clean_lang}": True})
    
    # Add languages as OR condition (any of these languages)
    if language_filters:
        if len(language_filters) == 1:
            logger.debug(f"Adding single language filter: {language_filters[0]}")
            filter_conditions.append(language_filters[0])
        else:
            logger.debug(f"Adding OR condition for languages: {language_filters}")
            filter_conditions.append({"$or": language_filters})
    
    # Remote testing and adaptive features
    if "remote" in query.lower():
        logger.debug("Detected remote testing requirement")
        filter_conditions.append({"remote_testing": True})
        
    if "adaptive" in query.lower():
        logger.debug("Detected adaptive testing requirement")
        filter_conditions.append({"adaptive_irt": True})
    
    # Return proper ChromaDB filter structure
    if not filter_conditions:
        logger.info("No filters identified from query")
        return None  # No filters found
    elif len(filter_conditions) == 1:
        logger.info(f"Single filter condition identified: {filter_conditions[0]}")
        return filter_conditions[0]  # Single filter
    else:
        logger.info(f"Multiple filter conditions identified: {filter_conditions}")
        return {"$and": filter_conditions}  # Multiple filters combined with AND

def extract_url_from_query(query):
    """Extract URLs from the user query."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, query)
    if urls:
        logger.info(f"Extracted URL from query: {urls[0]}")
        return urls[0]
    else:
        logger.debug("No URL found in query")
        return None

def extract_job_description(url):
    """Extract job description from a job listing webpage."""
    logger.info(f"Attempting to extract job description from URL: {url}")
    
    try:
        logger.debug("Sending HTTP request with custom headers")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        logger.debug(f"Response received: status code {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # First try to find job description by common class names or IDs
        job_desc_selectors = [
            'div.description', 'div.job-description', '#job-description',
            '.job-details', '.description', '[data-test="job-description"]',
            'section.description', 'div.details', '.details-pane', 
            '.job-desc', '.show-more-less-html'
        ]
        
        logger.debug("Attempting to find job description using common selectors")
        for selector in job_desc_selectors:
            job_desc = soup.select_one(selector)
            if job_desc:
                logger.info(f"Found job description using selector: {selector}")
                return job_desc.get_text(separator='\n', strip=True)
        
        # If specific selectors fail, use a more generic approach
        logger.debug("Common selectors failed, trying keyword-based approach")
        # Find sections with job-related terms in them
        job_keywords = ['responsibilities', 'requirements', 'qualifications', 'about the job', 'job summary', 'what you&nbspll do', 'what we&nbspre looking for']
        
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            heading_text = heading.get_text().lower()
            if any(keyword in heading_text for keyword in job_keywords):
                logger.debug(f"Found potential job section heading: '{heading_text}'")
                # Get the next sibling elements which likely contain the job description
                description = []
                current = heading.find_next_sibling()
                while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                    if current.get_text(strip=True):
                        description.append(current.get_text(strip=True))
                    current = current.find_next_sibling()
                if description:
                    logger.info(f"Extracted job description using keyword: '{heading_text}'")
                    return '\n'.join(description)
        
        # If all else fails, get the main content area and try to extract job details
        logger.debug("Keyword approach failed, trying main content extraction")
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            logger.info("Extracting job details from main content area")
            text = main_content.get_text(separator='\n', strip=True)
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        
        # Last resort: just get the page title and any text
        logger.debug("All structured extraction methods failed, using generic page extraction")
        title = soup.title.string if soup.title else "Job Listing"
        logger.info(f"Using page title and truncated content: '{title}'")
        return "{0}\n{1}".format(title, soup.get_text(separator='\n', strip=True)[:2000])
        
    except Exception as e:
        logger.error(f"Error extracting job description: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error extracting job description: {str(e)}")

def generate_search_query(job_description):
    """Generate a search query based on job description using Gemini."""
    prompt = f"""
    Based on the following job description, create a concise search query to find appropriate
    assessment tests that would help screen candidates for this position.
    
    JOB DESCRIPTION:
    {job_description[:3000]}  # Limit to avoid token limits
    
    Focus on:
    1. Technical skills required
    2. Soft skills mentioned
    3. Any specific assessment requirements
    4. Time constraints for assessments if mentioned
    
    Return ONLY the search query, nothing else.
    """
    
    try:
        model = GoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = model.invoke(prompt)
        logger.info(f"Generated search query: {response}")
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating search query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating search query: {str(e)}")

def search_assessments(query, max_results=10):
    """Search for assessments based on the query."""
    logger.info(f"Searching for assessments with query: '{query}', max_results={max_results}")
    
    try:
        # Load the vector store
        logger.debug(f"Loading vector store from: {VECTOR_STORE_PATH}")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        
        # Extract any time constraints from the query
        time_pattern = r'(\d+)\s*minutes'
        time_match = re.search(time_pattern, query.lower())
        max_duration = None
        
        if time_match:
            max_duration = int(time_match.group(1))
            logger.info(f"Detected time constraint: {max_duration} minutes")
        
        # Get filters from query
        logger.debug("Extracting filters from query")
        filters = extract_filters_from_query(query)
        logger.info(f"Extracted filters: {filters}")
        
        # Use the robust search function
        logger.debug("Initiating robust ChromaDB search")
        results = robust_chromadb_search(
            vector_store=vector_store,
            query=query,
            filters=filters,
            k=max(max_results * 2, 20)
        )
        
        # Post-process results
        logger.debug("Post-processing search results")
        processed_results = []
        
        # Handle different result formats from robust_chromadb_search
        if not results:
            logger.warning("No results returned from search")
            return []
            
        logger.info(f"Processing {len(results)} search results")
        for i, item in enumerate(results):
            try:
                logger.debug(f"Processing result item {i+1}/{len(results)}")
                # Handle different return structures
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                    logger.debug(f"Result item is tuple with document and score: {score}")
                elif isinstance(item, Document):
                    doc, score = item, 1.0
                    logger.debug("Result item is Document without score")
                else:
                    logger.warning(f"Unexpected result format: {type(item)}")
                    continue
                
                # Safely extract and convert duration
                duration = 0
                if 'duration' in doc.metadata:
                    try:
                        duration = int(float(doc.metadata['duration']))
                        logger.debug(f"Extracted duration: {duration} minutes")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting duration: {str(e)}")
                        pass
                
                # Apply duration filter if needed
                if max_duration and duration > max_duration:
                    logger.debug(f"Skipping result: duration {duration} exceeds max {max_duration}")
                    continue
                
                # Extract test types
                test_types = []
                for key in doc.metadata:
                    if key.startswith('test_type_') and doc.metadata[key]:
                        # Remove 'test_type_' prefix
                        test_type = key.replace('test_type_', '')
                        test_types.append(test_type)
                
                logger.debug(f"Extracted test types: {test_types}")
                
                # Create result with safe defaults for all fields
                assessment = AssessmentResult(
                    name=str(doc.metadata.get('name', 'Unknown')),
                    url=str(doc.metadata.get('url', '')),
                    duration=duration,
                    test_types=test_types,
                    description=doc.page_content[:500] if doc.page_content else "",
                    remote_testing=bool(doc.metadata.get('remote_testing', False)),
                    adaptive_irt=bool(doc.metadata.get('adaptive_irt', False))
                )
                logger.debug(f"Created AssessmentResult: {assessment.name}")
                processed_results.append(assessment)
                
                # Stop once we have enough results
                if len(processed_results) >= max_results:
                    logger.info(f"Reached maximum results limit ({max_results})")
                    break
            except Exception as item_error:
                logger.warning(f"Error processing search result item: {str(item_error)}", exc_info=True)
                continue
        
        logger.info(f"Returning {len(processed_results)} processed results")
        return processed_results
    
    except Exception as e:
        logger.error(f"Error searching for assessments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching for assessments: {str(e)}")
        
def robust_chromadb_search(vector_store, query, filters=None, k=10):
    """Perform a ChromaDB search with error handling for different versions and configurations."""
    logger.info(f"Performing robust ChromaDB search with filters: {filters}, k={k}")
    
    try:
        # First attempt: with filters if provided
        if filters:
            logger.debug("Attempting similarity search with filters")
            return vector_store.similarity_search_with_score(query=query, k=k, filter=filters)
        else:
            logger.debug("Attempting similarity search without filters")
            return vector_store.similarity_search_with_score(query=query, k=k)
    except TypeError as e:
        # Handle type errors which might be due to filter structure
        logger.warning(f"Type error in ChromaDB search: {str(e)}. Trying without filters.", exc_info=True)
        try:
            return vector_store.similarity_search_with_score(query=query, k=k)
        except Exception as retry_error:
            logger.error(f"Second attempt failed with error: {str(retry_error)}", exc_info=True)
            raise
    except ValueError as e:
        # Handle value errors which might be due to metadata structure
        logger.warning(f"Value error in ChromaDB search: {str(e)}. Trying without filters.", exc_info=True)
        try:
            return vector_store.similarity_search_with_score(query=query, k=k)
        except Exception as retry_error:
            logger.error(f"Second attempt failed with error: {str(retry_error)}", exc_info=True)
            raise
    except Exception as e:
        # Fall back to basic search with no bells and whistles
        logger.warning(f"Error in ChromaDB search: {str(e)}. Falling back to basic search.", exc_info=True)
        try:
            logger.debug("Attempting basic similarity search without scores")
            results = vector_store.similarity_search(query=query, k=k)
            logger.info(f"Basic search returned {len(results)} results")
            return [(doc, 1.0) for doc in results]  # Convert to (doc, score) format
        except Exception as basic_error:
            # Ultimate fallback, try using the collection directly
            logger.warning(f"Basic search failed: {str(basic_error)}. Falling back to direct collection access.", exc_info=True)
            results = []
            try:
                logger.debug("Attempting direct collection access with query embeddings")
                embeddings_query = embeddings.embed_query(query)
                logger.debug(f"Generated query embedding with {len(embeddings_query)} dimensions")
                
                chroma_results = vector_store._collection.query(
                    query_embeddings=[embeddings_query],
                    n_results=k
                )
                logger.debug("Direct collection query successful")
                
                for i in range(len(chroma_results['documents'][0])):
                    metadata = {}
                    if chroma_results.get('metadatas') and chroma_results['metadatas'][0]:
                        metadata = chroma_results['metadatas'][0][i]
                    
                    # Ensure metadata is a dictionary
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    doc = Document(
                        page_content=chroma_results['documents'][0][i],
                        metadata=metadata
                    )
                    results.append((doc, chroma_results['distances'][0][i] if chroma_results.get('distances') else 1.0))
                
                return results
            except Exception as final_e:
                logger.error(f"All search methods failed: {str(final_e)}", exc_info=True)
                return []

def fallback_search_assessments(query, max_results=10):
    """A simplified search function that avoids complex filter operations."""
    try:
        # Load the vector store
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        
        # Extract time constraint for manual filtering afterward
        time_pattern = r'(\d+)\s*minutes'
        time_match = re.search(time_pattern, query.lower())
        max_duration = None
        if time_match:
            max_duration = int(time_match.group(1))
        
        # Basic search through direct embedding and collection query
        try:
            embedding_vector = embeddings.embed_query(query)
            chroma_results = vector_store._collection.query(
                query_embeddings=[embedding_vector],
                n_results=max(max_results * 3, 30)  # Get more results for manual filtering
            )
            
            results = []
            if chroma_results and 'documents' in chroma_results and len(chroma_results['documents']) > 0:
                for i in range(len(chroma_results['documents'][0])):
                    metadata = {}
                    if chroma_results.get('metadatas') and chroma_results['metadatas'][0]:
                        metadata = chroma_results['metadatas'][0][i]
                    
                    # Ensure metadata is a dictionary
                    if not isinstance(metadata, dict):
                        metadata = {}
                        
                    doc = Document(
                        page_content=chroma_results['documents'][0][i],
                        metadata=metadata
                    )
                    results.append(doc)
            
        except Exception as direct_error:
            logger.warning(f"Direct collection access failed: {str(direct_error)}. Trying basic similarity search.")
            # If direct access fails, try basic similarity search
            try:
                results = vector_store.similarity_search(
                    query=query,
                    k=max(max_results * 3, 30)  # Get more results for manual filtering
                )
            except Exception as basic_error:
                logger.error(f"All search methods failed: {str(basic_error)}")
                return []
        
        # Process results manually
        processed_results = []
        
        for doc in results:
            try:
                # Ensure we have a dictionary for metadata
                if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                
                # Basic extraction of duration
                duration = 0
                if 'duration' in doc.metadata:
                    try:
                        duration = int(float(doc.metadata['duration']))
                    except (ValueError, TypeError):
                        pass
                
                # Skip if above max duration
                if max_duration and duration > max_duration:
                    continue
                
                # Extract test types
                test_types = []
                for key in doc.metadata:
                    if key.startswith('test_type_') and doc.metadata[key]:
                        test_type = key.replace('test_type_', '')
                        test_types.append(test_type)
                
                processed_results.append(AssessmentResult(
                    name=str(doc.metadata.get('name', 'Unknown')),
                    url=str(doc.metadata.get('url', '')),
                    duration=duration,
                    test_types=test_types,
                    description=doc.page_content[:500] if doc.page_content else "",
                    remote_testing=bool(doc.metadata.get('remote_testing', False)),
                    adaptive_irt=bool(doc.metadata.get('adaptive_irt', False))
                ))
                
                if len(processed_results) >= max_results:
                    break
                    
            except Exception as item_error:
                logger.warning(f"Error processing fallback result: {str(item_error)}")
                continue
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in fallback search: {str(e)}")
        return []  # Return empty results rather than raising an exception

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Natural language query or job description URL"),
    is_url: bool = Query(False, description="Whether the query is a URL to a job listing"),
    max_results: int = Query(5, description="Maximum number of results to return", ge=1, le=10)
):
    """
    Search for SHL assessments based on a natural language query or job description URL.
    Returns most relevant assessments based on the input.
    """
    logger.info(f"Received search request: query='{query}', is_url={is_url}, max_results={max_results}")
    
    try:
        processed_query = query
        job_url = None
        
        # If the query is a URL or contains a URL
        if is_url or extract_url_from_query(query):
            # Extract just the URL, not the surrounding text
            job_url = extract_url_from_query(query) if extract_url_from_query(query) else query
            logger.info(f"Extracting job description from URL: {job_url}")
            
            # Extract job description
            job_description = extract_job_description(job_url)
            
            # Generate search query from job description
            processed_query = generate_search_query(job_description)
            
            # Add time constraints if present in original query
            time_pattern = r'(\d+)\s*minutes'
            time_match = re.search(time_pattern, query.lower())
            if time_match:
                max_duration = time_match.group(0)
                if "time" not in processed_query.lower() and "minute" not in processed_query.lower():
                    processed_query += f" Assessment duration less than {max_duration}."
        
        # Rest of the function remains the same...
        # Try with the main search function
        try:
            results = search_assessments(processed_query, max_results)
        except Exception as e:
            logger.warning(f"Primary search failed: {str(e)}. Trying fallback.")
            # If main search fails, try the fallback
            results = fallback_search_assessments(processed_query, max_results)
            
        # If still no results, just provide an empty list
        if not results:
            results = []
        
        return SearchResponse(
            query=query,
            processed_query=processed_query if is_url or extract_url_from_query(query) else None,
            job_url=job_url,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)