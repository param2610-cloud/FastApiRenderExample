import os
import re
import pandas as pd
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(title="Assessment Recommendation System API", 
              description="API for searching and recommending SHL assessments",
              version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-recommendation-engine.vercel.app"],  # Allow specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Define response models
class AssessmentResult(BaseModel):
    name: str
    url: str
    description: str
    duration: float
    test_types: List[str]

class SearchResponse(BaseModel):
    search_query: str
    original_query: str
    is_url: bool
    job_description_url: Optional[str] = None
    results: List[AssessmentResult]

def clean_list_field(field):
    """Clean list fields that might be string representations of lists or comma-separated strings"""
    if pd.isna(field) or field == '':
        return []
        
    # If it's already a list, return it
    if isinstance(field, list):
        return field
        
    # If it looks like a string representation of a list: ['item1', 'item2']
    if isinstance(field, str) and field.startswith('[') and field.endswith(']'):
        try:
            # Try to eval it safely to convert string representation to actual list
            # Remove quotes for safer eval
            clean_str = field.replace("'", '"')
            result = eval(clean_str)
            if isinstance(result, list):
                return result
        except:
            pass
    
    # Otherwise, treat as comma-separated string
    if isinstance(field, str):
        return [item.strip() for item in field.split(',') if item.strip()]
    
    return []

def get_duration_range(duration):
    """Categorize duration into ranges for easier filtering"""
    try:
        duration = float(duration)
        if duration <= 15:
            return "very_short"
        elif duration <= 30:
            return "short"
        elif duration <= 45:
            return "medium"
        elif duration <= 60:
            return "standard"
        else:
            return "long"
    except (ValueError, TypeError):
        return "unknown"

def prepare_documents(df):
    """Convert DataFrame rows to Document objects with metadata"""
    documents = []
    
    # Create mappings of all possible values for each category
    test_type_mapping = {
        'A': 'Ability & Aptitude',
        'B': 'Biodata & Situational Judgment',
        'C': 'Competencies',
        'D': 'Development and 360',
        'E': 'Assessment Exercises',
        'K': 'Knowledge & Skills',
        'P': 'Personality & Behavior',
        'S': 'Simulation'
    }
    
    # Get all unique values from the dataframe first (for job levels and languages)
    all_job_levels = set()
    all_languages = set()
    
    for _, row in df.iterrows():
        job_levels = row['job_levels'] if isinstance(row['job_levels'], list) else []
        languages = row['languages'] if isinstance(row['languages'], list) else []
        
        for level in job_levels:
            all_job_levels.add(level.lower())
        for lang in languages:
            all_languages.add(lang.lower())
    
    # Now process each row with all possible values in mind
    for _, row in df.iterrows():
        # Handle lists properly - they're already processed by the clean_list_field function
        job_levels = row['job_levels'] if isinstance(row['job_levels'], list) else []
        languages = row['languages'] if isinstance(row['languages'], list) else []
        test_types = row['test_type'] if isinstance(row['test_type'], list) else []
        
        # Create same page content as before
        job_levels_str = ' and '.join(job_levels) if job_levels else 'various job levels'
        languages_str = ' and '.join(languages) if languages else 'multiple languages'
        test_types_str = ', '.join(test_types) if test_types else 'various assessments'
        
        # Create formatted test type descriptions
        test_type_descriptions = []
        test_type_categories = []
        
        for test_type in test_types:
            test_type = test_type.upper() if isinstance(test_type, str) else test_type
            description = test_type_mapping.get(test_type, f"Unknown ({test_type})")
            test_type_descriptions.append(f"{test_type}: {description}")
            
            if test_type in test_type_mapping:
                test_type_categories.append(test_type_mapping[test_type])
        
        test_types_detailed = ", ".join(test_type_descriptions) if test_type_descriptions else "No test types specified"
        
        # Create the page content
        page_content = (
            f"{row['name']}: A {job_levels_str} position in {languages_str}. "
            f"Job Description: {row['description']} "
            f"Test Types: {test_types_str} "
            f"Detailed Test Types: {test_types_detailed} "
            f"Duration: {row['duration']} minutes "
            f"Remote Testing: {'Available' if row['remote_testing'] else 'Not available'} "
            f"Adaptive Testing: {'Yes' if row['adaptive_irt'] else 'No'}"
        )
        
        # Start with basic metadata using simple types
        metadata = {
            "name": str(row['name']),
            "url": str(row['url']),
            "description": str(row['description']),
            "duration": float(row['duration']) if isinstance(row['duration'], (int, float)) or 
                      (isinstance(row['duration'], str) and row['duration'].replace('.', '', 1).isdigit()) else 0.0,
            "remote_testing": bool(row['remote_testing']),
            "adaptive_irt": bool(row['adaptive_irt']),
            "search_keywords": " ".join([str(row['name']), str(row['description']), 
                                        *[str(level) for level in job_levels],
                                        *[str(lang) for lang in languages], 
                                        *[str(tt) for tt in test_types]]).lower(),
        }
        
        # Add duration range as string
        metadata["duration_range"] = get_duration_range(row['duration'])
        
        # Add job level boolean flags
        for level in all_job_levels:
            metadata[f"job_level_{level.replace(' ', '_').replace('-', '_')}"] = any(jl.lower() == level for jl in job_levels) 
        
        # Add language boolean flags
        for lang in all_languages:
            metadata[f"language_{lang.replace(' ', '_').replace('-', '_').replace('#', 'sharp').replace('+', 'plus')}"] = any(l.lower() == lang for l in languages)
        
        # Add test type boolean flags - both for codes and categories
        for code in test_type_mapping:
            metadata[f"test_type_{code}"] = code in [tt.upper() if isinstance(tt, str) else tt for tt in test_types]
        
        # Add category flags
        metadata["contains_cognitive"] = any(cat in ["Ability & Aptitude", "Knowledge & Skills"] for cat in test_type_categories)
        metadata["contains_personality"] = any(cat in ["Personality & Behavior"] for cat in test_type_categories)
        metadata["contains_technical"] = any(cat in ["Knowledge & Skills", "Simulation"] for cat in test_type_categories)
        metadata["contains_soft_skill"] = any(cat in ["Competencies", "Biodata & Situational Judgment", "Personality & Behavior"] for cat in test_type_categories)
        
        # Add duration-based flags for faster filtering
        metadata["duration_under_30"] = metadata["duration"] <= 30
        metadata["duration_under_45"] = metadata["duration"] <= 45
        metadata["duration_under_60"] = metadata["duration"] <= 60
        
        # Create document with flat metadata structure
        document = Document(
            page_content=page_content,
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents

def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """Extract metadata filters from a natural language query."""
    
    filter_conditions = []
    
    # Job level detection
    job_levels = [
        "analyst", "director", "entry-level", "executive", "front line manager",
        "general population", "graduate", "manager", "mid-professional", 
        "professional individual contributor", "supervisor"
    ]
    
    for level in job_levels:
        if re.search(r'\b' + re.escape(level) + r'\b', query.lower()):
            filter_conditions.append(
                {"job_level_{0}".format(level.replace(' ', '_').replace('-', '_')): True}
            )
    
    # Test type categories detection
    if re.search(r'\b(cognitive|ability|aptitude)\b', query.lower()):
        filter_conditions.append({"contains_cognitive": True})
    
    if re.search(r'\b(personality|behavior|behaviour)\b', query.lower()):
        filter_conditions.append({"contains_personality": True})
        
    if re.search(r'\b(technical|knowledge|skill)\b', query.lower()):
        filter_conditions.append({"contains_technical": True})
        
    if re.search(r'\b(soft skill|competenc|situational|judgment)\b', query.lower()):
        filter_conditions.append({"contains_soft_skill": True})
    
    # Duration detection
    duration_match = re.search(r'(\d+)\s*(min|mins|minutes)', query.lower())
    if duration_match:
        minutes = int(duration_match.group(1))
        if minutes <= 15:
            filter_conditions.append({"duration_range": "very_short"})
        elif minutes <= 30:
            filter_conditions.append({"duration_range": "short"})
        elif minutes <= 45:
            filter_conditions.append({"duration_under_45": True})
        elif minutes <= 60:
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
    
    for lang in languages:
        if re.search(r'\b' + re.escape(lang) + r'\b', query.lower()):
            clean_lang = lang.replace(' ', '_').replace('-', '_')
            filter_conditions.append({f"language_{clean_lang}": True})
    
    # Remote testing and adaptive features
    if "remote" in query.lower():
        filter_conditions.append({"remote_testing": True})
        
    if "adaptive" in query.lower():
        filter_conditions.append({"adaptive_irt": True})
    
    # Return proper ChromaDB filter structure
    if not filter_conditions:
        return None  # No filters found
    elif len(filter_conditions) == 1:
        return filter_conditions[0]  # Single filter
    else:
        return {"$or": filter_conditions}  # Multiple filters combined with OR

def extract_url_from_query(query):
    """Extract URLs from the user query."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, query)
    return urls[0] if urls else None

def extract_job_description(url):
    """Extract job description from a job listing webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # First try to find job description by common class names or IDs
        job_desc_selectors = [
            'div.description', 'div.job-description', '#job-description',
            '.job-details', '.description', '[data-test="job-description"]',
            'section.description', 'div.details', '.details-pane', 
            '.job-desc', '.show-more-less-html'
        ]
        
        for selector in job_desc_selectors:
            job_desc = soup.select_one(selector)
            if job_desc:
                return job_desc.get_text(separator='\n', strip=True)
        
        # If specific selectors fail, use a more generic approach
        # Find sections with job-related terms in them
        job_keywords = ['responsibilities', 'requirements', 'qualifications', 'about the job', 'job summary', 'what you&nbspll do', 'what we&nbspre looking for']
        
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            heading_text = heading.get_text().lower()
            if any(keyword in heading_text for keyword in job_keywords):
                # Get the next sibling elements which likely contain the job description
                description = []
                current = heading.find_next_sibling()
                while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                    if current.get_text(strip=True):
                        description.append(current.get_text(strip=True))
                    current = current.find_next_sibling()
                if description:
                    return '\n'.join(description)
        
        # If all else fails, get the main content area and try to extract job details
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        
        # Last resort: just get the page title and any text
        title = soup.title.string if soup.title else "Job Listing"
        return f"{title}{soup.get_text(strip=True)[:2000]}"
        
    except Exception as e:
        return f"Error extracting job description: {str(e)}"

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
    
    model = GoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
    response = model.invoke(prompt)
    return response.strip()

def search_assessments(query, persist_directory="../shl_optimized_vector_db"):
    """Search for assessments based on the query."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Extract any time constraints from the query
    time_pattern = r'(\d+)\s*minutes'
    time_match = re.search(time_pattern, query.lower())
    max_duration = None
    
    if time_match:
        max_duration = int(time_match.group(1))
    
    # First try with filters
    filters = extract_filters_from_query(query)
    if filters:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter=filters
        )
        # Convert to Document objects
        filtered_results = [doc for doc, _ in results]
    else:
        # If no filters, perform standard search
        filtered_results = vector_store.similarity_search(
            query=query,
            k=5
        )
    
    # If we have time constraints, filter results manually
    if max_duration:
        duration_filtered = []
        for doc in filtered_results:
            duration = doc.metadata.get('duration')
            if duration and float(duration) <= max_duration:
                duration_filtered.append(doc)
        return duration_filtered if duration_filtered else filtered_results[:3]  # Return at least something
    
    return filtered_results

def process_user_query(user_query, persist_directory="../shl_optimized_vector_db"):
    """Process user query with URL to extract job description and search for assessments."""
    # Extract URL from query
    url = extract_url_from_query(user_query)
    if not url:
        # If no URL, treat as direct search query
        print("No URL found. Treating as direct search query.")
        results = search_assessments(user_query, persist_directory)
        
        # Format results
        formatted_results = "\nRecommended Assessments:\n" + "-" * 40 + "\n"
        if not results:
            formatted_results += "No matching assessments found."
        else:
            for i, result in enumerate(results, 1):
                formatted_results += f"Assessment {i}:\n"
                formatted_results += f"Name: {result.metadata.get('name', 'N/A')}\n"
                formatted_results += f"Duration: {result.metadata.get('duration', 'N/A')} minutes\n"
                
                # Format test types
                test_types = [t.replace('test_type_', '') for t in result.metadata 
                              if t.startswith('test_type_') and result.metadata[t]]
                formatted_results += f"Test types: {', '.join(test_types) if test_types else 'N/A'}\n"
                
                formatted_results += f"URL: {result.metadata.get('url', 'N/A')}\n"
                formatted_results += f"Description: {result.page_content[:200]}...\n\n"
        
        return f"Search Query: \"{user_query}\"\n\n{formatted_results}"
    
    # Extract job description from URL
    print(f"Extracting job description from URL: {url}")
    job_description = extract_job_description(url)
    if job_description.startswith("Error"):
        return job_description
    
    # Generate search query based on job description
    print("Generating search query from job description...")
    search_query = generate_search_query(job_description)
    
    # Incorporate any time constraints from the original query
    time_pattern = r'(\d+)\s*minutes'
    time_match = re.search(time_pattern, user_query.lower())
    if time_match:
        max_duration = time_match.group(0)
        if "time" not in search_query.lower() and "minute" not in search_query.lower():
            search_query += f" Assessment duration less than {max_duration}."
    
    try:
        # Search for assessments
        print(f"Searching for assessments with query: {search_query}")
        results = search_assessments(search_query, persist_directory)
        
        # Format results
        formatted_results = "\nRecommended Assessments:\n" + "-" * 40 + "\n"
        if not results:
            formatted_results += "No matching assessments found."
        else:
            for i, result in enumerate(results, 1):
                formatted_results += f"Assessment {i}:\n"
                formatted_results += f"Name: {result.metadata.get('name', 'N/A')}\n"
                formatted_results += f"Duration: {result.metadata.get('duration', 'N/A')} minutes\n"
                
                # Format test types
                test_types = [t.replace('test_type_', '') for t in result.metadata 
                              if t.startswith('test_type_') and result.metadata[t]]
                formatted_results += f"Test types: {', '.join(test_types) if test_types else 'N/A'}\n"
                
                formatted_results += f"URL: {result.metadata.get('url', 'N/A')}\n"
                formatted_results += f"Description: {result.page_content[:200]}...\n\n"
        
        # Return summary
        summary = f"""
Job Description URL: {url}

Generated Search Query: "{search_query}"

{formatted_results}
        """
        return summary
        
    except Exception as e:
        return f"Error searching for assessments: {str(e)}"

def prepare_data_pipeline(df_path, persist_directory="database/shl_vector_db"):
    """Prepare the data pipeline from CSV to vector database."""
    # Load the dataframe
    print(f"Loading data from {df_path}...")
    df = pd.read_csv(df_path)
    
    # Clean list fields
    print("Cleaning list fields...")
    for col in ['job_levels', 'languages', 'test_type']:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_field)
    
    # Extract unique values for reporting
    print("Extracting unique values...")
    job_levels_unique = set()
    languages_unique = set()
    test_types_unique = set()
    
    for job_levels_list in df['job_levels']:
        if isinstance(job_levels_list, list):
            for level in job_levels_list:
                job_levels_unique.add(level)
    
    for languages_list in df['languages']:
        if isinstance(languages_list, list):
            for language in languages_list:
                languages_unique.add(language)
    
    for test_types_list in df['test_type']:
        if isinstance(test_types_list, list):
            for test_type in test_types_list:
                test_types_unique.add(test_type)
    
    # Convert sets to sorted lists for better readability
    job_levels_unique = sorted(list(job_levels_unique))
    languages_unique = sorted(list(languages_unique))
    test_types_unique = sorted(list(test_types_unique))
    
    print(f"Found {len(job_levels_unique)} unique job levels")
    print(f"Found {len(languages_unique)} unique languages")
    print(f"Found {len(test_types_unique)} unique test types")
    
    # Prepare documents
    print("Preparing documents...")
    documents = prepare_documents(df)
    print(f"Created {len(documents)} documents")
    
    # Create embeddings and vector store
    print("Creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created and persisted to {persist_directory}")
    return vector_store

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Natural language query or job description URL"),
    is_url: bool = Query(False, description="Whether the query is a URL to a job listing"),
    max_results: int = Query(5, description="Maximum number of results to return", ge=1, le=10)
):
    """
    Search for assessments based on a natural language query or job description URL.
    
    - If is_url=True, the system will extract the job description from the URL and generate a search query.
    - If is_url=False, the query will be directly used to search for assessments.
    """
    # Process query based on whether it's a URL or direct query
    if is_url:
        # Extract URL from query if not already a URL
        url = query if query.startswith(('http://', 'https://')) else extract_url_from_query(query)
        
        if not url:
            return SearchResponse(
                search_query="",
                original_query=query,
                is_url=is_url,
                results=[]
            )
            
        # Extract job description from URL
        job_description = extract_job_description(url)
        if job_description.startswith("Error"):
            return SearchResponse(
                search_query=job_description,
                original_query=query,
                is_url=is_url,
                job_description_url=url,
                results=[]
            )
            
        # Generate search query based on job description
        search_query = generate_search_query(job_description)
        
        # Incorporate any time constraints from the original query
        time_pattern = r'(\d+)\s*minutes'
        time_match = re.search(time_pattern, query.lower())
        if time_match:
            max_duration = time_match.group(0)
            if "time" not in search_query.lower() and "minute" not in search_query.lower():
                search_query += f" Assessment duration less than {max_duration}."
    else:
        # Use the query directly
        url = None
        search_query = query
    
    try:
        # Search for assessments using existing function
        results = search_assessments(search_query, persist_directory="database/shl_vector_db")
        
        # Format results according to the response model
        formatted_results = []
        for result in results[:max_results]:  # Limit to max_results
            # Extract test types
            test_types = [t.replace('test_type_', '') for t in result.metadata 
                         if t.startswith('test_type_') and result.metadata[t]]
            
            # Create AssessmentResult object
            assessment = AssessmentResult(
                name=result.metadata.get('name', 'N/A'),
                url=result.metadata.get('url', 'N/A'),
                description=result.page_content[:500],  # Limit description length
                duration=float(result.metadata.get('duration', 0)),
                test_types=test_types
            )
            formatted_results.append(assessment)
        
        # Return the response
        return SearchResponse(
            search_query=search_query,
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=formatted_results
        )
        
    except Exception as e:
        # Return empty results with error message as search query
        return SearchResponse(
            search_query=f"Error searching for assessments: {str(e)}",
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=[]
        )

def main():
    """Main function to handle command line arguments and run the program."""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description='Assessment Recommendation System')
    parser.add_argument('--prepare', type=str, help='Path to CSV file to prepare data pipeline')
    parser.add_argument('--query', type=str, help='Query string for assessment search')
    parser.add_argument('--db_path', type=str, default="database/shl_vector_db", 
                      help='Path to the vector database directory')
    parser.add_argument('--api', action='store_true', help='Run as FastAPI server')
    
    args = parser.parse_args()
    
    if args.prepare:
        # Prepare data pipeline
        prepare_data_pipeline(args.prepare, args.db_path)
    elif args.query:
        # Process user query
        result = process_user_query(args.query, args.db_path)
        print(result)
    elif args.api:
        # Run as FastAPI server
        print("Starting FastAPI server...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Interactive mode
        print("Assessment Recommendation System")
        print("Enter 'exit' to quit")
        while True:
            query = input("\nEnter your query (or URL to job listing): ")
            if query.lower() == 'exit':
                break
                
            result = process_user_query(query, args.db_path)
            print(result)

if __name__ == "__main__":
    main()