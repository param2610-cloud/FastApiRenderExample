import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import dotenv
import sys

# Load environment variables
dotenv.load_dotenv()

# Setup paths
DATA_PATH = "./assessment_details_cleaned.csv"
OUTPUT_PATH = "../data/shl_optimized_vector_db_new"

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

def main():
    print("Starting vector database rebuild...")
    
    # Load data
    print("Loading data from CSV...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find file {DATA_PATH}")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Process list fields
    print("Processing data fields...")
    for col in ['job_levels', 'languages', 'test_type']:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_field)
    
    # Prepare documents
    print("Preparing documents...")
    documents = prepare_documents(df)
    print(f"Created {len(documents)} documents")
    
    # Create embeddings
    print("Initializing embeddings...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        return
    
    # Create vector store
    print(f"Creating vector store at {OUTPUT_PATH}...")
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=OUTPUT_PATH
        )
        print("Vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return
    
    print("Rebuild complete!")

if __name__ == "__main__":
    main()