import pandas as pd
import re

def clean_and_validate_df(df):
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Step 1: Drop rows with completely null values
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Step 2: Check for required columns
    required_columns = ['name', 'url', 'description']
    for col in required_columns:
        # Drop rows where these critical fields are missing
        cleaned_df = cleaned_df.dropna(subset=[col])
    
    # Step 3: Clean text fields
    text_columns = ['name', 'description']
    for col in text_columns:
        # Remove extra whitespace, newlines, and normalize text
        cleaned_df[col] = cleaned_df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)
    
    # Step 4: Clean and format URL
    cleaned_df['url'] = cleaned_df['url'].apply(lambda x: clean_url(x) if pd.notna(x) else x)
    
    # Step 5: Clean and standardize job_levels
    cleaned_df['job_levels'] = cleaned_df['job_levels'].apply(lambda x: clean_list_field(x) if pd.notna(x) else [])
    
    # Step 6: Clean languages field
    cleaned_df['languages'] = cleaned_df['languages'].apply(lambda x: clean_list_field(x) if pd.notna(x) else [])
    
    # Step 7: Ensure duration is an integer
    cleaned_df['duration'] = cleaned_df['duration'].apply(lambda x: clean_duration(x) if pd.notna(x) else None)
    
    # Step 8: Clean test_type
    cleaned_df['test_type'] = cleaned_df['test_type'].apply(lambda x: clean_list_field(x) if pd.notna(x) else [])
    
    # Step 9: Ensure boolean fields are actual booleans
    boolean_columns = ['remote_testing', 'adaptive_irt']
    for col in boolean_columns:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: clean_boolean(x) if pd.notna(x) else False)
    
    # Step 10: Fill remaining NaN values with appropriate defaults
    # Use scalar values for fillna
    cleaned_df = cleaned_df.fillna({
        'job_levels': '',
        'languages': '',
        'duration': 0,
        'test_type': '',
        'remote_testing': False,
        'adaptive_irt': False
    })
    
    # Convert empty strings back to empty lists for list fields
    list_columns = ['job_levels', 'languages', 'test_type']
    for col in list_columns:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: [] if x == '' else x)
    
    return cleaned_df

# Helper functions for cleaning specific field types
def clean_text(text):
    """Clean text by removing extra whitespace, newlines, etc."""
    if not isinstance(text, str):
        return str(text)
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def clean_url(url):
    """Clean and validate URL"""
    if not isinstance(url, str):
        return str(url)
    
    # Remove whitespace
    url = url.strip()
    
    # Ensure URL starts with http:// or https://
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    return url

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
def clean_duration(duration):
    """Ensure duration is an integer"""
    try:
        # Try to convert to int
        return int(float(str(duration).strip()))
    except (ValueError, TypeError):
        # If conversion fails, return 0
        return 0

def clean_boolean(value):
    """Convert various boolean representations to actual boolean"""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        # Convert string representations to boolean
        value = value.lower().strip()
        return value in ('true', 'yes', 'y', '1', 't')
    
    return False
def validate_dataframe(df):
    """Validate the structure of the DataFrame after cleaning"""
    print("Data validation report:")
    print(f"Total rows: {len(df)}")
    
    # Check for any remaining null values
    null_counts = df.isnull().sum()
    print(f"\nNull values by column:\n{null_counts}")
    
    # Check types of each column
    print("\nColumn types:")
    for col in df.columns:
        # For list columns, check first non-empty value
        if col in ['job_levels', 'languages', 'test_type']:
            non_empty = df[df[col].apply(lambda x: len(x) > 0)]
            if len(non_empty) > 0:
                first_val = non_empty.iloc[0][col]
                print(f"  {col}: {type(first_val)} (first non-empty value: {first_val})")
            else:
                print(f"  {col}: No non-empty values")
        else:
            first_val = df.iloc[0][col]
            print(f"  {col}: {type(first_val)} (first value: {first_val})")
    
    # Check for consistency in list fields
    for col in ['job_levels', 'languages', 'test_type']:
        non_list = df[~df[col].apply(lambda x: isinstance(x, list))].shape[0]
        if non_list > 0:
            print(f"\nWARNING: {non_list} rows have non-list values in {col}")
    
    return True

# Usage example
def main():
    # Load the CSV file
    df = pd.read_csv('../assessment_details_cleaned.csv')
    # Load the CSV file with proper string handling for list-like objects
    # df = pd.read_csv('assessments.csv', keep_default_na=False)

    # If needed, convert string representations of lists to actual lists
    for col in ['job_levels', 'languages', 'test_type']:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_field)
    # Clean and validate the DataFrame
    cleaned_df = clean_and_validate_df(df)
    validate_dataframe(cleaned_df)
    
    # Output some stats about the cleaning process
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Rows removed: {len(df) - len(cleaned_df)}")
    
    # Save the cleaned DataFrame
    cleaned_df.to_csv('processed_assessments.csv', index=False)
    print("Cleaned data saved to 'processed_assessments.csv'")

if __name__ == "__main__":
    main()