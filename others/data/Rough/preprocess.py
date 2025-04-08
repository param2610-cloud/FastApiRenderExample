import pandas as pd
import json

def preprocess_csv_to_json(input_file, output_file):
    """
    Convert assessment details CSV to structured JSON format
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Initialize list to store JSON records
    json_records = []
    
    # Process each row
    for _, row in df.iterrows():
        # Handle test_type conversion (from comma-separated string to single string)
        test_type = ''.join(str(row['test_type']).split(',')) if pd.notna(row['test_type']) else ""
        
        # Create JSON record
        record = {
            "name": row['name'],
            "url": row['url'],
            "description": row['description'],
            "job_levels": row['job_levels'].split(', ') if pd.notna(row['job_levels']) else [],
            "languages": row['languages'].split(', ') if pd.notna(row['languages']) else [],
            "duration": int(row['duration']) if pd.notna(row['duration']) else None,
            "test_type": test_type,
            "remote_testing": row['remote_testing'] == 'True',
            "adaptive_irt": row['adaptive_irt'] == 'True'
        }
        
        json_records.append(record)
    
    # Write JSON output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_records, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(json_records)} records and saved to {output_file}")
    return json_records

# Usage
if __name__ == "__main__":
    preprocess_csv_to_json("assessment_details.csv", "assessment_details.json")