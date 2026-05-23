#!/usr/bin/env python3
"""
Script to format SimpleWiki data from JSON format to CSV with columns:
index_normal,index_simple,original,simplified
"""

import json
import csv
import sys
from pathlib import Path

def format_simplewiki_data(input_file, output_file):
    """
    Convert SimpleWiki JSONL data to CSV format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output CSV file
    """
    try:
        entries = []
        
        # Read JSONL file (one JSON object per line)
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                        continue
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            
            # Write header
            writer.writerow(['index_normal', 'index_simple', 'original', 'simplified'])
            
            # Process each entry
            for entry in entries:
                index = entry.get('index', '')
                hard_sentences = entry.get('hard', {}).get('sentences', [])
                easy_sentences = entry.get('easy', {}).get('sentences', [])
                
                # Create pairs between hard and easy sentences
                max_pairs = max(len(hard_sentences), len(easy_sentences))
                
                for i in range(max_pairs):
                    hard_sent = hard_sentences[i] if i < len(hard_sentences) else ''
                    easy_sent = easy_sentences[i] if i < len(easy_sentences) else ''
                    
                    writer.writerow([index, i, hard_sent, easy_sent])
        
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Processed {len(entries)} entries")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default paths
    input_path = Path("/Users/mac/Desktop/Thesis/data/data_for_training/simplewiki-en_sentences")
    output_path = Path("/Users/mac/Desktop/Thesis/data/data_for_training/aligned_pairs.csv")
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    format_simplewiki_data(input_path, output_path)
