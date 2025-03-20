import pandas as pd
import csv

def parse_reviews(input_path, output_path):
    review_numbers = []
    review_texts = []
    labels = []
    
    # Read text file
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("|")
            if len(parts) == 3:
                review_number = parts[0].strip('"')
                review_text = parts[1].strip().strip('"').replace(",", " ")
                label = parts[2].strip()
                
                review_numbers.append(review_number)
                review_texts.append(review_text)
                labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Review Number": review_numbers,
        "Review Text": review_texts,
        "Label": labels
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONE)
    
    return df
