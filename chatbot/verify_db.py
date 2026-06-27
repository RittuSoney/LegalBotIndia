import pandas as pd
import os

current_dir = os.path.dirname(__file__)
master_csv = os.path.join(current_dir, 'Master_Indian_Laws_2026.csv')

print("🔍 Inspecting the Master Database...\n")

try:
    df = pd.read_csv(master_csv, encoding='unicode_escape')
    
    # Filter out only the rows tagged as "Comparative Section"
    comparative_rows = df[df['section'] == 'Comparative Section']
    
    print(f"📊 Total Laws in Database: {len(df)}")
    print(f"📊 Total Comparative Mappings Added: {len(comparative_rows)}\n")
    
    if len(comparative_rows) > 0:
        print("👀 Here are 3 random examples of what Gemini will read when searching:")
        print("-" * 50)
        
        # Grab 3 random samples and print the exact text
        samples = comparative_rows['law'].sample(3).tolist()
        for i, text in enumerate(samples):
            print(f"Sample {i+1}:")
            print(text)
            print("-" * 50)
            
except Exception as e:
    print(f"❌ Verification Error: {e}")