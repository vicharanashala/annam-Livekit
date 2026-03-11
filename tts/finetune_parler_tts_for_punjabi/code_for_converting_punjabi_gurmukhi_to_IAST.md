from aksharamukha import transliterate
import csv
 
input_file = "/home/jupyter-koustav/fine_tuning/test_p.tsv"
output_file = "/home/jupyter-koustav/fine_tuning/punjabi_roman.csv"
 
count = 0
with open(input_file, 'r', encoding='utf-8') as tsv_file, open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['audio', 'text', 'description'])
    
    for line in tsv_file:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        
        filename = parts[1].strip()
        transcript = parts[2].strip()
        gender = parts[6].strip() if len(parts) > 6 else 'FEMALE'
        
        roman_text = transliterate.process('Gurmukhi', 'IAST', transcript)
        
        if gender == 'MALE':
            description = "A male speaker speaks clearly in Punjabi with a natural tone and moderate pace in a quiet environment."
        else:
            description = "A female speaker speaks clearly in Punjabi with a natural tone and moderate pace in a quiet environment."
        
        audio_path = f"/home/jupyter-koustav/fine_tuning/test_p/{filename}"
        writer.writerow([audio_path, roman_text, description])
        count += 1
 
print(f"Conversion complete! {count} rows written to punjabi_roman.csv")