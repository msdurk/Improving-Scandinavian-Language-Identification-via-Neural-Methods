import json

# dataset to format a dataset to into fasttext structure

def process_datasets(input_file, output_all, output_short):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_all, 'w', encoding='utf-8') as out_all, \
         open(output_short, 'w', encoding='utf-8') as out_short:
        for line in infile:
            data = json.loads(line)
            text = data['text']
            label = data['languages'][0] 
            formatted_line = f"__label__{label} {text}\n"

            out_all.write(formatted_line)

            if len(text.split()) <= 6:
                out_short.write(formatted_line)


