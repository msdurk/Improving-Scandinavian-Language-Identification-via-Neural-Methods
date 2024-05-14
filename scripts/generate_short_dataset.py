import json

max_len = 6

def filter_short_texts(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            # Check if the 'text' field is present and split into words
            if 'text' in data:
                words = data['text'].split()
                if len(words) <= max_len:
                    json.dump(data, outfile)
                    outfile.write('\n')

# Example usage
input_filename = '/itf-fi-ml/home/liseche/exam_IN5550/data/gold_dev.jsonl'
output_filename = 'gold_dev_short.jsonl'
filter_short_texts(input_filename, output_filename)
