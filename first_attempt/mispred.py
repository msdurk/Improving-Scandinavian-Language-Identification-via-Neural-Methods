def calculate_mean_sentence_length(file_path):
    total_length = 0
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('Sentence:'):
                sentence = line.split('Sentence:')[1].strip()
                sentence = sentence.split('.')[0]
                
                sentence_length = len(sentence.split()) 
                total_length += sentence_length
                count += 1

    if count > 0:
        mean_length = total_length / count
    else:
        mean_length = 0
    
    return mean_length

file_path = 'lise.txt'

mean_sentence_length = calculate_mean_sentence_length(file_path)
print(f"The mean length of the sentences is: {mean_sentence_length:.2f} words")
