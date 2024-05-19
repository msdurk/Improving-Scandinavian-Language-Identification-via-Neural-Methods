import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_text_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as file:
        content = file.read()
    records = content.strip().split('\n\n')

    print(f"Total records found: {len(records)}")

    for record in records:
        lines = record.strip().split('\n')
        sentence = None
        predicted = set()
        target = set()
        error = None
        sentence_length = 0

        for line in lines:
            if 'Sentence:' in line:
                parts = line.split('Sentence:', 1)
                if len(parts) > 1:
                    sentence = parts[1].strip()
                    sentence_length = len(sentence.split())
            elif 'Predicted languages:' in line:
                parts = line.split('Predicted languages:', 1)
                if len(parts) > 1:
                    predicted = set(parts[1].strip().strip('{}').split(', '))
            elif 'Target languages:' in line:
                parts = line.split('Target languages:', 1)
                if len(parts) > 1:
                    target = set(parts[1].strip().strip('{}').split(', '))
            elif 'Error:' in line:
                parts = line.split('Error:', 1)
                if len(parts) > 1:
                    error = parts[1].strip().lower()
            elif 'error:' in line:
                parts = line.split('error:', 1)
                if len(parts) > 1:
                    error = parts[1].strip().lower()

        if sentence and error and error != "data":
            data.append([sentence, sentence_length, predicted, target, error])

    columns = ['Sentence', 'Sentence Length', 'Predicted Languages', 'Target Languages', 'Error Type']
    df = pd.DataFrame(data, columns=columns)
    print(f"Number of valid records: {len(df)}")
    return df

filepath = '/Users/victorialangoe/Documents/Documents - Victoria’s MacBook Pro/UiO/IN5550/exam_IN5550/annotated_documnets/annotated_fasttext_original.txt'
df = parse_text_file(filepath)

df['Error Type'] = df['Error Type'].map({'feil språk': 'Wrong Language', 'tvetydig': 'Ambiguous'})

df['Error Type'] = pd.Categorical(df['Error Type'], categories=['Wrong Language', 'Ambiguous'])

plt.figure(figsize=(12, 8))
sns.boxplot(x='Error Type', y='Sentence Length', data=df, order=['Wrong Language', 'Ambiguous'])

error_types = ['Wrong Language', 'Ambiguous']
colors = ['red', 'purple']

for i, (error_type, color) in enumerate(zip(error_types, colors)):
    mean_length = df[df['Error Type'] == error_type]['Sentence Length'].mean()
    total_count = df[df['Error Type'] == error_type].shape[0]
    plt.plot(i, mean_length, 'o', markersize=8, color=color, label=f"Mean for {error_type}: {mean_length:.2f}")
    print(f"{error_type} - Mean Sentence Length: {mean_length:.2f}, Total Count: {total_count}")

plt.title('Fasttext - Sentence Length vs. Error Type')
plt.xlabel('Error Type')
plt.ylabel('Length of Sentence (in words)')
plt.xticks(range(len(error_types)), error_types)
plt.xticks(rotation=45)

plt.legend()

plt.savefig('sentence_length_vs_error_type.jpg', format='jpeg')

plt.close()
