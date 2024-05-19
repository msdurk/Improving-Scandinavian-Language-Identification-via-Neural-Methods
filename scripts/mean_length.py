import argparse

def mean_sentence_length_from_file(filename):
    sentence_lengths = []
    
    try:
        # Open the file and process each line
        with open(filename, 'r') as file:
            for line in file:
                # Check if "Sentence: " is in the current line
                if "Sentence: " in line:
                    # Extract the part after "Sentence: "
                    start_index = line.index("Sentence: ") + len("Sentence: ")
                    sentence_part = line[start_index:]
                    words = sentence_part.split(" ")
                    # Append the length of this part to the list
                    sentence_lengths.append(len(words))
        
        # Calculate the mean length of the sentence parts
        if sentence_lengths:
            return sum(sentence_lengths) / len(sentence_lengths)
        else:
            return 0  # In case there are no "Sentence: " parts
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Calculate the mean length of the parts of lines after 'Sentence: ' in a text file.")
    
    # Add an argument for the input file
    parser.add_argument('filename', type=str, help='The path to the text file to be processed')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Use the provided filename to calculate the mean length
    mean_length = mean_sentence_length_from_file(args.filename)
    
    # Print the result
    print(f"The mean length of the parts after 'Sentence: ' is: {mean_length}")

if __name__ == "__main__":
    main()