
class CustomOpenXLMModel(torch.nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.args = args
        model_path = '/fp/projects01/ec30/mathisdu/gold/e3'
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.languages = ['nb', 'nn', 'da', 'sv', 'other']
        
        path = "https://data.statmt.org/lid/lid201-model.bin.gz"

        # download the model if it does not exist
        if not os.path.exists("lid201-model.bin"):
            urllib.request.urlretrieve(path, "lid201-model.bin.gz")
            os.system("gunzip lid201-model.bin.gz")
        
        self.open_lid = fasttext.load_model("lid201-model.bin")

    def preds_fasttext(self, text: str):
                # Define target labels
        label_mapping = {
            "nob_": "nb",
            "nno_": "nn",
            "dan_": "da",
            "swe_": "sv",
        }
        target_labels = set(label_mapping.values())

        # Initialize results dictionary with target languages set to 0 probability
        results = {label: 0.0 for label in target_labels}
        results['other'] = 0.0  # Initialize 'other' category

        # Get top 5 predictions
        prediction, probabilities = self.open_lid.predict(text, k=5)

        # Track the highest non-target probability
        highest_non_target_prob = 0.0

        for label, probability in zip(prediction, probabilities):
            label_clean = label.replace("__label__", "")
            found = False

            # Check and map label if it's a target language
            for key, value in label_mapping.items():
                if label_clean.startswith(key):
                    results[value] = probability
                    found = True
                    break

            # Track highest probability for non-target languages
            if not found and probability > highest_non_target_prob:
                highest_non_target_prob = probability

        # Assign highest non-target probability to 'other' if no specific target language probabilities were higher
        if results['other'] == 0.0:
            results['other'] = highest_non_target_prob
        
        return results


    def preds_model(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract logits and apply softmax to get probabilities
        logits = outputs.logits
        probs = softmax(logits, dim=-1).squeeze()

        # Assuming you know the mapping from model output indices to languages
        label_mapping = {
            0: "nb",
            1: "nn",
            2: "da",
            3: "sv",
            4: "other"
        }
        
        # Create a dictionary to hold the results with probabilities
        results = {}
        for idx, prob in enumerate(probs):
            label = label_mapping.get(idx, "unknown")
            results[label] = prob.item()  # Convert from tensor to float
    
        return results
