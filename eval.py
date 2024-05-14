import random
import time
from typing import List
from tqdm import tqdm
import os
import urllib.request
import json
import torch
import torchmetrics
from smart_open import open


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="nllb", help="The method to use for language identification, supported methods are: random, cld3, nllb, fasttext, openlid, langid, lingua, langdetect")
    parser.add_argument("--dataset", type=str, default="test.jsonl.gz", help="The dataset to use for evaluation")
    return parser.parse_args()


class AbstractLanguageIdentifier:
    def __init__(self, args):
        self.args = args
        self.languages = ["nb", "nn", "da", "sv", "other"]

    # This method should return a list of languages that the given text could be written in
    # A language can be either ("nb", "nn", "da", "sv" or "other")
    def identify(self, text: str) -> List[str]:
        raise NotImplementedError
    

class RandomLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)

    def identify(self, text: str) -> List[str]:
        return [random.choice(self.languages)]


class CLD3LanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import cld3  # pip install pycld3
        self.cld3 = cld3

    def identify(self, text: str) -> List[str]:
        prediction = self.cld3.get_language(text)
        language = prediction.language
        if not prediction.is_reliable:
            return ["other"]
        if language == "no":
            return ["nb"]
        if language not in self.languages:
            return ["other"]
        return [language]


class NllbLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(model_path)

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]


class FasttextLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext
        
        path = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

        # download the model if it does not exist
        if not os.path.exists("lid.176.bin"):
            urllib.request.urlretrieve(path, "lid.176.bin")
        
        self.model = fasttext.load_model("lid.176.bin")

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language == "no":
            return ["nb"]

        if language in self.languages:
            return [language]
        return ["other"]


class OpenlidLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext

        path = "https://data.statmt.org/lid/lid201-model.bin.gz"

        # download the model if it does not exist
        if not os.path.exists("lid201-model.bin"):
            urllib.request.urlretrieve(path, "lid201-model.bin.gz")
            os.system("gunzip lid201-model.bin.gz")
        
        self.model = fasttext.load_model("lid201-model.bin")
    
    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]
    

class LangidLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)     
        import langid  # pip install langid
        self.langid = langid

    def identify(self, text: str) -> List[str]:
        
        prediction = self.langid.classify(text)
        language = prediction[0]
        if language in self.languages:
            return [language]
        return ["other"]
    

class LinguaLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import lingua  # pip install lingua-language-detector
        from lingua import LanguageDetectorBuilder

        self.detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def identify(self, text: str) -> List[str]:
        language = self.detector.detect_language_of(text)
        language = language.iso_code_639_1.name.lower()
        if language in self.languages:
            return [language]
        return ["other"]


class LangdetectLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        from langdetect import detect
        self.detect = detect

    def identify(self, text: str) -> List[str]:
        language = self.detect(text)
        if language == "no":
            return ["nb"]
        if language in self.languages:
            return [language]
        return ["other"]

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class MBertLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        model_path = '/itf-fi-ml/home/liseche/exam_IN5550/models/mbert-finetuned4'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
        self.language_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def identify(self, text: str) -> List[str]:
        predictions = self.language_pipeline(text, max_length=512, truncation=True)
        label_map = {0: 'da', 1: 'nb', 2: 'nn', 3: 'other', 4: 'sv'}
        predicted_label = predictions[0]['label']
        predicted_language = label_map[int(predicted_label.split('_')[1])]
        print(predicted_language)
        return [predicted_language]

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

class RobertaLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        model_path = '/itf-fi-ml/home/liseche/exam_IN5550/results/final_model'
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path, num_labels=5)
        self.language_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def identify(self, text: str) -> List[str]:
        predictions = self.language_pipeline(text, max_length=512, truncation=True)
        label_map = {0: 'da', 1: 'nb', 2: 'nn', 3: 'other', 4: 'sv'}
        predicted_label = predictions[0]['label']
        predicted_language = label_map[int(predicted_label.split('_')[1])]
        print(predicted_language)
        return [predicted_language] 


def print_confusion_matrix(confusion_matrix, supported_languages):
    print("\t\t" + "\t".join(supported_languages))
    for i, row in enumerate(confusion_matrix):
        print("\t" + supported_languages[i], end="\t")
        for cell in row:
            print(int(cell), end="\t")
        print()


# prints the confusion matrix, accuracy, precision, recall and macro f1-score
# calculates the loose scores -- if the correct language is in the list of predicted languages, the prediction is considered correct
def evaluate(args, identifier: AbstractLanguageIdentifier):
    print(f"Evaluation of {args.method} started")
    print(f"Loading dataset from {args.dataset}...")
    samples = [json.loads(line) for line in open(args.dataset)]

    supported_languages = identifier.languages
    language_to_index = {language: i for i, language in enumerate(supported_languages)}

    # loose metrics
    loose_accuracy = torchmetrics.Accuracy("binary")
    loose_per_language_f1 = {language: torchmetrics.F1Score("binary") for language in supported_languages}
    loose_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}

    # strict metrics
    strict_accuracy = torchmetrics.Accuracy("binary")
    overlap_f1 = 0.0
    strict_per_language_f1 = {language: torchmetrics.F1Score("binary", ) for language in supported_languages}
    strict_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}

    print("Running inference...")
    start_time = time.time()

    for sample in tqdm(samples):
        text = sample["text"]
        sample["gold_languages"] = set(sample["languages"])
        sample["predicted_languages"] = set(identifier.identify(text))

    end_time = time.time()

    print("Calculating metrics...")
    
    for sample in samples:
        gold_languages = sample["gold_languages"]
        predicted_languages = sample["predicted_languages"]

        # loose metrics
        if predicted_languages.issubset(gold_languages):
            loose_accuracy.update(torch.ones(1), torch.ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(torch.ones(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.ones(1), torch.ones(1))
                elif language not in predicted_languages and language not in gold_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))
        else:
            loose_accuracy.update(torch.zeros(1), torch.ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(torch.ones(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.ones(1), torch.zeros(1))
                elif language in predicted_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
                elif language in gold_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
                else:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))

        # strict metrics
        if predicted_languages == gold_languages:
            strict_accuracy.update(torch.ones(1), torch.ones(1))
        else:
            strict_accuracy.update(torch.zeros(1), torch.ones(1))
        
        common_languages = len(predicted_languages.intersection(gold_languages))
        overlap_precision = common_languages / len(predicted_languages)
        overlap_recall = common_languages / len(gold_languages)
        if overlap_precision + overlap_recall > 0:
            overlap_f1 += 2 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall)

        for language in supported_languages:
            if language in predicted_languages and language in gold_languages:
                strict_per_language_f1[language].update(torch.ones(1), torch.ones(1))
                strict_per_language_mcc[language].update(torch.ones(1), torch.ones(1))
            elif language in predicted_languages:
                strict_per_language_f1[language].update(torch.ones(1), torch.zeros(1))
                strict_per_language_mcc[language].update(torch.ones(1), torch.zeros(1))
            elif language in gold_languages:
                strict_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                strict_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
            else:
                strict_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                strict_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))


    # pretty print the confusion matrix
    print(f"\n# Results for {args.method}:\n")

    print("## Loose metrics")
    print(f"\tLoose accuracy: {loose_accuracy.compute().item():.2%}")
    print(f"\tLoose macro F1: {sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tLoose macro MCC: {sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {loose_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {loose_per_language_mcc[language].compute().item():.2%}")

    print("\n\n## Strict metrics")
    print(f"\tStrict accuracy: {strict_accuracy.compute().item():.2%}")
    print(f"\tOverlap F1: {overlap_f1 / len(samples):.2%}")
    print(f"\tStrict macro F1: {sum([strict_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tStrict macro MCC: {sum([strict_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {strict_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {strict_per_language_mcc[language].compute().item():.2%}")

    print("\n\n## CPU inference time")
    print(f"\tTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"\tms / sentence: {(end_time - start_time) / len(samples) * 1000:.2f} ms")
    print()
    
    misclassifications_count = {lang: {} for lang in supported_languages}
    

    output_file = "evaluation_results.txt" 
    with open(output_file, "w") as f:
    
        for sample in tqdm(samples):
            text = sample["text"]
            gold_languages = set(sample["languages"])
            predicted_languages = set(identifier.identify(text))
    
            if not predicted_languages.issubset(gold_languages):
                f.write(f"Sentence: {text}\n")
                f.write(f"Predicted languages: {predicted_languages}, Target languages: {gold_languages}\n\n")
    
                for gold_lang in gold_languages:
                    if gold_lang not in predicted_languages:
                        for pred_lang in predicted_languages:
                            misclassifications_count[gold_lang][pred_lang] = misclassifications_count[gold_lang].get(pred_lang, 0) + 1
    
    with open(output_file, "a") as f:
        f.write("\nMisclassifications count:\n")
        for target_lang, miscounts in misclassifications_count.items():
            f.write(f"Target: {target_lang}\n")
            for pred_lang, count in miscounts.items():
                f.write(f"\t{pred_lang}: {count}\n")
            f.write("\n\n")
        f.write(f"\n# Results for {args.method}:\n")

        f.write("## Loose metrics\n")
        f.write(f"\tLoose accuracy: {loose_accuracy.compute().item():.2%}\n")
        f.write(f"\tLoose macro F1: {sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}\n")
        f.write(f"\tLoose macro MCC: {sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}\n\n")

        f.write("### Per-language metrics\n")
        for language in supported_languages:
            f.write(f"\t{language}:\n")
            f.write(f"\t\tF1: {loose_per_language_f1[language].compute().item():.2%}\n")
            f.write(f"\t\tMCC: {loose_per_language_mcc[language].compute().item():.2%}\n\n")

        f.write("\n\n## Strict metrics\n")
        f.write(f"\tStrict accuracy: {strict_accuracy.compute().item():.2%}\n")
        f.write(f"\tOverlap F1: {overlap_f1 / len(samples):.2%}\n")
        f.write(f"\tStrict macro F1: {sum([strict_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}\n")
        f.write(f"\tStrict macro MCC: {sum([strict_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}\n\n")

        f.write("### Per-language metrics\n")
        for language in supported_languages:
            f.write(f"\t{language}:\n")
            f.write(f"\t\tF1: {strict_per_language_f1[language].compute().item():.2%}\n")
            f.write(f"\t\tMCC: {strict_per_language_mcc[language].compute().item():.2%}\n\n")

        f.write("\n\n## CPU inference time\n")
        f.write(f"\tTotal runtime: {end_time - start_time:.2f} seconds\n")
        f.write(f"\tms / sentence: {(end_time - start_time) / len(samples) * 1000:.2f} ms\n")


def main():
    random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

    args = parse_args()
    if args.method == "random":
        identifier = RandomLanguageIdentifier(args)
    elif args.method == "cld3":
        identifier = CLD3LanguageIdentifier(args)
    elif args.method == "nllb":
        identifier = NllbLanguageIdentifier(args)
    elif args.method == "fasttext":
        identifier = FasttextLanguageIdentifier(args)
    elif args.method == "openlid":
        identifier = OpenlidLanguageIdentifier(args)
    elif args.method == "langid":
        identifier = LangidLanguageIdentifier(args)
    elif args.method == "lingua":
        identifier = LinguaLanguageIdentifier(args)
    elif args.method == "langdetect":
        identifier = LangdetectLanguageIdentifier(args)
    elif args.method == "mbert":
        identifier = MBertLanguageIdentifier(args)
    elif args.method == "xlmr":
        identifier = RobertaLanguageIdentifier(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    evaluate(args, identifier)


if __name__ == "__main__":
    main()