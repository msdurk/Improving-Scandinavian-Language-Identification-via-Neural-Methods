import torch

class CustomModel(torch.nn.Module):
    def __init__(self, fasttext_model, mbert_model):
        super(CustomModel, self).__init__()
        self.fasttext_model = fasttext_model
        self.mbert_model = mbert_model

    def forward(self, text):
        language = self.fasttext_model.predict(text)[0][0]

        # Define default output for unknown languages
        default_output = {'logits': torch.zeros(1, num_labels), 'hidden_states': None, 'attentions': None}

        if language in ['nb', 'nn', 'sv', 'da', 'other']:
            output = self.mbert_model(text)
        else:
            print(f"Unknown language: {language}")
            output = default_output

        return output
