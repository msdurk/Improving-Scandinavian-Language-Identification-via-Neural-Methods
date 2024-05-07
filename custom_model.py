import torch

class CustomModel(torch.nn.Module):
    def __init__(self, fasttext_model, mbert_model):
        super(CustomModel, self).__init__()
        self.fasttext_model = fasttext_model
        self.mbert_model = mbert_model

    def forward(self, input_ids, attention_mask=None, labels=None):
        try:
            if labels is not None:
                labels = torch.argmax(labels, dim=1)
            outputs = self.mbert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        except Exception as e:
            print(f"Error in model forwarding: {e}")
        return outputs

