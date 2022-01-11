import torch
import torch.nn as nn

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## Hint: Implementation should only takes 3~7 lines of code.
        logits_log_softmax_list = []
        for i in range(text.shape[1]):
            
            logits, _ = model(text[0,i].expand(1, 1))
            # convert logits to log probabilities
            logits_log_softmax = nn.LogSoftmax(dim=-1)(logits)
            logits_log_softmax_list.append(logits_log_softmax[0,0].cpu().numpy())

        # compute log-likelihood between prediction and gt
        ll = -nn.NLLLoss(reduction='sum')(torch.tensor(logits_log_softmax_list[:-1]), text[0,1:].cpu()).cpu().item()
        return ll
