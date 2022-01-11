import torch
import torch.nn as nn

def classification(model, text):
    """
    Classify whether the string `text` is randomly generated or not.
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: True if `text` is a random string. Otherwise return False
    """

    with torch.no_grad():
        ## TODO: Return True if `text` is a random string. Or else return False.
        ## Hint: Your answer should be VERY SHORT! Our implementation has only 2 lines,
        ##       and yours shouldn't be longer than 7 lines. You should look at the plots
        ##       you generated in Question 4 very carefully and make use of the log_likelihood() function.
        ##       There should be NO model training involved.

        logits_log_softmax_list = []
        for i in range(text.shape[1]):
            
            logits, _ = model(text[0,i].expand(1, 1))
            # convert logits to log probabilities
            logits_log_softmax = nn.LogSoftmax(dim=-1)(logits)
            logits_log_softmax_list.append(logits_log_softmax[0,0].cpu().numpy())

        # compute log-likelihood between prediction and gt
        ll = -nn.NLLLoss(reduction='mean')(torch.tensor(logits_log_softmax_list[:-1]), text[0,1:].cpu()).cpu().item()
        if ll < -8.0:
            return False
        else:
             return True