import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample(model, start_text, config):
    length = config.n_ctx // 2

    current_text = start_text
    past = None
    output = [start_text]
    with torch.no_grad():
        for _ in trange(length):
            logits, past = model(current_text, past=past)
            # Input parameters:
            #     current_text: the encoded text token at t-1
            #     past: the calculated hidden state of the previous text or None if no previous text given
            # Return:
            #     logits: a tensor of shape (batch_size, sequence_length, size_of_vocabulary)
            #     past: the calculated hidden state of the previous + current text

            current_logits = logits[:, -1, :]
            logits = top_k_logits(current_logits, k=config.top_k)

            ##TODO:
            ## 1) sample using the given `logits` tensor;
            ## 2) append the sample to the list `output`;
            ## 3) update `current_text` so that sampling can continue.
            ##    Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
            ##                   Pytorch multinomial sampling: https://pytorch.org/docs/stable/generated/torch.multinomial.html
            ## Hint: Implementation should only takes 3~5 lines of code.
            ##       The text generated should look like a technical paper.

            # convert logits to probabilities
            logits_prob = F.softmax(logits, dim=1)
            # sample a logit
            sampled_logit_idx = torch.multinomial(logits_prob, 1)
            output[0] = torch.cat((output[0], sampled_logit_idx), dim=1)
            # current_text = torch.cat((current_text, sampled_logit_idx), dim=1)
            current_text = sampled_logit_idx

        output = torch.cat(output, dim=1)
        return output
