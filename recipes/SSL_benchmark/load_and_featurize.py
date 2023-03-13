import torch
from transformers import AutoModel


def load_model(params): 
    hf_model = AutoModel.from_pretrained(params["hub"],output_hidden_states=True)
    hf_model.eval()
    return hf_model

def featurize(loaded_model, weights, wavs, wavs_lens, params):
    feats= hf_model(wavs)
    hidden_states= torch.stack(feats.hidden_states, dim=0).detach()
    #First dimension should be equal to the number of layers in the hparams
    norm_weights = torch.nn.functional.softmax(layers_weights, dim=-1)
    if layernorm : 
        hidden_states = [F.layer_norm(t, (t.shape[-1],)) for t in hidden_states]
    weighted_feats = hidden_states[0] * norm_weights[0]
    for i in range(1, len(x)): 
        weighted_feats += hidden_states[i] * norm_weights[i]

    return weighted_feats



