import torch
from transformers import AutoModel

class SpeechSSLModel : 
    def init(self, params, frozen=True, output_hidden_states=True):
        self.model = AutoModel.from_pretrained(params["hub"],output_hidden_states=True)
        if frozen : 
            self.model.eval()
        self.num_layers=  params["num_layers"]

    def featurize(self, wavs, wavs_lens, weights, layernorm=False):
        feats= self.model(wavs)
        hidden_states= torch.stack(feats.hidden_states, dim=0).detach()
        #First dimension should be equal to the number of layers in the hparams
        assert self.num_layers == hidden_states.shape[0], "Num layers not equal to num hidden states" 
        norm_weights = torch.nn.functional.softmax(weights, dim=-1)
        if layernorm : 
            hidden_states = [F.layer_norm(t, (t.shape[-1],)) for t in hidden_states]
        weighted_feats = hidden_states[0] * norm_weights[0]
        for i in range(1, len(hidden_states)): 
            weighted_feats += hidden_states[i] * norm_weights[i]

        return weighted_feats







