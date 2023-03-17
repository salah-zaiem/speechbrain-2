Speaker verification is done in two phases, first you need to train the speaker embedding representations doing :

```
python wav2vectrain.py hparams/hubert_large_train_xvectors.yaml 

```

then you have to test these representations on unseen speakers doing : 

```
python speaker_verification_cosine.py hparams/verification_hubert_xvector.yaml

```

In the second hparams file you need to  link the folder containing the results of the first training in the entries : 

* pretrain path 

* n mels should correspond to the dimension of the output of the SSL model ( 768, 1024 ... ) 

 
