#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

"""
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        new_feats= hf_model(wavs)

        # Loading the layers and weighting them 
        x= new_feats.hidden_states
        x= torch.stack(x, dim=0).detach()
        norm_weights = torch.nn.functional.softmax(self.layers_weights, dim=-1)
        layer_0 = x[0] * norm_weights[0]
        for i in range(1, len(x)): 
            layer_0 += x[i] * norm_weights[i]

        # Forward pass
        feats = layer_0
        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.model_optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.model_optimizer.step()
            self.wav2vec_optimizer.step()

        self.model_optimizer.zero_grad()
        self.wav2vec_optimizer.zero_grad()
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            torch.save(self.layers_weights, self.hparams.weights_path)
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )
    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        print(self.layers_weights)
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            [self.layers_weights]
        )
        self.model_optimizer = self.hparams.opt_class(self.modules.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("weights", self.layers_weights)




def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa
    """
    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": hparams["sentence_len"],
        },
    )
    """
    #Loading wav2vec2.0 
    if not hparams["pretrain"]:
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()
    from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2FeatureExtractor
    from transformers import AutoModel


    hf_model = AutoModel.from_pretrained(hparams["hub"], output_hidden_states=True)
    hf_model.eval()
   

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
   
    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    hf_model.to(speaker_brain.device)
    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    if not os.path.exists(hparams["weights_path"]): 
        #Zero initializiation like in S3PRL, not sure if it is the best choice, test torch.rand !   
        end_init = torch.cat([torch.zeros(hparams["num_layers"])])
    else : 
        end_init = torch.load(hparams["weights_path"])
        print("loaded weights")
    speaker_brain.layers_weights = torch.nn.Parameter(end_init, requires_grad=True)
    torch.save(speaker_brain.layers_weights, hparams["weights_path"])
    speaker_brain.layers_weights.to(speaker_brain.device)

    #for name, param in speaker_brain.modules.wav2vec2.named_parameters():
    #    print (name)
    #    print(param.data)
    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
