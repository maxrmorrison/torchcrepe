# torchcrepe
Pytorch implementation of the CREPE pitch tracker. The original Tensorflow implementation can be found [here](https://github.com/marl/crepe/). The provided model weights were obtained by converting the "tiny" model using [MMdnn](https://github.com/microsoft/MMdnn), an open-source model management framework.


### Installation

Clone this repo and run `pip install -e .` in the `torchcrepe` directory.


### Usage

##### Computing pitch and harmonicity from audio

```
import torchaudio
import torchcrepe

# Load audio
audio, sr = torchaudio.load( ... )

# Place the audio on the device you want CREPE to run on
audio = audio.to( ... )

# Here we'll use a 10 ms hop length
hop_length = int(sr / 100.)

# Compute pitch and harmonicity using CREPE
pitch, harmonicity = torchcrepe.predict(audio, sr, hop_length)
```


##### Computing the CREPE model output activations

```
probabilities = torchcrepe.infer(torchcrepe.preprocess(audio, sr, hop_length))
```


##### Computing the CREPE embedding space

As in Differentiable Digital Signal Processing, this uses the output of the fifth max-pooling layer as a pretrained pitch embedding

```
embeddings = torchcrepe.embed(audio, sr, hop_length)
```


### Tasks

- [ ] Filtering (mean and median)
- [ ] Viterbi decoding
- [x] DDSP embedding
- [x] Batch processing
