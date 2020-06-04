# torchcrepe
Pytorch implementation of the CREPE pitch tracker. The original Tensorflow
implementation can be found [here](https://github.com/marl/crepe/). The
provided model weights were obtained by converting the "tiny" model using
[MMdnn](https://github.com/microsoft/MMdnn), an open-source model management
framework.


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

# Here we'll use a 5 ms hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Compute pitch and harmonicity using viterbi decoding from CREPE logits
pitch, harmonicity = torchcrepe.predict(audio,
                                        sr,
                                        hop_length,
                                        fmin,
                                        fmax,
                                        viterbi=True)
```

When harmonicity is low, the pitch is less reliable. For some problems, it
makes sense to mask these less reliable pitch values. However, the pitch and
harmonicity can be noisy. `torchcrepe` provides a `filters` module for this.
The window sizes of the filters and harmonicity threshold should be tuned to
your data, if used at all. I have found 10-20 millisecond windows works well
for speech.


```
# Median filter noisy confidence value
harmonicity = torchcrepe.filters.median(harmonicity, window_size)

# Remove inharmonic regions
pitch = torchcrepe.filters.threshold(pitch, harmonicity, threshold)

# Optionally smooth pitch to remove quantization artifacts
pitch = torchcrepe.filters.mean(pitch, window_size)
```


##### Computing the CREPE model output activations

```
probabilities = torchcrepe.infer(torchcrepe.preprocess(audio, sr, hop_length))
```


##### Computing the CREPE embedding space

As in Differentiable Digital Signal Processing, this uses the output of the
fifth max-pooling layer as a pretrained pitch embedding

```
embeddings = torchcrepe.embed(audio, sr, hop_length)
```
