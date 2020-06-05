# torchcrepe
Pytorch implementation of the CREPE pitch tracker. The original Tensorflow
implementation can be found [here](https://github.com/marl/crepe/). The
provided model weights were obtained by converting the "tiny" model using
[MMdnn](https://github.com/microsoft/MMdnn), an open-source model management
framework.


### Installation

Clone this repo and run `pip install .` in the `torchcrepe` directory.


### Usage

##### Computing pitch and harmonicity from audio


```
import torchaudio
import torchcrepe

# Load audio
audio, sr = torchaudio.load( ... )

# Place the audio on the device you want CREPE to run on
audio = audio.to( ... )

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Compute pitch and harmonicity
pitch, harmonicity = torchcrepe.predict(audio, sr, hop_length, fmin, fmax)
```

By default, `torchcrepe` uses Viterbi decoding on the softmax of the network
logits. This is different than the original implementation, which uses a
weighted average near the argmax of cross-entropy probabilities. The argmax
operation can cause double/half frequency errors that are removed by
penalizing large pitch jumps via Viterbi decoding. The `decode` submodule
provides some options for decoding.

```
# Decode using viterbi decoding (default)
torchcrepe.predict(..., decoder=torchcrepe.decode.viterbi)

# Decode using weighted argmax (as in the original implementation)
torchcrepe.predict(..., decoder=torchcrepe.decode.weighted_argmax)

# Decode using argmax
torchcrepe.predict(..., decoder=torchcrepe.decode.argmax)
```

When harmonicity is low, the pitch is less reliable. For some problems, it
makes sense to mask these less reliable pitch values. However, the harmonicity
can be noisy and the pitch has quantization artifacts. `torchcrepe` provides a
`filters` submodule for this. The window sizes of the filters and harmonicity 
threshold should be tuned to your data. For clean speech, a
10-20 millisecond window with a threshold of 0.23 has worked.

```
# We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
win_length = 3

# Median filter noisy confidence value
harmonicity = torchcrepe.filter.median(harmonicity, win_length)

# Remove inharmonic regions
pitch = torchcrepe.threshold(pitch, harmonicity, 0.23)

# Optionally smooth pitch to remove quantization artifacts
pitch = torchcrepe.filter.mean(pitch, win_length)
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
