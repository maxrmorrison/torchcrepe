# torchcrepe
Pytorch implementation of the CREPE pitch tracker. The original Tensorflow
implementation can be found [here](https://github.com/marl/crepe/). The
provided model weights were obtained by converting the "tiny" and "full" models
using [MMdnn](https://github.com/microsoft/MMdnn), an open-source model
management framework.


### Installation
Perform the system-dependent PyTorch install using the instructions found
[here](https://pytorch.org/).

`pip install torchcrepe`


### Usage

##### Computing pitch and harmonicity from audio


```
import torchcrepe


# Load audio
audio, sr = torchcrepe.load.audio( ... )

# Place the audio on the device you want CREPE to run on
audio = audio.to( ... )

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Select a model capacity--one of "tiny" or "full"
model = 'tiny'

# Compute pitch and harmonicity
pitch = torchcrepe.predict(audio, sr, hop_length, fmin, fmax, model)
```

A harmonicity metric similar to the Crepe confidence score can also be
extracted by passing `return_harmonicity=True` to `torchcrepe.predict`.

By default, `torchcrepe` uses Viterbi decoding on the softmax of the network
output. This is different than the original implementation, which uses a
weighted average near the argmax of binary cross-entropy probabilities.
The argmax operation can cause double/half frequency errors. These can be
removed by penalizing large pitch jumps via Viterbi decoding. The `decode`
submodule provides some options for decoding.

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
can be noisy and the pitch has quantization artifacts. `torchcrepe` provides
submodules `filter` and `threshold` for this purpose. The filter and threshold
parameters should be tuned to your data. For clean speech, a 10-20 millisecond
window with a threshold of 0.21 has worked.

```
# We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
win_length = 3

# Median filter noisy confidence value
harmonicity = torchcrepe.filter.median(harmonicity, win_length)

# Remove inharmonic regions
pitch = torchcrepe.threshold.At(.21)(pitch, harmonicity)

# Optionally smooth pitch to remove quantization artifacts
pitch = torchcrepe.filter.mean(pitch, win_length)
```

For more fine-grained control over pitch thresholding, see
`torchcrepe.threshold.Hysteresis`. This is especially useful for removing
spurious voiced regions caused by noise in the harmonicity values, but
has more parameters and may require more manual tuning to your data.


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

##### Computing from files

`torchcrepe` defines the following functions convenient for predicting
directly from audio files on disk. Each of these functions also takes
a `device` argument that can be used for device placement (e.g.,
`device='gpu:0'`).

```
torchcrepe.predict_from_file(audio_file, ...)
torchcrepe.predict_from_file_to_file(
    audio_file, output_pitch_file, output_harmonicity_file, ...)
torchcrepe.predict_from_files_to_files(
    audio_files, output_pitch_files, output_harmonicity_files, ...)

torchcrepe.embed_from_file(audio_file, ...)
torchcrepe.embed_from_file_to_file(audio_file, output_file, ...)
torchcrepe.embed_from_files_to_files(audio_files, output_files, ...)
```

##### Command-line interface

```
usage: python -m torchcrepe
    [-h]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--hop_length HOP_LENGTH]
    [--output_harmonicity_files OUTPUT_HARMONICITY_FILES [OUTPUT_HARMONICITY_FILES ...]]
    [--embed]
    [--fmin FMIN]
    [--fmax FMAX]
    [--model MODEL]
    [--decoder DECODER]
    [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio file to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The file to save pitch or embedding
  --hop_length HOP_LENGTH
                        The hop length of the analysis window
  --output_harmonicity_files OUTPUT_HARMONICITY_FILES [OUTPUT_HARMONICITY_FILES ...]
                        The file to save harmonicity
  --embed               Performs embedding instead of pitch prediction
  --fmin FMIN           The minimum frequency allowed
  --fmax FMAX           The maximum frequency allowed
  --model MODEL         The model capacity. One of "tiny" or "full"
  --decoder DECODER     The decoder to use. One of "argmax", "viterbi", or
                        "weighted_argmax"
  --gpu GPU             The gpu to perform inference on
```


### Tests

The module tests can be run as follows.

```
pip install pytest
pytest
```
