# torchcrepe
Pytorch implementation of the CREPE pitch tracker. The original Tensorflow implementation can be found [here](https://github.com/marl/crepe/). The provided model weights were obtained by converting the "tiny" model using [MMdnn](https://github.com/microsoft/MMdnn), an open-source model management framework.


### Installation

Clone this repo and run `pip install -e .` in the `torchcrepe` directory.


### Usage

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
pitch, harmonicity = torchcrepe.predict(audio.squeeze(), sr, hop_length)
```

The CREPE model is loaded once and automatically placed on the same device as  the model input (i.e., if `audio.device == 'cuda:#'` then inference will occur  on device `'cuda:#'`).


### Tasks

- [ ] Viterbi decoding
- [x] DDSP embedding
- [x] Batch processing
