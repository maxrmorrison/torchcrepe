#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

import torchcrepe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio',
                        type=Path,
                        help='Audio file for real-audio benchmarks')
    parser.add_argument('--repeat-audio', type=int, default=1,
                        help='Repeat audio N times to create a longer benchmark clip')
    parser.add_argument('--device', default='cpu',
                        help='Inference device, e.g. cpu or mps')
    parser.add_argument('--model', default='full', choices=('tiny', 'full'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--frames', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--json', action='store_true')
    return parser.parse_args()


def synchronize(device):
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()


def load_audio(path):
    sample_rate, audio = wavfile.read(path)
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(np.iinfo(audio.dtype).max, 1)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32, copy=False)
    return torch.from_numpy(audio), int(sample_rate)


def time_call(fn, *, warmup, iters, device):
    for _ in range(warmup):
        fn()
        synchronize(device)
    durations_ms = []
    for _ in range(iters):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    durations_ms.sort()
    return durations_ms[len(durations_ms) // 2]


def benchmark_decoder_core(args):
    results = []
    generator = torch.Generator().manual_seed(0)
    for frames in args.frames:
        probabilities = torch.rand((1, torchcrepe.PITCH_BINS, frames),
                                   generator=generator)
        legacy_bins, _ = torchcrepe.decode.viterbi_legacy(probabilities, dither=False)
        fast_bins, _ = torchcrepe.decode.viterbi_banded_fast(probabilities, dither=False)
        results.append({
            'frames': frames,
            'parity_ok': bool(torch.equal(legacy_bins, fast_bins)),
            'viterbi_legacy_ms': time_call(
                lambda: torchcrepe.decode.viterbi_legacy(probabilities, dither=False),
                warmup=args.warmup,
                iters=args.iters,
                device='cpu',
            ),
            'viterbi_banded_fast_ms': time_call(
                lambda: torchcrepe.decode.viterbi_banded_fast(probabilities, dither=False),
                warmup=args.warmup,
                iters=args.iters,
                device='cpu',
            ),
        })
    return results


def benchmark_real_audio(args):
    if args.audio is None:
        raise ValueError('--audio is required for real-audio benchmarks')
    audio_path = args.audio
    audio, sample_rate = load_audio(str(audio_path))
    if args.repeat_audio > 1:
        audio = audio.repeat(1, args.repeat_audio)

    infer_device = args.device

    def predict(decoder_name):
        return torchcrepe.predict(
            audio,
            sample_rate,
            model=args.model,
            decoder=torchcrepe.decode.get_decoder(decoder_name),
            batch_size=args.batch_size,
            device=infer_device,
        )

    preprocessed = list(torchcrepe.preprocess(audio,
                                              sample_rate,
                                              None,
                                              args.batch_size,
                                              infer_device,
                                              True))
    probabilities = []
    with torch.no_grad():
        for frames in preprocessed:
            batch = torchcrepe.infer(frames, args.model, infer_device, embed=False)
            batch = batch.reshape(audio.size(0), -1, torchcrepe.PITCH_BINS).transpose(1, 2)
            probabilities.append(batch.detach().to('cpu'))
    probabilities = torch.cat(probabilities, dim=2)

    legacy_bins, _ = torchcrepe.decode.viterbi_legacy(probabilities, dither=False)
    fast_bins, _ = torchcrepe.decode.viterbi_banded_fast(probabilities, dither=False)

    results = {
        'audio': str(audio_path),
        'repeat_audio': args.repeat_audio,
        'frames': int(probabilities.shape[-1]),
        'decoder_parity_ok': bool(torch.equal(legacy_bins, fast_bins)),
        'decode_cached_viterbi_legacy_ms': time_call(
            lambda: torchcrepe.decode.viterbi_legacy(probabilities, dither=False),
            warmup=args.warmup,
            iters=args.iters,
            device='cpu',
        ),
        'decode_cached_viterbi_banded_fast_ms': time_call(
            lambda: torchcrepe.decode.viterbi_banded_fast(probabilities, dither=False),
            warmup=args.warmup,
            iters=args.iters,
            device='cpu',
        ),
        'predict_viterbi_legacy_ms': time_call(
            lambda: predict('viterbi_legacy'),
            warmup=args.warmup,
            iters=args.iters,
            device=infer_device,
        ),
        'predict_viterbi_banded_fast_ms': time_call(
            lambda: predict('viterbi_banded_fast'),
            warmup=args.warmup,
            iters=args.iters,
            device=infer_device,
        ),
        'predict_viterbi_ms': time_call(
            lambda: predict('viterbi'),
            warmup=args.warmup,
            iters=args.iters,
            device=infer_device,
        ),
    }
    return results


def render_markdown(core_results, real_audio_result, args):
    lines = [
        '## Torchcrepe Decoder Benchmark',
        '',
        f'**Device:** `{args.device}`',
        f'**Model:** `{args.model}`',
        f'**Batch size:** `{args.batch_size}`',
        '',
        '---',
        '',
        '## Synthetic Decoder Core',
        '',
        '| Frames | Legacy | Fast | Speedup | Parity |',
        '|------:|------:|------:|------:|:------:|',
    ]
    for result in core_results:
        speedup = result['viterbi_legacy_ms'] / result['viterbi_banded_fast_ms']
        lines.append(
            f"| {result['frames']} | **{result['viterbi_legacy_ms']:.3f} ms** | "
            f"**{result['viterbi_banded_fast_ms']:.3f} ms** | "
            f"`{speedup:.2f}x` | {'✅' if result['parity_ok'] else '❌'} |"
        )

    lines.extend([
        '',
        '---',
        '',
        '## Real Audio Cached Decoder',
        '',
        '| Decoder | Time |',
        '|:--|--:|',
        f"| `viterbi_legacy` | **{real_audio_result['decode_cached_viterbi_legacy_ms']:.3f} ms** |",
        f"| `viterbi_banded_fast` | **{real_audio_result['decode_cached_viterbi_banded_fast_ms']:.3f} ms** |",
        '',
        '---',
        '',
        '## Real Audio Predict Path',
        '',
        f"**Audio:** `{real_audio_result['audio']}`",
        f"**Frames:** `{real_audio_result['frames']}`",
        f"**Repeated:** `{real_audio_result['repeat_audio']}`",
        '',
        '| Decoder | Time |',
        '|:--|--:|',
        f"| `viterbi_legacy` | **{real_audio_result['predict_viterbi_legacy_ms']:.3f} ms** |",
        f"| `viterbi_banded_fast` | **{real_audio_result['predict_viterbi_banded_fast_ms']:.3f} ms** |",
        f"| `viterbi` | **{real_audio_result['predict_viterbi_ms']:.3f} ms** |",
        '',
        f"> {'✅' if real_audio_result['decoder_parity_ok'] else '❌'} "
        f"Decoder parity on cached model outputs: `{real_audio_result['decoder_parity_ok']}`",
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    core_results = benchmark_decoder_core(args)
    real_audio_result = benchmark_real_audio(args)
    payload = {
        'core': core_results,
        'real_audio': real_audio_result,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(render_markdown(core_results, real_audio_result, args))


if __name__ == '__main__':
    main()
