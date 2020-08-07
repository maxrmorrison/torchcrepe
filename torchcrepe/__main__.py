import argparse
import os

import torchcrepe


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('audio_file', help='The audio file to process')
    parser.add_argument(
        'output_file',
        help='The file to save pitch or embedding')
    parser.add_argument(
        'hop_length',
        type=int,
        help='The hop length of the analysis window')

    # Optionally save harmonicity
    parser.add_argument(
        '--output_harmonicity_file',
        help='The file to save harmonicity')

    # Optionally create embedding instead of pitch contour
    parser.add_argument(
        '--embed',
        action='store_true',
        help='Performs embedding instead of pitch prediction')

    # Optional arguments
    parser.add_argument(
        '--fmin',
        default=50.,
        type=float,
        help='The minimum frequency allowed')
    parser.add_argument(
        '--fmax',
        default=torchcrepe.MAX_FMAX,
        type=float,
        help='The maximum frequency allowed')
    parser.add_argument(
        '--model',
        default='full',
        help='The model capacity. One of "tiny" or "full"')
    parser.add_argument(
        '--decoder',
        default='viterbi',
        help='The decoder to use. One of "argmax", "viterbi", or ' +
             '"weighted_argmax"')
    parser.add_argument(
        '--device',
        type=int,
        help='The device to perform inference on')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_pitch_file), exist_ok=True)
    if args.output_harmonicity_file is not None:
        os.makedirs(os.path.dirname(args.output_harmonicity_file),
                    exist_ok=True)

    # Get inference device
    device = 'cpu' if args.device is None else f'gpu:{args.gpu}'

    # Get decoder
    if args.decoder == 'argmax':
        decoder = torchcrepe.decode.argmax
    elif args.decoder == 'weighted_argmax':
        decoder = torchcrepe.decode.weighted_argmax
    elif args.decoder == 'viterbi':
        decoder = torchcrepe.decode.viterbi

    # Infer pitch or embedding and save to disk
    if args.embed:
        torchcrepe.embed_from_file_to_file(args.audio_file,
                                           args.hop_length,
                                           args.output_pitch_file,
                                           args.model,
                                           device)
    else:
        torchcrepe.predict_from_file_to_file(args.audio_file,
                                             args.hop_length,
                                             args.output_file,
                                             args.output_harmonicity_file,
                                             args.fmin,
                                             args.fmax,
                                             args.model,
                                             decoder,
                                             device)


# Run module entry point
main()
