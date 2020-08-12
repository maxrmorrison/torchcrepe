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
    parser.add_argument(
        '--audio_files',
        nargs='+',
        required=True,
        help='The audio file to process')
    parser.add_argument(
        '--output_files',
        nargs='+',
        required=True,
        help='The file to save pitch or embedding')
    parser.add_argument(
        '--hop_length',
        type=int,
        help='The hop length of the analysis window')

    # Optionally save harmonicity
    parser.add_argument(
        '--output_harmonicity_files',
        nargs='+',
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
        '--batch_size',
        type=int,
        help='The number of frames per batch')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to perform inference on')

    return parser.parse_args()


def make_parent_directory(file):
    """Create parent directory for file if it does not already exist"""
    parent = os.path.dirname(os.path.abspath(file))
    os.makedirs(parent, exist_ok=True)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Ensure output directory exist
    [make_parent_directory(file) for file in args.output_files]
    if args.output_harmonicity_files is not None:
        [make_parent_directory(file) for file in args.output_harmonicity_files]

    # Get inference device
    device = 'cpu' if args.gpu is None else f'cuda:{args.gpu}'

    # Get decoder
    if args.decoder == 'argmax':
        decoder = torchcrepe.decode.argmax
    elif args.decoder == 'weighted_argmax':
        decoder = torchcrepe.decode.weighted_argmax
    elif args.decoder == 'viterbi':
        decoder = torchcrepe.decode.viterbi

    # Infer pitch or embedding and save to disk
    if args.embed:
        torchcrepe.embed_from_files_to_files(args.audio_files,
                                             args.output_files,
                                             args.hop_length,
                                             args.model,
                                             args.batch_size,
                                             device)
    else:
        torchcrepe.predict_from_files_to_files(args.audio_files,
                                               args.output_files,
                                               args.output_harmonicity_files,
                                               args.hop_length,
                                               args.fmin,
                                               args.fmax,
                                               args.model,
                                               decoder,
                                               args.batch_size,
                                               device)


# Run module entry point
main()
