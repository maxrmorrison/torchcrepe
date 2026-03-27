from torchcrepe import __main__
import torchcrepe


def test_parse_args_accepts_decoder_variants():
    args = __main__.parse_args([
        '--audio_files', 'in.wav',
        '--output_files', 'out.pt',
        '--decoder', 'viterbi_banded_fast',
    ])
    assert args.decoder == 'viterbi_banded_fast'


def test_get_decoder_returns_expected_function():
    assert (
        torchcrepe.decode.get_decoder('viterbi_legacy')
        is torchcrepe.decode.viterbi_legacy
    )
