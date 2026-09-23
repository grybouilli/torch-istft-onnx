import pytest
import torch

from torch_istft_onnx.istft import ISTFT


class TestIstft:
    @pytest.fixture
    def n_fft(self) -> int:
        return 256

    @pytest.fixture
    def hop_length(self) -> int:
        return 64

    @pytest.fixture
    def istft(self, n_fft: int, hop_length: int) -> torch.nn.Module:
        istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), normalized=True
        )
        return istft

    def test_forward_pass(self, n_fft: int, istft: ISTFT):
        # freqs x frames x m
        input = torch.randn(n_fft + 2, 20, 2)  # last dim is 2 for real and imaginary parts
        output = istft(input)
        assert len(output.shape) == 2

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_torch_stft_inversion(self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT):
        input = torch.randn((batch_size, 4096))

        spectro = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            return_complex=False,  # not return complex tensor
        )

        output = istft(spectro)

        assert len(output.shape) == 2
        assert output.shape[1] == (input.shape[1] // hop_length) * hop_length

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_consistent_values_with_torch_istft(self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT):
        input = torch.randn((batch_size, 4096))
        spectro = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            center=True,
            return_complex=False,  # not return complex tensor
        )

        # convert the spectro to complex tensors
        spectro_complex = torch.complex(spectro[..., 0], spectro[..., 1])
        torch_output = torch.istft(
            spectro_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
        )

        output = istft(spectro)

        assert torch_output.shape == output.shape
        # issue with high error in the last `hop_length` samples
        assert (torch_output[..., :-hop_length] - output[..., :-hop_length]).abs().mean() < 1e-5

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_consistent_values_with_torch_istft_complex_input(
        self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT
    ):
        input = torch.randn((batch_size, 4096))
        spectro_complex = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            center=True,
            return_complex=True,  # returns a complex tensor directly
        )

        torch_output = torch.istft(
            spectro_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
        )

        # ISTFT module expects a real tensor of shape (..., freq, time, 2)
        # so we convert the complex tensor back to real/imag channels
        spectro = torch.stack([spectro_complex.real, spectro_complex.imag], dim=-1)
        output = istft(spectro)

        assert torch_output.shape == output.shape
        # issue with high error in the last `hop_length` samples
        assert (torch_output[..., :-hop_length] - output[..., :-hop_length]).abs().mean() < 1e-5

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_mismatch_in_last_hop_length_samples(self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT):
        """
        Regression test for the known mismatch between this ISTFT implementation
        and torch.istft in the last `hop_length` samples.

        The bulk of the signal should match closely (mean abs error < 1e-5),
        while the tail is expected to diverge. This test verifies both sides of
        that boundary to catch regressions in either direction:
        - a fix that accidentally breaks the good region, or
        - a fix that resolves the tail mismatch (at which point this test should
        be updated to tighten the tail tolerance).
        """
        input = torch.randn((batch_size, 4096))
        spectro_complex = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            center=True,
            return_complex=True,
        )

        torch_output = torch.istft(
            spectro_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
        )

        spectro = torch.stack([spectro_complex.real, spectro_complex.imag], dim=-1)
        output = istft(spectro)

        assert torch_output.shape == output.shape

        error = (torch_output - output).abs()

        # The main body of the signal should match torch.istft closely.
        for block in range(n_fft // hop_length):
            main_body_error = error[..., block * hop_length : (block + 1) * hop_length].mean()
            assert main_body_error < 1e-5, (
                f"Unexpected error in block {block} of main body of signal: {main_body_error:.2e} " f"(expected < 1e-5)"
            )

        # The last `hop_length` samples have been known to diverge from torch.istft.
        # Assert the mismatch is no longer present
        tail_error = error[..., -hop_length:].mean()
        assert tail_error < 1e-5, (
            f"Mismatch in the last {hop_length} samples vs torch.istft, " f"with error of {tail_error:.2e}."
        )
