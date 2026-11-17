import pytest
import torch

from survey_fusion.training.spectral_calibration import SpectralCalibrationConfig, SpectralNuisanceCalibrator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("sizes", [[1] * 62, [2] * 32, [1, 2, 3, 1, 5, 1, 8]])
@pytest.mark.parametrize("with_prior", [False, True])
@pytest.mark.parametrize('singleton_mode',['0','backward','both'])
@pytest.mark.parametrize('tf32',[False,True])
@pytest.mark.parametrize('packed', ['0', '1'])
def test_batched_nuisance_preserves_prediction_and_parameter_gradient(sizes, with_prior,singleton_mode,tf32,packed,monkeypatch):
    monkeypatch.setenv('SPLAT_CALIB_SINGLETONS',singleton_mode)
    monkeypatch.setenv('SPLAT_CALIB_PACKED', packed)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',tf32)
    generator = torch.Generator(device="cuda").manual_seed(3251)
    calibrator = SpectralNuisanceCalibrator.__new__(SpectralNuisanceCalibrator)
    torch.nn.Module.__init__(calibrator)
    calibrator.cfg = SpectralCalibrationConfig()
    calibrator.band_names = ["u", "g", "r", "i", "z"]
    calibrator.log_scales = torch.nn.Parameter(torch.randn(5, generator=generator, device="cuda") * 0.1)
    calibrator._mix_weights = {}
    calibrator._covered = {}
    calibrator._batch_mix_table = None
    calibrator._batch_covered_table = None
    calibrator._batch_exposure_bases = None
    for e in range(len(sizes)):
        weights = torch.rand((101 + e, 5), generator=generator, device="cuda")
        weights[::2,1:]=0
        calibrator._mix_weights[e] = weights / weights.sum(1, keepdim=True)
        calibrator._covered[e] = (torch.rand(101 + e, generator=generator, device="cuda") > 0.1).float()
    exposure_ids = list(range(len(sizes)))
    offsets = [0]
    indices = []
    for e, count in enumerate(sizes):
        offsets.append(offsets[-1] + count)
        indices.append(torch.randint(101 + e, (count,), generator=generator, device="cuda"))
    local = torch.cat(indices)
    source = torch.randn(sum(sizes), generator=generator, device="cuda")
    grad = torch.randn(sum(sizes), generator=generator, device="cuda")
    grad[::13] = -0.0

    def evaluate(batched):
        prediction = source.detach().requires_grad_()
        scales = (
            calibrator.scale_for_batch(exposure_ids, offsets, local)
            if batched
            else torch.cat(
                [calibrator.scale_for_indices(e, local[offsets[e] : offsets[e + 1]]) for e in exposure_ids]
            )
        )
        result = prediction * scales
        loss = (result * grad).sum()
        if with_prior:
            loss = loss + calibrator.prior_penalty()
        return result, *torch.autograd.grad(loss, (prediction, calibrator.log_scales))

    reference = evaluate(False)
    actual = evaluate(True)
    for a, b in zip(actual, reference):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    # The cache is reused after parameter updates; it contains only immutable data.
    with torch.no_grad():
        calibrator.log_scales.add_(0.013)
    reference = evaluate(False)
    actual = evaluate(True)
    for a, b in zip(actual, reference):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
