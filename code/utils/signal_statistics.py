"""32 fixed amplitude/phase statistics from IQ only; real-valued and sampling-rate free."""
import torch


STATISTICS_DIM = 32


def multiply(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@torch.no_grad()
def signal_statistics(iq, lengths):
    with torch.autocast(device_type=iq.device.type, enabled=False):
        iq = iq.float()
        coordinate = torch.arange(iq.shape[1], device=iq.device)[None]
        mask = coordinate < lengths[:, None]
        count = lengths[:, None].float().clamp_min(1)
        centered = (iq - (iq * mask[..., None]).sum(1, keepdim=True) / count[..., None]) * mask[..., None]
        scale = (centered.square().sum((1, 2)) / count[:, 0]).sqrt().clamp_min(1e-8)
        real, imag = (centered / scale[:, None, None]).unbind(-1)
        power = real.square() + imag.square()
        amplitude = power.clamp_min(0).sqrt()

        def average(x):
            return (x * mask).sum(1) / count[:, 0]

        values = [average(amplitude)]
        values.extend(torch.log1p(average(power.pow(k))) for k in [2, 3, 4])
        values.extend(((power < threshold) & mask).sum(1) / count[:, 0] for threshold in [.25, .5, 1., 2., 4.])
        correlation1 = None
        for lag in [1, 2, 4, 8, 16, 32]:
            valid = coordinate[:, lag:] < lengths[:, None]
            rr = real[:, lag:] * real[:, :-lag] + imag[:, lag:] * imag[:, :-lag]
            ri = imag[:, lag:] * real[:, :-lag] - real[:, lag:] * imag[:, :-lag]
            denominator = (lengths - lag).float().clamp_min(1)
            cr, ci = (rr * valid).sum(1) / denominator, (ri * valid).sum(1) / denominator
            values.append((cr.square() + ci.square()).sqrt())
            if lag == 1:
                correlation1 = cr, ci
                transition_r, transition_i = rr, ri
                transition_valid = valid
        magnitude = (transition_r.square() + transition_i.square()).sqrt().clamp_min(1e-8)
        ur, ui = transition_r / magnitude, transition_i / magnitude
        for exponent in [1, 2, 4, 8]:
            pr, pi = ur, ui
            for _ in range(exponent.bit_length() - 1):
                pr, pi = multiply(pr, pi, pr, pi)
            denominator = (lengths - 1).float().clamp_min(1)
            mr, mi = (pr * transition_valid).sum(1) / denominator, (pi * transition_valid).sum(1) / denominator
            values.append((mr.square() + mi.square()).sqrt())

        def cumulants(r, i):
            r2, i2 = multiply(r, i, r, i)
            m2r, m2i = average(r2), average(i2)
            r4, i4 = multiply(r2, i2, r2, i2)
            square_r, square_i = multiply(m2r, m2i, m2r, m2i)
            c4r, c4i = average(r4) - 3 * square_r, average(i4) - 3 * square_i
            return ((m2r.square() + m2i.square()).sqrt(),
                    torch.log1p((c4r.square() + c4i.square()).sqrt()),
                    torch.tanh(average(power.square()) - m2r.square() - m2i.square() - 2 * average(power).square()))
        circularity, c40, c42 = cumulants(real, imag)
        values.extend([circularity, c40, c42])
        omega = torch.atan2(correlation1[1], correlation1[0])
        angle = coordinate.float() * omega[:, None]
        corrected_r = real * angle.cos() + imag * angle.sin()
        corrected_i = imag * angle.cos() - real * angle.sin()
        corrected_circularity, corrected_c40, _ = cumulants(corrected_r, corrected_i)
        values.extend([corrected_circularity, corrected_c40])
        magnitude = amplitude.clamp_min(1e-8)
        ur, ui = corrected_r / magnitude, corrected_i / magnitude
        for exponent in [1, 2, 4, 8]:
            pr, pi = ur, ui
            for _ in range(exponent.bit_length() - 1):
                pr, pi = multiply(pr, pi, pr, pi)
            values.append((average(pr).square() + average(pi).square()).sqrt())
        mean_amplitude = average(amplitude)
        amplitude_variance = average((amplitude - mean_amplitude[:, None]).square()).clamp_min(1e-8)
        centered_amplitude = amplitude - mean_amplitude[:, None]
        for lag in [1, 2, 4, 8]:
            valid = coordinate[:, lag:] < lengths[:, None]
            correlation = (centered_amplitude[:, lag:] * centered_amplitude[:, :-lag] * valid).sum(1)
            values.append((correlation / (lengths - lag).float().clamp_min(1) / amplitude_variance).clamp(-1, 1))
        result = torch.stack(values, -1)
        if result.shape[-1] != STATISTICS_DIM:
            raise AssertionError('Signal statistics dimension mismatch')
        return result
