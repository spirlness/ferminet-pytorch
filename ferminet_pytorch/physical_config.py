"""Physical configuration shared by the two wavefunction APIs."""

from typing import Optional

from .jastrow import KatoCuspJastrow
from .spin import SpinProjector


def cusp_defaults(envelope_type, smooth_cuspless, has_jastrow, jastrow=None):
    nuclear_cusp = has_jastrow and getattr(jastrow, "cusp_en", True)
    # The validated cusp-compatible default uses the soft-exponential tail;
    # Gaussian remains available as an explicit compatibility baseline.
    env = envelope_type or ("soft_exponential" if nuclear_cusp else "exponential")
    smooth = True if smooth_cuspless is None else smooth_cuspless
    if env not in ("gaussian", "exponential", "soft_exponential"):
        raise ValueError("Unknown envelope_type")
    if nuclear_cusp and env == "exponential":
        raise ValueError(
            "An electron-nuclear cusp Jastrow requires a cuspless gaussian or soft_exponential envelope"
        )
    if has_jastrow and not smooth:
        raise ValueError("A cusp Jastrow requires smooth_cuspless=True")
    return env, smooth


def resolve_spin(n_up, n_down, spin_adapted, projector=None, kwargs=None):
    if projector is not None or not spin_adapted:
        return projector
    config = dict(kwargs or {})
    config.setdefault("mode", "singlet" if n_up == n_down else "none")
    return SpinProjector(n_up, n_down, **config)


def pair_spin_for(n_up, n_down, projector) -> Optional[int]:
    if n_up + n_down == 2 and isinstance(projector, SpinProjector):
        return {"singlet": 0, "triplet": 1}.get(projector.mode)
    return None


def validate_pair_cusp(jastrow, pair_spin):
    if (
        pair_spin is not None
        and isinstance(jastrow, KatoCuspJastrow)
        and jastrow.cusp_ee
    ):
        expected = 0.25 if pair_spin == 1 else 0.5
        if jastrow.gamma_pairs[0].item() != expected:
            raise ValueError(
                f"Jastrow cusp is incompatible with projected spin; use two_electron_spin={pair_spin}"
            )
