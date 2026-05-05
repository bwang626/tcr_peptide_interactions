"""
Tests for embeddings/autoencoder/model.py
Run from repo root: python -m pytest embeddings/autoencoder/test_autoencoder.py -v
"""

import numpy as np
import pytest
import torch
import pandas as pd

import sys, os, importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_here = os.path.dirname(os.path.abspath(__file__))
_one_hot_model = _load("one_hot.model", os.path.join(_here, "..", "one_hot", "model.py"))
_ae_model      = _load("autoencoder.model", os.path.join(_here, "model.py"))

encode_sequence  = _one_hot_model.encode_sequence
encode_sequences = _one_hot_model.encode_sequences
decode_one_hot   = _one_hot_model.decode_one_hot
VOCAB_SIZE       = _one_hot_model.VOCAB_SIZE
AA_TO_IDX        = _one_hot_model.AA_TO_IDX
SequenceAutoencoder  = _ae_model.SequenceAutoencoder
AutoencoderEmbedder  = _ae_model.AutoencoderEmbedder

# ─── Fixtures ────────────────────────────────

FAKE_TCR_SEQS = [
    "CASSLAPGATNEKLFF",
    "CASSPGTASYNEKLFF",
    "CASSLGQAYEQYF",
    "CASSQDRGTEAFF",
    "CASSGLAGGYNEQFF",
]

FAKE_PEPTIDE_SEQS = [
    "GILGFVFTL",
    "NLVPMVATV",
    "GLCTLVAML",
    "ELAGIGILTV",
    "SIINFEKL",
]


# ─── Encoding tests ──────────────────────────

def test_encode_sequence_shape():
    enc = encode_sequence("CASSLAPG", max_len=15)
    assert enc.shape == (15, VOCAB_SIZE)

def test_encode_sequence_padding():
    enc = encode_sequence("AAA", max_len=10)
    # Positions 3–9 should be all-zero except at pad index 0
    for pos in range(3, 10):
        assert enc[pos, AA_TO_IDX["-"]] == 1.0

def test_encode_sequence_truncation():
    long_seq = "A" * 50
    enc = encode_sequence(long_seq, max_len=20)
    assert enc.shape == (20, VOCAB_SIZE)

def test_encode_sequences_batch():
    batch = encode_sequences(FAKE_TCR_SEQS, max_len=30)
    assert batch.shape == (len(FAKE_TCR_SEQS), 30, VOCAB_SIZE)
    # Each position should sum to 1 (one-hot)
    assert np.allclose(batch.sum(axis=-1), 1.0)

def test_decode_roundtrip():
    seq = "CASSLAPG"
    enc = encode_sequence(seq, max_len=20)
    decoded = decode_one_hot(enc)
    assert decoded == seq


# ─── Model shape tests ────────────────────────

def test_forward_shapes():
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=32, max_len=20)
    x = torch.tensor(encode_sequences(FAKE_TCR_SEQS, max_len=20))
    logits, mu, logvar = model(x)
    assert logits.shape == (len(FAKE_TCR_SEQS), 20, VOCAB_SIZE)
    assert mu.shape == (len(FAKE_TCR_SEQS), 32)
    assert logvar.shape == (len(FAKE_TCR_SEQS), 32)

def test_encode_shape():
    model = SequenceAutoencoder(seq_type="peptide", latent_dim=16, max_len=15)
    x = torch.tensor(encode_sequences(FAKE_PEPTIDE_SEQS, max_len=15))
    z = model.encode(x)
    assert z.shape == (len(FAKE_PEPTIDE_SEQS), 16)

def test_vae_reparameterize():
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=32, vae=True)
    model.train()
    mu = torch.zeros(5, 32)
    logvar = torch.zeros(5, 32)
    z = model.reparameterize(mu, logvar)
    # z should not be all zeros (stochastic)
    assert not torch.allclose(z, mu)


# ─── Training tests ───────────────────────────

def test_fit_and_transform():
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20)
    model.fit(FAKE_TCR_SEQS * 10, epochs=3, batch_size=8)  # tiny training run
    embeddings = model.transform(FAKE_TCR_SEQS)
    assert embeddings.shape == (len(FAKE_TCR_SEQS), 16)
    assert not np.any(np.isnan(embeddings))

def test_transform_before_fit_raises():
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=16)
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(FAKE_TCR_SEQS)

def test_vae_fit():
    model = SequenceAutoencoder(seq_type="peptide", latent_dim=16, max_len=15, vae=True)
    model.fit(FAKE_PEPTIDE_SEQS * 10, epochs=3, batch_size=8)
    z = model.transform(FAKE_PEPTIDE_SEQS)
    assert z.shape == (len(FAKE_PEPTIDE_SEQS), 16)


# ─── Save/load tests ──────────────────────────

def test_save_load(tmp_path):
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20)
    model.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)
    path = str(tmp_path / "tcr_ae.pt")
    model.save(path)

    loaded = SequenceAutoencoder.load(path)
    z1 = model.transform(FAKE_TCR_SEQS)
    z2 = loaded.transform(FAKE_TCR_SEQS)
    assert np.allclose(z1, z2, atol=1e-5)


# ─── Embedder tests ───────────────────────────

def test_autoencoder_embedder():
    tcr_ae = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20)
    pep_ae = SequenceAutoencoder(seq_type="peptide", latent_dim=8, max_len=15)
    tcr_ae.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)
    pep_ae.fit(FAKE_PEPTIDE_SEQS * 5, epochs=2, batch_size=4)

    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    embedder = AutoencoderEmbedder(tcr_ae=tcr_ae, peptide_ae=pep_ae)
    combined = embedder.transform(df)
    assert combined.shape == (5, 16 + 8)

def test_embedder_no_concat():
    tcr_ae = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20)
    pep_ae = SequenceAutoencoder(seq_type="peptide", latent_dim=8, max_len=15)
    tcr_ae.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)
    pep_ae.fit(FAKE_PEPTIDE_SEQS * 5, epochs=2, batch_size=4)

    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    embedder = AutoencoderEmbedder(tcr_ae=tcr_ae, peptide_ae=pep_ae, concat=False)
    result = embedder.transform(df)
    assert result["tcr"].shape == (5, 16)
    assert result["peptide"].shape == (5, 8)


# ─── VAE-specific tests ───────────────────────

def test_vae_z_differs_from_mu_during_training():
    """During training, VAE samples z ~ N(mu, sigma) so z != mu."""
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=32, vae=True)
    model.train()
    x = torch.tensor(encode_sequences(FAKE_TCR_SEQS, max_len=30))
    _, mu, logvar = model(x)
    z = model.reparameterize(mu, logvar)
    assert not torch.allclose(z, mu), "VAE z should differ from mu during training (stochastic)"

def test_vae_z_equals_mu_at_inference():
    """At inference (eval mode), VAE should be deterministic: z == mu."""
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=32, vae=True)
    model.eval()
    x = torch.tensor(encode_sequences(FAKE_TCR_SEQS, max_len=30))
    with torch.no_grad():
        _, mu, logvar = model(x)
        z = model.reparameterize(mu, logvar)
    assert torch.allclose(z, mu), "VAE z should equal mu at inference (deterministic)"

def test_vae_kl_loss_nonzero():
    """VAE loss should exceed plain AE loss due to KL term."""
    seqs = FAKE_TCR_SEQS * 10
    x = torch.tensor(encode_sequences(seqs, max_len=30))

    plain = SequenceAutoencoder(seq_type="tcr", latent_dim=32, vae=False)
    vae   = SequenceAutoencoder(seq_type="tcr", latent_dim=32, vae=True, kl_weight=1.0)

    # Use same weights so only the loss function differs
    vae.load_state_dict(plain.state_dict())

    plain.eval(); vae.eval()
    with torch.no_grad():
        logits_p, mu_p, logvar_p = plain(x)
        logits_v, mu_v, logvar_v = vae(x)
        loss_plain = plain.loss(x, logits_p, mu_p, logvar_p)
        loss_vae   = vae.loss(x, logits_v, mu_v, logvar_v)

    assert loss_vae > loss_plain, "VAE loss should be higher than plain AE loss due to KL term"

def test_vae_and_plain_ae_produce_different_embeddings():
    """After training, plain AE and VAE should produce different latent vectors."""
    plain = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20, vae=False)
    vae   = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20, vae=True)
    plain.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)
    vae.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)

    z_plain = plain.transform(FAKE_TCR_SEQS)
    z_vae   = vae.transform(FAKE_TCR_SEQS)
    assert not np.allclose(z_plain, z_vae), "Plain AE and VAE should learn different embeddings"

def test_vae_repeated_inference_is_deterministic():
    """Calling transform() twice on a fitted VAE should give identical results."""
    model = SequenceAutoencoder(seq_type="tcr", latent_dim=16, max_len=20, vae=True)
    model.fit(FAKE_TCR_SEQS * 5, epochs=2, batch_size=4)
    z1 = model.transform(FAKE_TCR_SEQS)
    z2 = model.transform(FAKE_TCR_SEQS)
    assert np.allclose(z1, z2), "VAE transform() should be deterministic at inference"

def test_vae_latent_space_smoother_than_plain_ae():
    """
    VAE latent vectors should have smaller variance across the batch than plain AE,
    since the KL term regularises z towards N(0,1).
    """
    plain = SequenceAutoencoder(seq_type="tcr", latent_dim=32, max_len=30, vae=False)
    vae   = SequenceAutoencoder(seq_type="tcr", latent_dim=32, max_len=30, vae=True, kl_weight=0.5)
    seqs  = FAKE_TCR_SEQS * 20
    plain.fit(seqs, epochs=5, batch_size=8)
    vae.fit(seqs,   epochs=5, batch_size=8)

    z_plain = plain.transform(FAKE_TCR_SEQS)
    z_vae   = vae.transform(FAKE_TCR_SEQS)
    assert z_vae.std() < z_plain.std(), (
        f"VAE latent std ({z_vae.std():.4f}) should be smaller than "
        f"plain AE latent std ({z_plain.std():.4f}) due to KL regularisation"
    )