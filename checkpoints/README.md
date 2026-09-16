# Checkpoints

Weights are distributed separately and are intentionally not committed here.
Inference requires the frozen RFfusion `autoencoder.ckpt`, an LFSF checkpoint,
and, for LFSF-RF, an LFSF-RF checkpoint. Pass their locations explicitly to
`scripts/infer.py`; filenames are not prescribed.

The hashes of the paper checkpoints are recorded in `checksums.txt`.
