# Blocked Items

- The public VQ checkpoint download location and redistribution terms require
  author confirmation. The verified local/reviewer artifact hash is documented.
- A full clean-environment 230-scene rerun is not stored in this source tree;
  it requires the external test datasets and VQ artifact.
- Golden pixel regression requires the private reviewer archive's reference
  outputs. Existing aggregate/per-scene CSVs are evidence only and are not used
  as metric inputs.
- The authors' top-level code license and final public repository URL require
  author approval before a reviewer/public link is advertised.

## Resolved Protocol Conflict

The repair task described 256 as the inference default, but the archived
paper-producing pipeline evaluates the frozen model at native scene resolution.
A controlled Mobile Depth run at 256 then upsampled produced 32.3676 dB, while
native inference reproduced the archived `balls` scene within 0.0031 dB
(36.9211 versus 36.9181). Therefore 256 is documented as the training crop and
native resolution is the executable paper protocol.
