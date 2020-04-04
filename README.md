# pept-dev
A private development mirror of the official uob-positron-imaging-centre/pept repository.

We are currently working on:
- Implementing Voxel-based algorithms.
- **Extending the base classes to accommodate Time-of-Flight information** (ie. using two timestamps per LoR). These might be non-trivial and cascade down onto the tracking algorithms.

Commits regarding the ergonomics of the package should go directly to the official repository - that includes:
- Making classes more robust (e.g. have plotting functions select multiple samples from a class instead of just one). At the moment, these changes can break backwards compatibility. Sorry.
- Improving the documentation.


