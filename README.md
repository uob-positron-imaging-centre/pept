# pept-dev
A private development mirror of the official uob-positron-imaging-centre/pept repository.

We are currently working on:
- Implementing Voxel-based algorithms.
- **Extending the base classes to accommodate Time-of-Flight information** (ie. using two timestamps per LoR). These might be non-trivial and cascade down onto the tracking algorithms.

Commits regarding the ergonomics of the package should go directly to the official repository - that includes:
- Making classes more robust (e.g. have plotting functions select multiple samples from a class instead of just one). At the moment, these changes can break backwards compatibility. Sorry.
- Improving the documentation.

Suggestion for working on the package:
1. Clone the repository in your terminal:
```
git clone https://github.com/anicusan/pept-dev
```

2. Install the package from the repo **in edit-mode**. This means that any change you make to the package will automatically be reflected in your code that uses it, without having to reinstall it.
```
cd pept-dev
pip install -e .
```

3. Now you can add new stuff to the package. You can write scripts in another directory to test the new features you've added.


