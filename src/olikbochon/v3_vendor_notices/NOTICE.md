# Version 3 bundled Unicode dependencies

The self-contained Kaggle runtime vendors the Python source required by the
locked default normalizer:

- `ftfy` 6.0.3 by Robyn Speer, distributed under the MIT License. Source
  release: `https://pypi.org/project/ftfy/6.0.3/` and
  `https://github.com/rspeer/python-ftfy`.
- `wcwidth` 0.8.2 by Jeff Quast and contributors, distributed under the MIT
  License. Source release: `https://pypi.org/project/wcwidth/0.8.2/` and
  `https://github.com/jquast/wcwidth`.

The complete applicable license texts are stored beside this notice as
`FTFY_LICENSE.txt` and `WCWIDTH_LICENSE.txt`. Command-line entry points,
documentation, tests, distribution metadata, bytecode, caches, and wcwidth's
unreached terminal-specific APIs/data tables are not bundled. The wcwidth
initializer is adapted to export only the upstream `wcwidth` and `wcswidth`
functions required by ftfy. The dependency files listed in
`VENDOR_MANIFEST.json` are embedded deterministically in the generated notebook
runtime. Two inert trailing-space characters were removed from the ftfy source;
runtime normalization behavior is unchanged.
