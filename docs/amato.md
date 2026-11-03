# The Amato Stage-1 family

The Amato (2020) spatio-temporal deep-learning family is one of the
nine Stage-1 interpolators compared in Schicker et al. (2026). The
reference implementation lives in
<https://github.com/federhub/ST_DeepLearning>.

## Why it is a stub in this library

The upstream repository does not declare a licence. Under default
copyright law that is "all rights reserved by the author", which
means we cannot redistribute even a modified fork. Our port of the
two files

- `src/austrianwindatlas/stage1/amato/svd.py`
- `src/austrianwindatlas/stage1/amato/reconstruct.py`

is therefore shipped as a stub that raises
`NotImplementedError`. The error message points you at the upstream
repository.

## How to use the Amato family anyway

1. Clone the upstream repository alongside this library.
2. Follow the setup in `federhub/ST_DeepLearning/README.md` to
   reproduce the Amato training on your own data.
3. Our Enhanced-MAE and Enhanced-WMSE modules
   (`src/austrianwindatlas/stage1/enhanced_mae.py`) are
   *independent* implementations in the same architectural family,
   written from scratch; they are shipped under MIT and work
   out of the box. They are the ones that actually appear in the
   paper's multi-criteria ranking.

## When this stub becomes a real module

Permission has been requested from the upstream author to
redistribute the modified files under MIT with attribution and a
citation to Amato et al. (2020). If granted, the stub is replaced
by the real implementation, a `LICENSE.upstream` file is added
inside the `amato/` directory with the upstream notice verbatim,
and the `NOTICE` entry is updated.

Status (as of 2026-11-03): request pending; see the paper repo's
`AWA_TRANSITION.md` for the latest state.
