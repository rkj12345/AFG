# Third-party code and weights

This release includes code derived from CUT3R, DUSt3R and CroCo. The original
copyright notices in source files are retained. The root `LICENSE` describes
the CUT3R Creative Commons Attribution-NonCommercial-ShareAlike 4.0 terms and
also contains an MIT notice from the TTT3R repository. CroCo has its own
`src/croco/LICENSE` and `src/croco/NOTICE`. Review these terms before use or
redistribution; the presence of the MIT notice does not remove restrictions
on third-party components.

`src/dust3r/update_gates.py` adapts the TTSA3R update rule from the source
TTT3R working tree and the MeMix patch selector from
https://github.com/dongjiacheng06/MeMix. MeMix is licensed under CC BY-NC-SA
4.0; its own repository provides the full method, additional options, and
evaluation harness. The AFG release includes the default MeMix selector and
three update combinations (`cut_memix`, `ttt_memix`, `ttsa_memix`).

The CUT3R checkpoint is not included. Download it from the CUT3R/TTT3R link
in `README.md` and comply with its applicable terms.
