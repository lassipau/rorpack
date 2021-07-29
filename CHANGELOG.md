# Changelog for RORPack

The main changes to "RORPack" are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added 

- This Changelog file.
- Timoshenko beam example from the conference paper by Paunonen, Le Gorrec, and Ramirez at LHMNC 2018.
- Feedthrough parameter Dc for LowGainRC.

### Changed

- Changed comparison with "is" to "==" in laplacian.py.
- Changed the sign of G2 in controller/construct_internal_model.
- Changed the sign of G2 in LowGainRC and PassiveRC accordingly with the above change.

### Removed

- 

### Fixed

- Fixed signs for matrices B and Bd and stabilizing matrices K and L for the controller construction in examples heat1d 1, 2 and 3.

## [v0.9.0] - 2021-07-07

### Added 

- 

[unreleased]: https://github.com/lassipau/rorpack/tree/dev
[v0.9.0]: 

