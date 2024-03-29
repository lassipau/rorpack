# Changelog for RORPack

The main changes to "RORPack" are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [unreleased]

### Added

-

### Changed

-

### Removed

-

### Fixed

-


## [v1.0.0]

### Added 

- This Changelog file.
- Timoshenko beam example from the conference paper by Paunonen, Le Gorrec, and Ramirez at LHMNC 2018.
- Feedthrough parameter Dc for LowGainRC.
- Observer-based ROM controller.
- PassiveRC now accepts scalar feedthrough parameter Dc. If scalar Dc is given, it is interpreted as Dc*I, where I is dim_Y x dim_Y.

### Changed

- Changed comparison with "is" to "==" in laplacian.py.
- Changed the sign of G2 in controller/construct_internal_model.
- Changed the sign of G2 in LowGainRC and PassiveRC accordingly with the above change.
- Transfer function values PKvals and PLvals inserted to ObserverBasedRC and DualObserverBasedRC and removed from the examples.
- ObserverBasedRC and DualObserverBasedRC no longer use PKvals or PLvals. Old implementations are still in controller.py.
- Updated the documentation to match the changes listed here.

### Removed

- PKvals, PLvals, RLBLvals, CKRKvlas as arguments for ObserverBasedRC and DualObserverBasedRC.

### Fixed

- Fixed signs for matrices B and Bd and stabilizing matrices K and L for the controller construction in examples heat1d 1, 2 and 3.
- Fixed the simulation of the closed loop system to use the correct matrix CK in determining the control input signal.
- Floating point rounding handling in heat_1d_3.py.


[unreleased]: https://github.com/lassipau/rorpack/tree/dev
[v1.0.0]: https://github.com/lassipau/rorpack/releases/tag/v1.0.0

