## Repositories / paths
- `/testbed` — astropy source checkout (git repo).
- Key files:
  - `astropy/modeling/separable.py` — separability logic (`is_separable`, `separability_matrix`, `_separable`, `_coord_matrix`, `_cstack`, `_cdot`, `_arith_oper`, `_operators`).
  - `astropy/modeling/core.py` — `Model`, `CompoundModel`, `ModelDefinitionError`.
  - `astropy/modeling/mappings.py` — `Mapping` model.
  - `astropy/modeling/tests/test_separable.py` — separability test suite.
  - `astropy/timeseries/core.py` — `BaseTimeSeries(QTable)`.
  - `astropy/timeseries/tests/test_common.py`.
  - `astropy/table/table.py` — Table construction (structured-ndarray -> `NdarrayMixin` around lines 1241–1262).
  - `astropy/table/ndarray_mixin.py`, `astropy/table/mixins/registry.py`, `astropy/table/column.py`.
  - `astropy/coordinates/builtin_frames/__init__.py`, `utils.py`, `itrs.py`, `altaz.py`, `hadec.py`, `itrs_observed_transforms.py`.
  - `astropy/coordinates/matrix_utilities.py`, `baseframe.py`, `transformations.py`.
  - `astropy/coordinates/tests/test_intermediate_transformations.py`.
  - `astropy/units/quantity.py` — `Quantity.__array_ufunc__` around lines 620–685 (duck-type deferral inserted before `converters_and_unit`).
  - `astropy/io/ascii/rst.py` — `RST`, `SimpleRSTHeader`, `SimpleRSTData`.
  - `astropy/io/ascii/fixedwidth.py` — `FixedWidth`, `FixedWidthHeader`, `FixedWidthData`, `FixedWidthSplitter`, `FixedWidthTwoLineDataSplitter`.
  - `astropy/io/ascii/core.py` — `DefaultSplitter`, base `BaseHeader/BaseData` machinery.
  - `astropy/io/ascii/tests/test_rst.py`.
  - `docs/changes/<subpkg>/` — Towncrier-style changelog fragments.

## Classes / symbols
- `astropy.modeling.core.Model` attrs used: `n_inputs`, `n_outputs`, `separable`, `_calculate_separability_matrix()`.
- `CompoundModel` attrs used: `left`, `right`, `op`.
- `Mapping(mapping=[...])` model exposes `mapping`, `n_inputs`, `n_outputs`.
- `BaseTimeSeries` (subclass of `QTable`): class-level `_required_columns = None`, `_required_columns_enabled = True`, `_required_columns_relax = False`. `TimeSeries` requires `time` first.
- `astropy.utils.exceptions.AstropyUserWarning`, `AstropyWarning`, `AstropyDeprecationWarning`.
- `EarthLocation.get_itrs(obstime=None)`; `.to_geodetic('WGS84')` -> `(lon, lat, height)`.
- `frame_transform_graph.transform(FunctionTransform, FromFrame, ToFrame)` decorator; `.get_transform(A, B).transforms`.
- `astropy.units.Quantity` (subclass of `np.ndarray`): `__array_ufunc__` implements duck-type deferral then unit-aware conversion via `converters_and_unit`.
- `astropy.utils.masked.Masked` — Masked wrapper (subclass of ndarray, so not deferred by Quantity).
- `np.lib.mixins.NDArrayOperatorsMixin` — commonly used by duck-array classes to trigger `__array_ufunc__` deferral.
- `astropy.io.ascii.rst.RST` — reads/writes reST simple tables; supports `header_rows` kwarg. `_format_name='rst'`.
- `astropy.io.ascii.fixedwidth.FixedWidth.__init__(delimiter_pad, bookend, header_rows=None, ...)`; `header_rows` default `['name']`. `header.header_rows`/`data.header_rows` mirror the param.

## Built-in coordinate frames (from builtin_frames.__init__ __all__)
- `ICRS`, `FK5`, `FK4`, `FK4NoETerms`, `Galactic`, `Galactocentric`, `galactocentric_frame_defaults`, `Supergalactic`, `AltAz`, `HADec`, `GCRS`, `CIRS`, `ITRS`, `HCRS`, `TEME`, `TETE`, `PrecessedGeocentric`, `GeocentricMeanEcliptic`, `BarycentricMeanEcliptic`, `HeliocentricMeanEcliptic`, `GeocentricTrueEcliptic`, `BarycentricTrueEcliptic`, `HeliocentricTrueEcliptic`, `SkyOffsetFrame`, `GalacticLSR`, `LSR`, `LSRK`, `LSRD`, `BaseEclipticFrame`, `BaseRADecFrame`, `make_transform_graph_docs`, `HeliocentricEclipticIAU76`, `CustomBarycentricEcliptic`.

## Example model classes referenced
- `astropy.modeling.models`: `Shift`, `Scale`, `Rotation2D`, `Polynomial2D`, `Linear1D`, `Pix2Sky_TAN`.

## Existing changelog fragments (docs/changes/table/)
- `12631.api.rst`, `12637.api.rst`, `12637.feature.rst`, `12644.feature.rst`, `12680.feature.rst`, `12716.bugfix.rst`, `12825.feature.rst`, `12842.bugfix.rst`, `13129.feature.rst`, `13233.bugfix.rst`, `README.rst`, `template.rst`.
