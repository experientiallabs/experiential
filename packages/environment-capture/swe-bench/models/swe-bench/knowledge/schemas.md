## separability_matrix / is_separable outputs
- `separability_matrix(transform)` -> `np.ndarray` bool, shape `(n_outputs, n_inputs)`.
- `is_separable(transform)` -> 1-D bool `np.ndarray` length `n_outputs`.
- `_coord_matrix(model, pos, noutp)` -> `np.ndarray` shape `(noutp, model.n_inputs)` with 0/1 float.
- `_cstack` returns `np.hstack([cleft, cright])`; `_cdot` returns `np.dot(cleft, cright)`; `_arith_oper` returns `np.ones((left_outputs, left_inputs))`.

## ITRS <-> Observed matrices
- `itrs_to_observed_mat(observed_frame)` -> 3x3 `np.ndarray`.
  - `AltAz`: `diag(-1,1,1) @ rotation_matrix(pi/2 - lat, 'y') @ rotation_matrix(lon, 'z')`.
  - Else (HADec): `diag(1,-1,1) @ rotation_matrix(lon, 'z')`.
- Forward: `observed_frame.realize_frame((itrs.cartesian - loc.get_itrs().cartesian).transform(M))`.
- Reverse: `itrs_frame.realize_frame(observed.cartesian.transform(M.T) + loc.get_itrs().cartesian)`.

## Quantity.__array_ufunc__ deferral check
- Executed before `converters_and_unit`:
  ```python
  out = kwargs.get("out", ())
  for item in inputs + out:
      if (hasattr(item, "__array_ufunc__")
          and not isinstance(item, np.ndarray)
          and item.__array_ufunc__ is not None):
          return NotImplemented
  ```
- Note: `out` kwarg is always a tuple in the ufunc protocol, safe to concatenate.

## RST writer output shape
- Without `header_rows` (default `['name']`): 3 position-line rows total (top/mid/bottom `=====`), one name row, N data rows.
- With `header_rows=['name','unit']`: `[pos, name, unit, pos, data..., pos]`.
- Position line is copied from `lines[len(self.header.header_rows)]` produced by the underlying `FixedWidth.write`.
- `RST.read(table)` sets `self.data.start_line = 2 + len(self.header.header_rows)` before delegating to `super().read`.

## Error messages / formats
- `ModelDefinitionError` from `_arith_oper`:
  `"Unsupported operands for arithmetic operator: left (n_inputs={li}, n_outputs={lo}) and right (n_inputs={ri}, n_outputs={ro}); models must have the same n_inputs and the same n_outputs for this operator."`
- `ModelDefinitionError` from `_cdot`:
  `'Models cannot be combined with the "|" operator; left coord_matrix is {cright}, right coord_matrix is {cleft}'` (variables intentionally swapped in message text).
- `ValueError` from `BaseTimeSeries._check_required_columns`:
  - `"{cls} object is invalid - expected '{col0}' as the first column{plural} but time series has no columns"`
  - `"{cls} object is invalid - the following required column{s} {is|are} missing: {comma_joined_missing}"`
  - `"{cls} object is invalid - expected '{col0}' as the first column{plural} but found '{colnames[0]}'"`
  - `plural = 's' if len(required_columns) > 1 else ''` in all three.
- `TypeError` in Table column add: `"Mixin handler for object of type {module}.{cls} did not return a valid mixin column"`.
- `FutureWarning` in Table column add for structured ndarray (issued before `data.view(NdarrayMixin)`), text: `"Adding a column of a structured numpy array to a table has historically converted it to a `~astropy.table.NdarrayMixin` column. In the future, the structured array will instead be stored in a regular `~astropy.table.Column`. To avoid this warning and get the future behavior, wrap the data in a `~astropy.table.Column` explicitly, e.g. ``t['col'] = Column(data)``. To retain the current behavior, wrap the data in a `~astropy.table.NdarrayMixin`, e.g. ``t['col'] = data.view(NdarrayMixin)``."` with `category=FutureWarning`.
- Offline IERS `TypeError`: `TypeError: unsupported operand type(s) for -: 'Time' and 'float'` from `astropy/utils/iers/iers.py:271`.
- Quantity `__array_function__` failure (pre-existing): `TypeError: concatenate() got an unexpected keyword argument 'dtype'` from `astropy/units/quantity.py:1688`.
- `UnitConversionError` still raised when combining plain Quantities of incompatible units; only duck arrays trigger `NotImplemented` deferral.

## Test runner surface
- `pytest` prints `Internet access disabled` banner.
- Common benign import warning: `RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 80 from C header, got 96 from PyObject`.
- Leap-second offline: `AstropyWarning: leap-second auto-update failed due to the following exception: IERSStaleWarning('leap-second file is expired.')`.
