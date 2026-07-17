## Environment / Repo
- Codebase is an editable/installed astropy checkout at `/testbed` (git-tracked). `git diff` shows in-place edits.
- Python execution and pytest work offline; a global "Internet access disabled" notice is printed by test infra.
- A `RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 80 from C header, got 96 from PyObject` is emitted on import (benign, from binary/ABI skew).
- `astropy.time` leap-second auto-update fails offline with `AstropyWarning: leap-second auto-update failed ... IERSStaleWarning('leap-second file is expired.')`; this can cause `test_common.py::TestTimeSeries::test_join` to fail on warnings-as-errors runs.
- Offline IERS state also breaks some coordinate tests: `astropy/utils/iers/iers.py:271` raises `TypeError: unsupported operand type(s) for -: 'Time' and 'float'` at `mjd = np.floor(jd1 - MJD_ZERO + jd2)` — surfaces in e.g. `test_icrs_cirs`, `test_straight_overhead`.
- Pre-existing failures in unrelated suites (present on clean `main`):
  - `astropy/table/tests/test_mixin.py::test_skycoord_representation` fails with `TypeError: concatenate() got an unexpected keyword argument 'dtype'` from `astropy/units/quantity.py:1688`.
  - In `astropy/units/tests/test_quantity_non_ufuncs.py`: `TestLinAlg::test_eig`, `test_eigh` fail with `TypeError: <lambda>() missing 1 required positional argument: 'eigenvectors'`; `test_svd`, `test_qr` fail similarly; `test_testing_completeness` and `TestFunctionHelpersCompleteness::test_all_included` fail with set-equality asserts; `TestUfuncLike::test_round_` fails with `DeprecationWarning: 'round_' is deprecated as of NumPy 1.25.0`.
  - In `astropy/units/tests/test_quantity_array_methods.py`: `TestQuantityStatsFuncs::test_min/max*` fail with `AstropyWarning: function 'min'/'max' is not known to astropy`.
  - In `astropy/utils/masked/tests/test_function_helpers.py`: `TestMethodLikes::test_cumproduct`, `test_round_` (NumPy 1.25 deprecations); `TestUfuncLike::test_fix` (`TypeError: cannot write to unmasked output`); `test_basic_testing_completeness`.

## astropy.modeling.separable rules
- `is_separable(transform)`:
  - If `n_inputs == 1 and n_outputs > 1`: returns `np.array([False]*n_outputs).T`.
  - Otherwise returns `where(separable_matrix.sum(axis=1) != 1, False, True)`.
- `separability_matrix(transform)`:
  - If `n_inputs == 1 and n_outputs > 1`: returns `np.ones((n_outputs, n_inputs), dtype=bool)`.
  - Otherwise returns boolean-cast of `_separable(transform)`.
- `_separable` dispatches:
  1. `transform._calculate_separability_matrix()` if it returns non-`NotImplemented`.
  2. Else if `CompoundModel`, apply `_operators[op]` to recursive `_separable(left/right)`.
  3. Else `_coord_matrix(transform, 'left', n_outputs)`.
- Operators map: `{'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper, '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}`.
- Arithmetic operators (`+ - * / **`) always yield a fully non-separable (all-ones) result and require matching `n_inputs`/`n_outputs`, else raise `ModelDefinitionError`.
- `|` operator (`_cdot`) uses `np.dot(cleft, cright)`. On `ValueError` it raises `ModelDefinitionError`.
- Bug/fix in `_cstack` (`&` operator): right ndarray must be placed as `cright[-right.shape[0]:, -right.shape[1]:] = right` (not `= 1`).
- Separability tests live at `astropy/modeling/tests/test_separable.py` (11 tests, pass after fix).

## astropy.timeseries.core `_check_required_columns`
- Runs only when `self._required_columns_enabled` and `self._required_columns is not None`.
- Uses `_required_columns_relax` mode: while relaxed, only the prefix `self._required_columns[:len(self.colnames)]` is required. Toggles off automatically once all required columns are present at the start of `colnames`.
- Raises `ValueError` with `"{cls} object is invalid - ..."` messages (see schemas).
- `_delay_required_column_checks()` context manager toggles `_required_columns_enabled` off during the block, then re-runs `_check_required_columns` on exit.

## astropy.table.Table column addition
- When adding data that is not a `Column`/mixin and is an `np.ndarray` with `len(data.dtype) > 1` (structured array), Table historically auto-converts via `data.view(NdarrayMixin)`.
- Current/expected behavior: emit a `FutureWarning` before the view, telling users to wrap in `Column(data)` or `data.view(NdarrayMixin)`. Warning issued around line 1243 of `astropy/table/table.py`.
- Mixin-handler path: if `get_mixin_handler(data)` returns a handler, its output must satisfy `_is_mixin_for_table`, else raises `TypeError('Mixin handler for object of type {module}.{cls} did not return a valid mixin column')`.

## astropy.coordinates ITRS <-> Observed (AltAz/HADec) direct transforms
- Direct `FunctionTransform`s registered ITRS<->AltAz and ITRS<->HADec treat ITRS coords as time-invariant; they do NOT synchronize obstimes.
- Transform math:
  - Topocentric ITRS = `itrs.cartesian - observed_frame.location.get_itrs().cartesian` (forward), then apply `itrs_to_observed_mat(observed_frame)`.
  - Reverse: apply `matrix_transpose(itrs_to_observed_mat(observed_coo))` then add `observed_coo.location.get_itrs().cartesian`.
  - AltAz matrix: `diag(-1,1,1) @ Ry(pi/2 - lat) @ Rz(lon)`.
  - HADec matrix: `diag(1,-1,1) @ Rz(lon)`.
  - Location taken from `observed_frame.location.to_geodetic('WGS84')`.
- Round-trip ITRS->AltAz->ITRS and ITRS->HADec->ITRS is exact (0 m) for a fixed obstime.
- Registration is in `astropy/coordinates/builtin_frames/itrs_observed_transforms.py`, imported from `astropy/coordinates/builtin_frames/__init__.py`.

## astropy.units.Quantity.__array_ufunc__ duck-type deferral
- At the top of `Quantity.__array_ufunc__`, before `converters_and_unit(...)`, iterate over `inputs + kwargs.get('out', ())` and return `NotImplemented` for any item that:
  1. has an `__array_ufunc__` attribute,
  2. is not an `np.ndarray` instance,
  3. has non-`None` `__array_ufunc__`.
  This lets NumPy defer to duck-array types (e.g., dataclasses using `NDArrayOperatorsMixin`) rather than raising a `UnitConversionError` when the Quantity converter is invoked. Users still get `UnitConversionError` when combining plain Quantities with incompatible units.
- Behavior is verified against a `DuckArray(NDArrayOperatorsMixin)` wrapping a `Quantity`: binary ops in either order and `np.<ufunc>` calls succeed and return the duck type.
- `Masked(Quantity)` (subclass of ndarray) is unaffected — `isinstance(item, np.ndarray)` short-circuits the check.

## astropy.io.ascii RST writer/reader
- `astropy.io.ascii.rst.RST` (`_format_name='rst'`) is a `FixedWidth` subclass with `bookend=False`, `delimiter_pad=None`.
- `SimpleRSTHeader`: `position_line=0`, `start_line=1`, `position_char='='`; right-most column end is unbounded (`ends[-1] = None`).
- `SimpleRSTData`: `splitter_class=FixedWidthTwoLineDataSplitter`, `end_line=-1`. `start_line` for reading is set dynamically in `RST.read` to `2 + len(self.header.header_rows)` so extra header rows (e.g., `unit`) are skipped.
- `RST.__init__` accepts `header_rows=None` and forwards to `FixedWidth.__init__(..., header_rows=header_rows)`. Without `header_rows` the default header is `['name']` (as inherited).
- `RST.write(lines)` wraps the `FixedWidth` output with a top and bottom `=====` position line copied from `lines[len(self.header.header_rows)]` (the position line lives just after the header rows).
- With `header_rows=['name','unit']` the output is: position line, name row, unit row, position line, data rows..., position line.
- Reading tables that use continuation lines or column-span dashes is not supported.
- `astropy/io/ascii/tests/test_rst.py` (9 tests) passes in ~0.02s.

## System limits observed
- Full test file `astropy/modeling/tests/test_separable.py` runs in ~0.16s (11 tests). `astropy/timeseries/tests/test_common.py` runs in <1s (24 tests).
- `astropy/coordinates/tests/test_intermediate_transformations.py` runs ~1.5s (69 collected).
- `astropy/table/tests/test_table.py -k structured/ndarray/mixin`: 4 tests in ~0.56s.
- `astropy/units/tests/test_quantity_ufuncs.py`: 201 passed + 4 skipped in ~0.38s.
- `astropy/units/` full: ~2921 passed, ~9.7s (with pre-existing failures noted above).
- `astropy/utils/masked/` full: ~924 passed in ~2.76s.
