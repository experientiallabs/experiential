"""`python -m wmo`: the same CLI as the `wmo` console script.

Exists so a subprocess can invoke the CLI through the interpreter that imported this
package (`sys.executable -m wmo ...`) without depending on PATH - which is how
`wmo reproduce` replays a `commands` manifest with the exact CLI the manifest pins.
"""

from wmo.cli import main

if __name__ == "__main__":
    main()
