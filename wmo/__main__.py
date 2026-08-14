"""`python -m wmo`: the same CLI as the `wmo` console script.

Exists so a subprocess can invoke the CLI through the interpreter that imported this
package (`sys.executable -m wmo ...`) without depending on PATH, which is what a caller
replaying a pinned command manifest needs.
"""

from wmo.cli import main

if __name__ == "__main__":
    main()
