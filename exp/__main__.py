"""`python -m exp`: the same CLI as the `exp` console script.

Exists so a subprocess can invoke the CLI through the interpreter that imported this
package (`sys.executable -m exp ...`) without depending on PATH, which is what a caller
replaying a pinned command manifest needs.
"""

from exp.cli import main

if __name__ == "__main__":
    main()
