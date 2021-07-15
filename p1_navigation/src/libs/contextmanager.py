# Source: https://stackoverflow.com/questions/5136611/capture-stdout-from-a-script

import contextlib
@contextlib.contextmanager
def capture():
    import sys
    import io
    oldout,olderr = sys.stdout, sys.stderr
    try:
        out=[io.StringIO(), io.StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()
