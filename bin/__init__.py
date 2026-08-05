"""Command-line tools for the ofrac package.

Each module here exposes `main(argv=None)`, which parses `argv` (defaulting to
`sys.argv[1:]`) and returns an exit status. That signature is what
`[project.scripts]` needs from a console-script target, and it also lets a test
drive a tool by passing an argument list instead of shelling out.

The hyphenated files alongside these are deprecated aliases, kept for callers
that still use the old names. They are scripts, not modules -- a hyphen is not
legal in a Python module name -- so they are not importable from here.
"""
