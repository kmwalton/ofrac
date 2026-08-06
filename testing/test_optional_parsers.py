"""Tests for how ofracs treats the optional parser dependencies.

Parsers for the other simulators' formats live in packages that are not always
installed -- `hgstools` is not even on PyPI -- so populate_parsers() is meant
to leave out what it cannot import and carry on. It must not confuse that with
a dependency that *is* installed and is broken, which is a fault worth
hearing about rather than quietly parsing less.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                      # the repo root, which holds ofrac/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ofrac import ofracs  # noqa: E402


def _raising(name):
    """An import_module stand-in that fails as a missing `name` would."""
    def _import(modname):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return _import


def _all_absent(modname):
    """An import_module stand-in for a machine with none of the optional
    packages: whatever is asked for, its top-level package is missing."""
    top = modname.split('.')[0]
    raise ModuleNotFoundError(f"No module named {top!r}", name=top)


class TestOptionalImport(unittest.TestCase):
    """_optional_import tells absence apart from a broken install."""

    def _patched(self, name):
        return mock.patch.object(ofracs.importlib, 'import_module',
            side_effect=_raising(name))

    def test_absent_dependency_is_not_an_error(self):
        with self._patched('hgstools'):
            self.assertIsNone(
                ofracs._optional_import('hgstools.pyhgs.parser.fractran'))

    def test_absent_intermediate_package_is_not_an_error(self):
        with self._patched('hgstools.pyhgs'):
            self.assertIsNone(
                ofracs._optional_import('hgstools.pyhgs.parser.fractran'))

    def test_absent_module_itself_is_not_an_error(self):
        with self._patched('hgstools.pyhgs.parser.fractran'):
            self.assertIsNone(
                ofracs._optional_import('hgstools.pyhgs.parser.fractran'))

    def test_flat_pyhgs_alias_is_tolerated(self):
        """hgstools imports its own subpackage flatly in a couple of places."""
        with self._patched('pyhgs'):
            self.assertIsNone(
                ofracs._optional_import('hgstools.pyhgs.parser.fractran'))

    def test_a_dependency_of_the_dependency_raises(self):
        """Installed but unusable is a fault, not an absence."""
        with self._patched('regex'):
            with self.assertRaises(ModuleNotFoundError) as caught:
                ofracs._optional_import('hgstools.pyhgs.parser.fractran')
        self.assertEqual(caught.exception.name, 'regex')

    def test_a_near_miss_name_still_raises(self):
        """'hgstool' is not a package prefix of 'hgstools...' -- do not match
        on a bare string prefix."""
        with self._patched('hgstool'):
            with self.assertRaises(ModuleNotFoundError):
                ofracs._optional_import('hgstools.pyhgs.parser.fractran')

    def test_a_present_dependency_is_returned(self):
        with mock.patch.object(ofracs.importlib, 'import_module',
                return_value='the module'):
            self.assertEqual(
                ofracs._optional_import('hgstools.pyhgs.parser.fractran'),
                'the module')


class TestPopulateParsers(unittest.TestCase):

    def setUp(self):
        # populate_parsers is cached and appends to a module-level list, so
        # both have to be wound back or these tests read each other's leavings
        ofracs.populate_parsers.cache_clear()
        self._warnings = list(ofracs._import_warning_strings)
        ofracs._import_warning_strings.clear()

    def tearDown(self):
        ofracs.populate_parsers.cache_clear()
        ofracs._import_warning_strings[:] = self._warnings

    def test_no_optional_parsers_still_yields_the_pickle_reader(self):
        """The reported bug: a clean install could not parse anything."""
        with mock.patch.object(ofracs.importlib, 'import_module',
                side_effect=_all_absent):
            parsers = ofracs.populate_parsers()

        self.assertEqual(list(parsers), [ofracs.OFracGrid.PickleParser])
        self.assertTrue(ofracs._import_warning_strings,
            'the unavailable parsers should be reported')

    def test_a_broken_optional_dependency_propagates(self):
        with mock.patch.object(ofracs.importlib, 'import_module',
                side_effect=_raising('regex')):
            with self.assertRaises(ModuleNotFoundError):
                ofracs.populate_parsers()

    def test_warnings_do_not_accumulate_over_repeated_calls(self):
        """parse() calls this per file, so appending each time repeated the
        same warnings once per input."""
        with mock.patch.object(ofracs.importlib, 'import_module',
                side_effect=_all_absent):
            ofracs.populate_parsers()
            after_one = len(ofracs._import_warning_strings)
            ofracs.populate_parsers()
            ofracs.populate_parsers()

        self.assertEqual(len(ofracs._import_warning_strings), after_one)

    def test_the_parser_list_cannot_be_mutated_by_a_caller(self):
        """It is cached and handed to every caller."""
        with mock.patch.object(ofracs.importlib, 'import_module',
                side_effect=_all_absent):
            parsers = ofracs.populate_parsers()

        with self.assertRaises(AttributeError):
            parsers.append(object)


if __name__ == '__main__':
    unittest.main()
