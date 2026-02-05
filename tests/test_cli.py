import argparse

from code_index.cli import build_parser


def test_cli_has_expected_commands():
    parser = build_parser()
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert subparsers
    commands = subparsers[0].choices.keys()
    for name in ("init", "search", "status", "update", "serve"):
        assert name in commands
