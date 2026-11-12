"""Command line interface."""
import argparse
import json
import sys
from . import __version__
from .audit import audit


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='embedding-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = sub.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    command.add_argument('--ignore', action='append', default=[])
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{id} {severity}: {message}'.format(**finding))
        if args.verbose:
            print('Production evidence: ' + ', '.join(result['evidence']))
            print('Test evidence: ' + ', '.join(result['test_evidence']))
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in result['findings']) else 0


if __name__ == '__main__':
    sys.exit(main())
