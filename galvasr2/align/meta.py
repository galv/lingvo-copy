import sys
import json
import argparse

forbidden_keys = ['start', 'end', 'text', 'transcript']


def main(args):
    parser = argparse.ArgumentParser(description='Annotate .tlog or .script files by adding meta data')
    parser.add_argument('target', type=str, help='')
    parser.add_argument('assignments', nargs='+', help='Meta data assignments of the form <key>=<value>')
    args = parser.parse_args()

    with open(args.target, 'r', encoding='utf-8') as json_file:
        entries = json.load(json_file)

    for assignment in args.assignments:
        key, value = assignment.split('=')
        if key in forbidden_keys:
            print('Meta data key "{}" not allowed - forbidden: {}'.format(key, '|'.join(forbidden_keys)))
            sys.exit(1)
        for entry in entries:
            entry[key] = value

    with open(args.target, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, indent=2)


if __name__ == '__main__':
    main(sys.argv[1:])
