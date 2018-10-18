#!/usr/bin/env python2

from optparse import OptionParser


import argparse
from argparse import ArgumentParser
import sys
import gbench
from gbench import util, report
from gbench.util import *


def check_inputs(in1, in2, flags):
    """
    Perform checking on the user provided inputs and diagnose any abnormalities
    """
    in1_kind, in1_err = classify_input_file(in1)
    in2_kind, in2_err = classify_input_file(in2)
    output_file = find_benchmark_flag('--benchmark_out=', flags)
    output_type = find_benchmark_flag('--benchmark_out_format=', flags)
    if in1_kind == IT_Executable and in2_kind == IT_Executable and output_file:
        print(("WARNING: '--benchmark_out=%s' will be passed to both "
               "benchmarks causing it to be overwritten") % output_file)
    if in1_kind == IT_JSON and in2_kind == IT_JSON and len(flags) > 0:
        print("WARNING: passing optional flags has no effect since both "
              "inputs are JSON")
    if output_type is not None and output_type != 'json':
        print(("ERROR: passing '--benchmark_out_format=%s' to 'compare.py`"
               " is not supported.") % output_type)
        sys.exit(1)
        
        
def create_parser():
    parser = ArgumentParser(
        description='versatile benchmark output compare tool')
    subparsers = parser.add_subparsers(
        help='This tool has multiple modes of operation:',
        dest='mode')

    parser_a = subparsers.add_parser(
        'benchmarks',
        help='The most simple use-case, compare all the output of these two benchmarks')
    baseline = parser_a.add_argument_group(
        'baseline', 'The benchmark baseline')
    baseline.add_argument(
        'test_baseline',
        metavar='test_baseline',
        type=argparse.FileType('r'),
        nargs=1,
        help='A benchmark executable or JSON output file')
    contender = parser_a.add_argument_group(
        'contender', 'The benchmark that will be compared against the baseline')
    contender.add_argument(
        'test_contender',
        metavar='test_contender',
        type=argparse.FileType('r'),
        nargs=1,
        help='A benchmark executable or JSON output file')
    parser_a.add_argument(
        'benchmark_options',
        metavar='benchmark_options',
        nargs=argparse.REMAINDER,
        help='Arguments to pass when running benchmark executables')



    parser_b = subparsers.add_parser(
        'filters', help='Compare filter one with the filter two of benchmark')
    baseline = parser_b.add_argument_group(
        'baseline', 'The benchmark baseline')
    baseline.add_argument(
        'test',
        metavar='test',
        type=argparse.FileType('r'),
        nargs=1,
        help='A benchmark executable or JSON output file')
    baseline.add_argument(
        'filter_baseline',
        metavar='filter_baseline',
        type=str,
        nargs=1,
        help='The first filter, that will be used as baseline')
    contender = parser_b.add_argument_group(
        'contender', 'The benchmark that will be compared against the baseline')
    contender.add_argument(
        'filter_contender',
        metavar='filter_contender',
        type=str,
        nargs=1,
        help='The second filter, that will be compared against the baseline')
    parser_b.add_argument(
        'benchmark_options',
        metavar='benchmark_options',
        nargs=argparse.REMAINDER,
        help='Arguments to pass when running benchmark executables')



    parser_c = subparsers.add_parser(
        'benchmarksfiltered',
        help='Compare filter one of first benchmark with filter two of the second benchmark')
    baseline = parser_c.add_argument_group(
        'baseline', 'The benchmark baseline')
    baseline.add_argument(
        'test_baseline',
        metavar='test_baseline',
        type=argparse.FileType('r'),
        nargs=1,
        help='A benchmark executable or JSON output file')
    baseline.add_argument(
        'filter_baseline',
        metavar='filter_baseline',
        type=str,
        nargs=1,
        help='The first filter, that will be used as baseline')
    contender = parser_c.add_argument_group(
        'contender', 'The benchmark that will be compared against the baseline')
    contender.add_argument(
        'test_contender',
        metavar='test_contender',
        type=argparse.FileType('r'),
        nargs=1,
        help='The second benchmark executable or JSON output file, that will be compared against the baseline')
    contender.add_argument(
        'filter_contender',
        metavar='filter_contender',
        type=str,
        nargs=1,
        help='The second filter, that will be compared against the baseline')
    parser_c.add_argument(
        'benchmark_options',
        metavar='benchmark_options',
        nargs=argparse.REMAINDER,
        help='Arguments to pass when running benchmark executables')



    parser_d = subparsers.add_parser(
        'hpcombi', help='Compare filter one with the filter two of benchmark')

    parser_d.add_argument(
        'tests',
        metavar='tests',
        type=str,
        nargs=1,
        help='A benchmark executable or JSON output files')
    parser_d.add_argument(
        'comps',
        metavar='comps',
        type=str,
        nargs=1,
        help='Comparisons to do')
    parser_d.add_argument(
        'same',
        metavar='same',
        type=str,
        nargs=1,
        help='Constant parameters')
    parser_d.add_argument(
        'benchmark_options',
        metavar='benchmark_options',
        nargs=argparse.REMAINDER,
        help='Arguments to pass when running benchmark executables')
    return parser
    
    
def main():
	# ~ parser = OptionParser()
	# ~ parser.add_option("-f", "--files", type="string", dest="files", default='', help ="Files to parse (default : )")
	# ~ parser.add_option("-c", "--compare", type="string", dest="comp", default='', help ="Parameters to compare (default : )")
	# ~ parser.add_option("-s", "--same", type="string", dest="same", default='', help ="Same parameters (default : )")
	# ~ (options, args) = parser.parse_args()
	# ~ options.files
	# ~ options.comp
	# ~ options.same



    # Parse the command line flags
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()
    assert not unknown_args
    benchmark_options = args.benchmark_options




    if args.mode == 'filters':
        test_baseline = args.test[0].name
        test_contender = args.test[0].name
        filter_baseline = args.filter_baseline[0]
        filter_contender = args.filter_contender[0]
        expFilter = ''
        # NOTE: if filter_baseline == filter_contender, you are analyzing the
        # stdev

        description = 'Comparing %s to %s (from %s)' % (
            filter_baseline, filter_contender, args.test[0].name)


    if args.mode == 'hpcombi':
        # ~ print args.tests
        # ~ print args.comps
        # ~ print args.same
        tests_baseline = set( args.tests[0].split(',') )
        tests_contender = set( args.tests[0].split(',') )
        comps = set( args.comps[0].split(',') )
        same = set( args.same[0].split(',') )
        same.discard('')
        filter_baseline = ''
        filter_contender = ''
        
        new_tests_baseline = set()
        for test in tests_baseline:
            new_tests_baseline |= get_files_set(test)
        tests_baseline = new_tests_baseline
        new_tests_contender = set()
        for test in tests_contender:
            new_tests_contender |= get_files_set(test)
        tests_contender = new_tests_contender
        
        description = 'Comparisons: %s\nConstant parameters: %s'% \
                                ( ', '.join( comps ), ', '.join( same ) ) \
                                 + '\nFiles: ' + ', '.join( (tests_baseline | tests_contender) )

    for testA in tests_baseline :
        for testB in tests_contender :
            check_inputs(testA, testB, benchmark_options)

    options_baseline = []
    options_contender = []

    if filter_baseline and filter_contender:
        options_baseline = ['--benchmark_filter=%s' % filter_baseline]
        options_contender = ['--benchmark_filter=%s' % filter_contender]

    # Run the benchmarks and report the results
    json1_orig = {'benchmarks':[]}
    json2_orig = {'benchmarks':[]}
    for testA in tests_baseline :
        json1_orig['benchmarks'] += gbench.util.run_or_load_benchmark(
            testA, benchmark_options + options_baseline)['benchmarks']
    for testB in tests_contender :
        json2_orig['benchmarks'] += gbench.util.run_or_load_benchmark(
            testB, benchmark_options + options_contender)['benchmarks']

    names = [ bench['name'].split('_') for bench in json1_orig['benchmarks'] ]
    special = ['+', '*', '(', ')', '[', ']', '{', '}', '^', '$'] 
    nameSets = []
    for name in names :
        for i, word in enumerate(name) :
            # Escape special caractere for regular expression
            for car in special :
               word = word.replace(car, '\\' + car)
            try:
                nameSets[i].add(word)
            except IndexError:
                nameSets.append( {word} )
    
    same = gbench.util.get_same_set(same, nameSets)
    comps = gbench.util.get_comps_list( json1_orig, comps, same, nameSets)
    expFilter = gbench.util.get_regex(same, nameSets)

    json1 = {'benchmarks':[]}
    json2 = {'benchmarks':[]}
    # Now, filter the benchmarks so that the difference report can work
    for comp in comps:
        replacement = '[%s vs. %s]' % tuple(comp.split('/'))
        json1['benchmarks'] += gbench.report.filter_benchmark(
            json1_orig, comp.split('/')[0], replacement, expFilter)['benchmarks']
        json2['benchmarks'] += gbench.report.filter_benchmark(
            json2_orig, comp.split('/')[1], replacement, expFilter)['benchmarks']

    # Diff and output
    output_lines = gbench.report.generate_difference_report(json1, json2)
    print(description)
    for ln in output_lines:
        print(ln)
if __name__ == '__main__':
    main()
