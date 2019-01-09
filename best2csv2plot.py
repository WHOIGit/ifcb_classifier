"""This script aggregates and presents summary results from multiple runs"""

# imports
from subprocess import run, PIPE
from ast import literal_eval
import pandas as pd
import sys, argparse

# final columns to show
index_col = 'name'
columns = ['f1','epoch','validation_loss','hours_elapsed','mins_per_epoch']

# parsing command line inputs to determine output
parser = argparse.ArgumentParser(description='List output/*.best.txt result stats to stdout')
parser.add_argument('--name', type=str, default='*', help='specify which <name>.best.txt summary files to search for in output/ . Uses wildcards. Default *')
parser.add_argument('--csv', nargs='?', default=False, const=True, type=str, help='output as csv. may specify output filename')
args = parser.parse_args()

# collecting filenames/paths where data is stored
search_string = '{}.best.txt'.format(args.name)
cmd = ['find', 'output', '-type','f', '-name', search_string]
results = run(cmd,stdout=PIPE).stdout
results = results.decode().strip().split('\n')

# aggregating file content to DataFrame
df = pd.DataFrame(columns = columns)
for result in results:
    with open(result) as f:
        content = f.read()
        content = literal_eval(content)
        df = df.append(content,ignore_index=True)

# processing some data columns
df = df.set_index(index_col)
df['hours_elapsed'] = [secs/60/60 for secs in df['secs_elapsed']]
df['mins_per_epoch']= [secs/60/epochs for secs,epochs in zip(df['secs_elapsed'],df['epoch'])]
df['f1'] = [f1*100 for f1 in df['f1']]
df = df.sort_values('f1',ascending=False)

# outputting results
if isinstance(args.csv,str):
    df[columns].to_csv(args.csv, float_format='%.1f')
elif args.csv is True:
    print(df[columns].to_csv(float_format='%.1f'))
# TODO barcharts!
else:
    pd.set_option('expand_frame_repr', False)
    print(df[columns])
