"""This script aggregates and presents summary results in output/*/*.best.txt"""

# imports
from subprocess import run, PIPE
from ast import literal_eval
import pandas as pd
import sys, argparse
import matplotlib.pyplot as plt

# final columns to show
index_col = 'name'
columns = ['f1','epoch','validation_loss','hours_elapsed','mins_per_epoch']

# parsing command line inputs to determine output
parser = argparse.ArgumentParser(description='List output/*.best.txt result stats to stdout')
parser.add_argument('--name', type=str, default='*', help='specify which <name>.best.txt summary files to search for in output/ . Uses wildcards. Default *')
parser.add_argument('--csv',  default=False, action='store_true', help='output as csv')
parser.add_argument('--plot', type=str, choices=['bar','box'], help='output a plot')
parser.add_argument('--columns', '--cols', nargs='*', help='limit output to just the specified columns. \ndefault f1 for bar plot, default perclass_f1 for box plot.')
parser.add_argument('--outfile','-o', type=str, help='file to output results to. plots are saved as PNGs')
args = parser.parse_args()

# collecting filenames/paths where data is stored
search_string = '{}.best.txt'.format(args.name)
cmd = ['find', 'output', '-type','f', '-name', search_string]
results = run(cmd,stdout=PIPE).stdout
results = results.decode().strip().split('\n')

# aggregating file content to DataFrame
df = pd.DataFrame(columns = columns)
df_perclass = pd.DataFrame()
for result in results:
    with open(result) as f:
        content = f.read()
        content = literal_eval(content)
        perclass_f1 = content.pop('perclass_f1', {})
        #try: 
        #    perclass_f1 = {'CLASS: '+key:val for key:val in perclass_f1.items()}
        #    content.update(perclass_f1)
        #except AttributeError: pass
        df = df.append(content, ignore_index=True)
        df_perclass = df_perclass.append(perclass_f1, ignore_index=True)

# processing some data columns
df = df.set_index(index_col)
df['hours_elapsed'] = [secs/60/60 for secs in df['secs_elapsed']]
df['mins_per_epoch']= [secs/60/epochs for secs,epochs in zip(df['secs_elapsed'],df['epoch'])]
df['f1'] = [f1*100 for f1 in df['f1']]
df = df.sort_values('f1',ascending=False)


## outputting results ##

# show all columns
if args.columns == []:
    print('Available columns are:', list(df.columns))
    
# show or print csv
elif args.csv:
    if args.outfile:
        df[columns].to_csv(args.outfile, float_format='%.1f')
    else:
        print(df[columns].to_csv(float_format='%.1f'))
        
# show or save bar or box plot
elif args.plot:
    if args.plot == 'bar' and args.columns is None:
        ax = df['f1'].plot(kind='barh')
        ax.set_xlabel('F1 percentage')
        ax.set_ylabel('')
    elif args.plot == 'bar' and args.columns == ['perclass_f1']:
        df_perclass.plot(kind='barh')
    elif args.plot == 'box' and args.columns == [] or args.columns == ['perclass_f1']:
        df_perclass.plot(kind='box',vert=False)
    else:
        df[args.columns].plot(kind=args.plot)

    # output plot: save or show
    if args.outfile:
        plt.savefig(args.outfile, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
        
# simply show the basics of the summary file
else:
    pd.set_option('expand_frame_repr', False)
    print(df[columns])
