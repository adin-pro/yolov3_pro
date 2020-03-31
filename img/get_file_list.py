import argparse
import os

parser = argparse.ArgumentParser(description = 'get file list in a directory')
parser.add_argument('--source', type = str, help = 'the absolute path of directory')
parser.add_argument('--dest', type = str, default= './' ,help = 'where output the listfile')
parser.add_argument('--output', type = str, default = 'list.txt', help= 'output filename')
args = parser.parse_args()
	
if args.source:
	print('The path is:'+args.source)
	print('Loading...')
	list = os.listdir(args.source)
	output = args.dest + args.output
	with open(output, mode='w') as fd:
		for item in list:
			fd.write('img/'+args.source+item+'\n')
	print('Success! The output is '+output)
else:
	print('Please type the path')

		


