import csv

inputname = 'songdata.csv'
outputname = 'songdata_new.csv'
with open(inputname) as inputfile, open(outputname, 'w') as outputfile:
    r = csv.reader(inputfile)
    for row in r:
        print(row[3], file=outputfile)
    r.close()
