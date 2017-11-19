import csv

inputname = 'songdata.csv'
outputname = 'songdata_new.csv'
with open(inputname) as inputfile, open(outputname, 'w') as outputfile:
    r = csv.reader(inputfile)
    for row in r:
        song = row[3].strip()
        if 'the ' not in song or 'my ' not in song:
            continue
        print(row[3], file=outputfile)
