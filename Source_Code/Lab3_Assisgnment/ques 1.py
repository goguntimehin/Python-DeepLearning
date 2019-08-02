
infile = open(r'C:\Users\Josh\Downloads\SampleTextFile_10kb.txt')
data = infile.read()
numOfChars = len(data)
numOfWords = len(data.split())
numOfLines = len(data.splitlines())
print(numOfChars,numOfWords,numOfLines)



