
file = open(r'C:\Users\Josh\Downloads\SampleTextFile_10kb.txt')
words = list(file.read().split())
print(tuple(words))
print(len(words))
print(tuple(words[0:2]))
print(tuple(words[2:4]))
print(tuple(words[4:6]))
print(tuple(words[6:8]))
print(tuple(words[8:10]))
print(tuple(words[10:12]))
print(tuple(words[12:14]))
print(tuple(words[14:16]))







