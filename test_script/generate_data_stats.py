# train: 3 5; 3 5 7 8; 3 5 7 8 9 10
# test: 3 5 7 8 9 10


# training data
file = "../reproduce/training_query/stats.txt"
queries  = []
with open(file,'r') as f:
    lines = f.readlines()
    for line in lines:
        queries.append(line)


new_file1 = "../reproduce/stats_new/train/stats1.sql"
with open(new_file1,'a') as f:
    for i in range(0,250,1):
        f.write(queries[i])
        
new_file2 = "../reproduce/stats_new/train/stats2.sql"
with open(new_file2,'a') as f:
    for i in range(250,500,1):
        f.write(queries[i])
        
new_file3 = "../reproduce/stats_new/train/stats3.sql"
with open(new_file3,'a') as f:
    for i in range(500,750,1):
        f.write(queries[i])

new_file4 = "../reproduce/stats_new/train/stats4.sql"
with open(new_file4,'a') as f:
    for i in range(750,1000,1):
        f.write(queries[i])
        
        






