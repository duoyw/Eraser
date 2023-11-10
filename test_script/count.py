queries = []
with open("pg_stats_test.log", 'r') as f:
    for line in f.readlines():
        arr = line.strip().split("####")
        number = arr[0][1:]
        queries.append(int(number))
queries = sorted(queries)
for i in range(len(queries)):
    if i not in queries:
        print(i)