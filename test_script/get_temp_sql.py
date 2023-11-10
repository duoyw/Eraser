
# 3 5 7 8 9 10

hash_map1 = {"select   l_orderkey":0, "select   n_name":1, "select   supp_nation":2,
            "select   o_year":3, "select   nation":4, "select   c_custkey":5}
hash_map2 = {0:3, 1:5, 2:7, 3:8, 4:9, 5:10}
query_file = "../reproduce/test_query/tpch.txt"
new_file_name_prefix = "../reproduce/tpch_temp/test/"

# read original queries
queries = []
with open(query_file, 'r') as f:
    for line in f.readlines():
        arr = line.strip().split("#####")[1]
        queries.append(arr)

# handle the queries
queries_all = [[] for _ in range(6)]
for query in queries:
    for key in hash_map1.keys():
        if key in query:
            queries_all[hash_map1[key]].append(query)

# store the queries
for i in range(6):
    new_file_name_name = new_file_name_prefix + 'q' + str(hash_map2[i]) + '.sql'
    with open(new_file_name_name, 'w') as f:
        for j in range(len(queries_all[i])):
            write_line = 'q' + str(j) + "####" + queries_all[i][j] + '\n'
            f.write(write_line)



    