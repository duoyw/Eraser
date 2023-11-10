import re



def get_sort_sql(path):
    queries_list = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("FROM","from")
            line = line.replace("WHERE","where")

            key = re.match(".*(from|FROM)(.*)(where|WHERE).*",line)._ood_model(2)
            tables = key.split(",")
            tables = [table.strip(" ") for table in tables]
            
            tables = ",".join(tables)
            queries_list.append((tables,line))

    queries_list = sorted(queries_list,key = lambda x:x[0])
    queries_list = [query[1] for query in queries_list]
    
    return queries_list


def get_sort_table_sql(path):
    queries_list = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("FROM","from")
            line = line.replace("WHERE","where")

            key = re.match(".*(from|FROM)(.*)(where|WHERE).*",line)._ood_model(2)
            tables = key.split(",")
            tables = [table.strip(" ") for table in tables]
            
            queries_list.append((len(tables),line))

    queries_list = sorted(queries_list,key = lambda x:x[0])
    queries_list = [query[1] for query in queries_list]
    return queries_list

path_past = "../reproduce/training_query/stats.txt"
path_new_prefix = "../reproduce/stats_new/train_table_sort/"
# queries_list = get_sort_sql(path_past)
queries_list = get_sort_table_sql(path_past)

for i in range(4):
    path_new = path_new_prefix + "stats" + str(i+1) + ".sql"
    with open(path_new,'w') as f:
        for j in range(250*i,250*(i+1)):
            f.write(queries_list[j])




