import re



def get_sql(path):
    queries_list = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            queries_list.append(line)

    return queries_list


path_past = "../reproduce/training_query/job.txt"
path_new_prefix = "../reproduce/job_new/train/"
# queries_list = get_sort_sql(path_past)
queries_list = get_sql(path_past)

for i in range(4):
    path_new = path_new_prefix + "job" + str(i+1) + ".sql"
    with open(path_new,'w') as f:
        for j in range(250*i,250*(i+1)):
            f.write(queries_list[j])




