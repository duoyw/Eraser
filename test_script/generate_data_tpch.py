# train: 3 5; 3 5 7 8; 3 5 7 8 9 10
# test: 3 5 7 8 9 10


# training data
file_name_prefix = "../reproduce/tpch_new/"

train_list2 = [3,5,7,8]
train_list3  = [3,5,7,8,9,10]



train_list1  = [i for i in range(23)]
train_list1.remove(0)
train_list1.remove(15)
index = 0
new_file_name_name = file_name_prefix + 'qall.sql'
for i in range(len(train_list1)):
    file_name_name = file_name_prefix + 'd' + str(train_list1[i]) + '.sql'
    query_list = []
    str_ = ""
    flag = False
    with open(file_name_name, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n"," ")
            line = line.replace("\t"," ")
            if '--' in line :
                if flag:
                    query_list.append(str_)
                str_ = ""
            else:
                flag = True
                str_ += line
                
        query_list.append(str_)

    for  query in query_list:
        with open(new_file_name_name, 'a') as f_:
            write_line = 'q' + str(index) + "#####" + query + '\n'
            index += 1
            f_.write(write_line)
                


# for i in range(len(train_list1)):
#     file_name_name = file_name_prefix + 'd' + str(train_list1[i]) + '.sql'
#     with open(file_name_name, 'r') as f:
#         for line in f.readlines():

#             arr = line.strip().split("#####")[1]

#             with open(new_file_name_name, 'a') as f:
#                 write_line = 'q' + str(index) + "#####" + arr + '\n'
#                 index += 1
#                 f.write(write_line)



# index = 0
# new_file_name_name = file_name_prefix + 'q3578.txt'
# for i in range(len(train_list2)):
#     file_name_name = file_name_prefix + 'q' + str(train_list2[i]) + '.sql'
#     with open(file_name_name, 'r') as f:
#         for line in f.readlines():
#             arr = line.strip().split("#####")[1]

#             with open(new_file_name_name, 'a') as f:
#                 write_line = 'q' + str(index) + "#####" + arr + '\n'
#                 index += 1
#                 f.write(write_line)

# index = 0
# new_file_name_name = file_name_prefix + 'q3578910.txt'
# for i in range(len(train_list3)):
#     file_name_name = file_name_prefix + 'q' + str(train_list3[i]) + '.sql'
#     with open(file_name_name, 'r') as f:
#         for line in f.readlines():
#             arr = line.strip().split("#####")[1]

#             with open(new_file_name_name, 'a') as f:
#                 write_line = 'q' + str(index) + "#####" + arr + '\n'
#                 index += 1
#                 f.write(write_line)


# # testing data
# file_name_prefix = "../reproduce/tpch_temp/test/"
# test_list  = [3,5,7,8,9,10]
# index = 0
# new_file_name_name = file_name_prefix + 'q3578910.txt'
# for i in range(len(test_list)):
#     file_name_name = file_name_prefix + 'q' + str(test_list[i]) + '.sql'
#     with open(file_name_name, 'r') as f:
#         for line in f.readlines():
#             arr = line.strip().split("#####")[1]

#             with open(new_file_name_name, 'a') as f:
#                 write_line = 'q' + str(index) + "#####" + arr + '\n'
#                 index += 1
#                 f.write(write_line)
