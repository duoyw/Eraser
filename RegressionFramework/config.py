from RegressionFramework.Plan.PlanConfig import PgNodeConfig

# data filepath
base_path = "./"
data_base_path = "./data/"
cache_base_path = data_base_path + "cache/"
model_base_path = "./model/"

# configure need to be adjusted by validation set
# we adjust alpha and beta to control lambda
alpha = 0.6
# we will show the result of each beta, list format is often used for validation set
betas = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
min_leaf_ele_count = 5
# the decision if segment model have not a corresponding structure for the incoming plan,
# -1 mean to refuse risk model, but 1 mean to trust risk model
decision_for_new_structure = -1


# configure not to change
db_node_config = PgNodeConfig
max_col_limit = 20
shifted_sql_budget_ratio = 0.5
overlap_thres = 0.99
max_join_keys_size = 300
min_join_keys_size = 200
sqls_for_each_join_key = 3
histogram_bin_size = 10
sqls_for_each_bin = 1
ignore_node_type = ["Hash", "Sort", "Bitmap Index Scan", "Aggregate", "Limit"]
