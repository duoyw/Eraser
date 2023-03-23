from RegressionFramework.Plan.PlanConfig import PgNodeConfig

base_path = "/mnt/yuchu.yc/Ai4Db/"
data_base_path = "/mnt/yuchu.yc/Ai4Db/data/"
cache_base_path = "/mnt/yuchu.yc/Ai4Db/cache/"
model_base_path = "/mnt/yuchu.yc/Ai4Db/model/"
db_node_config = PgNodeConfig


class GroupEnable:
    struct_enable = True
    scan_type_enable = False
    table_name_enable = False
    join_type_enable = False
    join_key_enable = False
    filter_enable = False
    filter_col_enable = False
    filter_op_enable = False
    filter_value_enable = False


max_col_limit = 20

# shifted plan parameter
shifted_sql_budget_ratio = 0.5
# join order, candidate join order, join key filter condition
overlap_thres = 0.99
max_join_keys_size = 300
min_join_keys_size = 200

# shifted join key production
sqls_for_each_join_key = 3

# shifted filter value
histogram_bin_size = 10
sqls_for_each_bin = 1

shifted_space_thres = 0.6


class DbConfig:
    def get_tables_counts(self):
        raise


class JobConfig(DbConfig):
    def __init__(self):
        self.alias_table = {

        }
