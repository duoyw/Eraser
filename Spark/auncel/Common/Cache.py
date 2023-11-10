import json
import os

import joblib

from model_config import CACHE_FILE_PATH


class Cache:
    def __init__(self, name, directory, enable=True):
        super().__init__()
        self.directory = directory
        self.enable = enable
        self.name = name.split("/")[-1]

    def save(self, target):
        if self.enable:
            file = self._get_absolute_path()
            key = self.get_identifier()
            joblib.dump([key, target], file)

    def read(self):
        if not self.enable:
            raise RuntimeError
        file = self._get_absolute_path()
        if not self.exist():
            raise RuntimeError("File do not exist")
        res = joblib.load(file)
        if res[0] != self.get_identifier():
            raise RuntimeError("identical file name but identifier is not same")
        return res[1]

    def exist(self):
        if not self.enable:
            return False
        file = self._get_absolute_path()
        if not os.path.exists(file):
            return False
        return True

    def get_file_name(self):
        return self.name

    def get_identifier(self):
        return "empty"

    def _get_absolute_path(self):
        return os.path.join(self.directory, self.get_file_name())


class ParsePlanCache(Cache):
    def __init__(self, name, is_compress, is_correct_card, directory=CACHE_FILE_PATH, enable=True):
        super().__init__(name, directory, enable)
        self.is_compress = is_compress
        self.is_correct_card = is_correct_card

    def get_file_name(self):
        return super().get_file_name() + "_PlanCache"

    def get_identifier(self):
        return "is_compress={}, is_correct_card={}".format(self.is_compress, self.is_correct_card)


class PredictTimeCache(Cache):
    def __init__(self, name, model_name, directory=CACHE_FILE_PATH, enable=True):
        super().__init__(name, directory, enable)
        self.model_name = model_name

    def get_file_name(self):
        return super().get_file_name() + "_{}_predict_cache".format(self.model_name)


class AdaptiveGroupCache(Cache):
    def __init__(self, model_name, dataset_name, struct_enable, scan_type_enable, table_name_enable, join_type_enable,
                 join_key_enable, filter_enable, filter_col_enable, filter_op_enable, filter_value_enable,
                 directory=CACHE_FILE_PATH, enable=True):
        super().__init__("", directory, enable)
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.struct_enable = struct_enable

        self.scan_type_enable = scan_type_enable
        self.table_name_enable = table_name_enable

        self.join_type_enable = join_type_enable
        self.join_key_enable = join_key_enable

        self.filter_enable = filter_enable
        self.filter_col_enable = filter_col_enable
        self.filter_op_enable = filter_op_enable
        self.filter_value_enable = filter_value_enable

    def get_file_name(self):
        return super().get_file_name() + "_{}_{}_{}_adaptive_group_cache".format(self.model_name, self.dataset_name,
                                                                                 self.get_enable_key())

    def get_enable_key(self):
        key = []
        key.append("T" if self.struct_enable else "F")
        key.append("T" if self.scan_type_enable else "F")
        key.append("T" if self.table_name_enable else "F")
        key.append("T" if self.join_type_enable else "F")
        key.append("T" if self.join_key_enable else "F")
        key.append("T" if self.filter_enable else "F")
        key.append("T" if self.filter_col_enable else "F")
        key.append("T" if self.filter_op_enable else "F")
        key.append("T" if self.filter_value_enable else "F")
        return "_".join(key)

    def get_identifier(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.struct_enable, self.scan_type_enable, self.table_name_enable,
                                                   self.join_type_enable, self.join_key_enable, self.filter_enable,
                                                   self.filter_col_enable,
                                                   self.filter_op_enable, self.filter_value_enable)
