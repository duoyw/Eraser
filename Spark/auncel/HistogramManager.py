class HistogramManager:
    def __init__(self, file_path):
        self.col2Info = {}
        self.buckets = None
        self.load(file_path)

    def load(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                table, column, boundary, bar_ele_count = line.split("#####")
                if "date" in column.lower():
                    boundary = [float(x)/1000000 for x in boundary[1:-1].split(",")]
                else:
                    boundary = [float(x) for x in boundary[1:-1].split(",")]
                table_col = table + "." + column
                self.col2Info[table_col] = (table, column, boundary, float(bar_ele_count.strip("\n")))
                self.buckets = len(boundary)

    def get_bins(self, table_col):
        return self.col2Info[table_col][2]

    def get_col_min_max_vals(self):
        table_col_min_max = {}
        for table_col, info in self.col2Info.items():
            min_val = info[2][0]
            max_val = info[2][-1]
            table_col_min_max[table_col] = (min_val, max_val)
        return table_col_min_max
