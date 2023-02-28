import argparse

from utils import *

# python test.py --query_path ../reproduce/test_query/stats.txt --output_query_latency_file stats.test
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",default = "lero/reproduce/test_query/stats.txt",
                        help="Load the queries")
    parser.add_argument("--output_query_latency_file", metavar="PATH",default = "lero/test_script/stats.test")

    args = parser.parse_args()
    test_queries = []
    # print(os.getcwd())
    with open(args.query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            test_queries.append((arr[0], arr[1]))
    print("Read", len(test_queries), "test queries.")
    # q0#####SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId= u.Id AND u.UpVotes>=0;
    for (fp, q) in test_queries:
        do_run_query(q, fp, ["SET enable_lero TO True"], args.output_query_latency_file, True, None, None)
