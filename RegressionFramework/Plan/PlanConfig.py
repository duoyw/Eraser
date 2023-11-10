class SparkNodeConfig:
    SCAN_TYPES = ["org.apache.spark.sql.execution.FileSourceScanExec"]
    JOIN_TYPES = ["org.apache.spark.sql.execution.joins.BroadcastHashJoinExec",
                  "org.apache.spark.sql.execution.joins.SortMergeJoinExec",
                  "org.apache.spark.sql.execution.joins.BroadcastNestedLoopJoinExec"]
    FILTER_TYPES = ["org.apache.spark.sql.execution.FilterExec"]
    PROJECT_TYPES = 'org.apache.spark.sql.execution.ProjectExec'
    BroadcastNestedLoopJoinType = "org.apache.spark.sql.execution.joins.BroadcastNestedLoopJoinExec"

    # expression type
    SUBSTRING_TYPE = 'org.apache.spark.sql.catalyst.expressions.Substring'
    LITERAL_TYPE = "org.apache.spark.sql.catalyst.expressions.Literal"
    IN_TYPE = "org.apache.spark.sql.catalyst.expressions.In"
    INSET_TYPE = 'org.apache.spark.sql.catalyst.expressions.InSet'
    ATTRIBUTE_REFERENCE_TYPE = "org.apache.spark.sql.catalyst.expressions.AttributeReference"
    COALESCE_TYPE = "org.apache.spark.sql.catalyst.expressions.Coalesce"
    iS_NULL_TYPE = "org.apache.spark.sql.catalyst.expressions.IsNull"
    IN_SET_TYPE = "org.apache.spark.sql.catalyst.expressions.InSet"

    OPERATOR_TYPE = {
        "org.apache.spark.sql.catalyst.expressions.EqualTo": "=",
        "org.apache.spark.sql.catalyst.expressions.LessThan": "<",
        "org.apache.spark.sql.catalyst.expressions.LessThanOrEqual": "<",
        "org.apache.spark.sql.catalyst.expressions.GreaterThan": ">",
        "org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual": ">"
    }


class PgNodeConfig:
    SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
    JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
    UNKNOWN_OP_TYPE = "Unknown"
    OTHER_TYPES = ['Bitmap Index Scan']
    OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
               + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES

    FILTER_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
    PROJECT_TYPES = 'org.apache.spark.sql.execution.ProjectExec'
    BroadcastNestedLoopJoinType = "org.apache.spark.sql.execution.joins.BroadcastNestedLoopJoinExec"

    # expression type
    SUBSTRING_TYPE = 'org.apache.spark.sql.catalyst.expressions.Substring'
    LITERAL_TYPE = "org.apache.spark.sql.catalyst.expressions.Literal"
    IN_TYPE = "org.apache.spark.sql.catalyst.expressions.In"
    INSET_TYPE = 'org.apache.spark.sql.catalyst.expressions.InSet'
    ATTRIBUTE_REFERENCE_TYPE = "org.apache.spark.sql.catalyst.expressions.AttributeReference"
    COALESCE_TYPE = "org.apache.spark.sql.catalyst.expressions.Coalesce"
    iS_NULL_TYPE = "org.apache.spark.sql.catalyst.expressions.IsNull"
    IN_SET_TYPE = "org.apache.spark.sql.catalyst.expressions.InSet"

    OPERATOR_TYPE = {
        "org.apache.spark.sql.catalyst.expressions.EqualTo": "=",
        "org.apache.spark.sql.catalyst.expressions.LessThan": "<",
        "org.apache.spark.sql.catalyst.expressions.LessThanOrEqual": "<",
        "org.apache.spark.sql.catalyst.expressions.GreaterThan": ">",
        "org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual": ">"
    }
