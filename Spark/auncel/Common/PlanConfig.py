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