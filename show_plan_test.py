import unittest

from RegressionFramework.Common.dotDrawer import PlanDotDrawer
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.config import base_path


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_show_plan_(self):
        file_path = base_path+"plan2"
        with open(file_path) as f:
            plan_str = f.readline()

        plan_object = PlanFactory.get_plan_instance("pg", plan_str)

        dot = PlanDotDrawer.get_plan_dot_str(plan_object)
        pass


if __name__ == '__main__':
    unittest.main()
