from Common.Batch import Batch
from Common.PlanFactory import PlanFactory
from UncertantyModel.UncertaintyEstimate import ConfidenceEstimate
from model import AuncelModel, AuncelModelPairConfidenceWise
from model_config import db_type
from utils import json_str_to_json_obj, cal_accuracy


class SimilarModelEstimate(ConfidenceEstimate):
    def __init__(self, plans, predicts, dataset, confidence_model: AuncelModel):
        if isinstance(plans[0], str):
            self.plans = [json_str_to_json_obj(p) for p in plans]
        self.plans = [PlanFactory.get_plan_instance(db_type, self.plans[i], i, predicts[i]) for i in range(len(plans))]
        self.confidence_model = confidence_model
        self.dataset = dataset
        self.k = 20

    def estimate(self, plan, predict):
        model: AuncelModelPairConfidenceWise = self.confidence_model
        buff = []
        i = 0
        plan_batch = Batch(self.plans, batch_size=self.k * 2)
        while not plan_batch.is_end():
            if i % self.k * 4 == 0 and i >0:
                print("candidate_plan total count is {}, cur is {}".format(len(self.plans), i))
            plans = plan_batch.next()
            target_plan_feature = model.to_feature([plan], self.dataset)
            candidate_plan_feature = model.to_feature([p.plan_json for p in plans], self.dataset)
            similar_res = model.predict_pair(target_plan_feature * len(candidate_plan_feature), candidate_plan_feature)
            for i in range(len(similar_res)):
                if similar_res[i] == 1:
                    buff.append(plans[i])
            if len(buff) > self.k:
                break
            i += len(plans)

        buff = buff[0:self.k + 2]
        accuracies = [cal_accuracy(p.predict, p.execution_time) for p in buff]
        accuracies = sorted(accuracies)[1:-1]
        cur_accuracy = cal_accuracy(predict, float(json_str_to_json_obj(plan)["Execution Time"]))
        mean_accuracy = sum(accuracies) / len(accuracies)
        print("cur plan accuracy is {}, mean accuracy is {}".format(cur_accuracy, mean_accuracy))
        return mean_accuracy, predict
