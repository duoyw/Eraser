from Plan import Plan, json_str_to_json_obj
from UncertantyModel.UncertaintyEstimate import ConfidenceEstimate
from model import AuncelModel
from utils import cal_accuracy


class SingleModelConfidenceEstimate(ConfidenceEstimate):
    def __init__(self, confidence_model: AuncelModel, dataset, train_set_name=None):
        self.model: AuncelModel = confidence_model
        self.dataset = dataset

    def estimate(self, plan, predict):
        model = self.model
        accuracy = model.predict_confidence(model.to_feature([plan], self.dataset)).item()
        cur_accuracy = cal_accuracy(predict, float(json_str_to_json_obj(plan)["Execution Time"]))
        print("cur_accuracy is {}, predict accuracy is {}".format(cur_accuracy, accuracy))
        return accuracy, predict
