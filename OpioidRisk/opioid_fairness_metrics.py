"""
This file defines several fairness metrics for opioid prescribing case study
"""

import numpy as np
from sklearn.metrics import confusion_matrix

class AverageRiskScore():
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return "Average Risk Score: "+str(self.threshold)

    def measure(self, env, agent):
        all_patients = env.state.params.patient_pool
        opioid_model = agent.opioid_model

        patients_g0 = all_patients[all_patients['gender'] == 0] # male
        patients_g1 = all_patients[all_patients['gender'] == 1] # female
        score_g0 = opioid_model.predict(patients_g0.drop(['adverse_flag'], axis=1))
        score_g1 = opioid_model.predict(patients_g1.drop(['adverse_flag'], axis=1))
        avg_scores = (np.mean(score_g0), np.mean(score_g1))
        del patients_g0, patients_g1, score_g0, score_g1
        return avg_scores

    def is_satisfied(self, opioid_history, t):
        avg_scores = opioid_history.get_history_at_time(t)['avg_score']
        score_g0, score_g1 = avg_scores
        if abs(score_g0-score_g1) > self.threshold:
            return False, abs(score_g0-score_g1)
        else:
            return True, abs(score_g0-score_g1)
        

class OpioidUnfairnessScore():
    def __init__(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return "Unfairness Score metric: "+str(self.threshold)

    def measure(self, env, agent, decision_threshold):
        # measure average risk score, tpr, fpr, precision, f1, accuracy
        all_patients = env.state.params.patient_pool
        opioid_model = agent.opioid_model

        score_all = opioid_model.predict(all_patients.drop(['adverse_flag'], axis=1))

        patients_g0 = all_patients[all_patients['gender'] == 0] # male
        patients_g1 = all_patients[all_patients['gender'] == 1] # female
        score_g0 = opioid_model.predict(patients_g0.drop(['adverse_flag'], axis=1))
        score_g1 = opioid_model.predict(patients_g1.drop(['adverse_flag'], axis=1))
        avg_scores = (np.mean(score_all), np.mean(score_g0), np.mean(score_g1))

        # get the predicted labels
        predicted_labels_all = score_all >= decision_threshold
        predicted_labels_g0 = score_g0 >= decision_threshold
        predicted_labels_g1 = score_g1 >= decision_threshold

        # get recall, precision, f1, accuracy
        tn, fp, fn, tp = confusion_matrix(all_patients['adverse_flag'], predicted_labels_all).ravel()
        tn_g0, fp_g0, fn_g0, tp_g0 = confusion_matrix(patients_g0['adverse_flag'], predicted_labels_g0).ravel()
        tn_g1, fp_g1, fn_g1, tp_g1 = confusion_matrix(patients_g1['adverse_flag'], predicted_labels_g1).ravel()
        tpr = tp / (tp + fn)
        tpr_g0 = tp_g0 / (tp_g0 + fn_g0)
        tpr_g1 = tp_g1 / (tp_g1 + fn_g1)
        fpr = fp / (fp + tn)
        fpr_g0 = fp_g0 / (fp_g0 + tn_g0)
        fpr_g1 = fp_g1 / (fp_g1 + tn_g1)
        precision = tp / (tp + fp)
        precision_g0 = tp_g0 / (tp_g0 + fp_g0)
        precision_g1 = tp_g1 / (tp_g1 + fp_g1)
        f1 = 2*precision*tpr / (precision+tpr)
        f1_g0 = 2*precision_g0*tpr_g0 / (precision_g0+tpr_g0)
        f1_g1 = 2*precision_g1*tpr_g1 / (precision_g1+tpr_g1)
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        accuracy_g0 = (tp_g0+tn_g0) / (tp_g0+tn_g0+fp_g0+fn_g0)
        accuracy_g1 = (tp_g1+tn_g1) / (tp_g1+tn_g1+fp_g1+fn_g1)

        return avg_scores, (tpr, tpr_g0, tpr_g1), (fpr, fpr_g0, fpr_g1), \
            (precision, precision_g0, precision_g1), (f1, f1_g0, f1_g1), \
                (accuracy, accuracy_g0, accuracy_g1)

    def is_satisfied(self, opioid_history, t):
        avg_scores = opioid_history.get_history_at_time(t)['avg_score']
        _, score_g0, score_g1 = avg_scores
        if abs(score_g0-score_g1) > self.threshold:
            return False, abs(score_g0-score_g1)
        else:
            return True, abs(score_g0-score_g1)
        
    def get_unfairness_score(self, opioid_history, t):
        # return gap in risk score, tpr, fpr, precision, f1, accuracy
        history = opioid_history.get_history_at_time(t)

        avg_scores = history['avg_score']
        _, score_g0, score_g1 = avg_scores
        gap_score = abs(score_g0-score_g1)

        tpr = history['tpr']
        _, tpr_g0, tpr_g1 = tpr
        gap_tpr = abs(tpr_g0-tpr_g1)

        fpr = history['fpr']
        _, fpr_g0, fpr_g1 = fpr
        gap_fpr = abs(fpr_g0-fpr_g1)

        precision = history['precision']
        _, precision_g0, precision_g1 = precision
        gap_precision = abs(precision_g0-precision_g1)

        f1 = history['f1']
        _, f1_g0, f1_g1 = f1
        gap_f1 = abs(f1_g0-f1_g1)

        accuracy = history['accuracy']
        _, accuracy_g0, accuracy_g1 = accuracy
        gap_accuracy = abs(accuracy_g0-accuracy_g1)

        return gap_score, gap_tpr, gap_fpr, gap_precision, gap_f1, gap_accuracy


class OpioidUtility():
    def __str__(self):
        return "Opioid utility metric"

    def get_utility_score(self, opioid_history, t):
        # return all populations tpr, fpr, precision, f1, accuracy
        history = opioid_history.get_history_at_time(t)
        tpr = history['tpr'][0]
        fpr = history['fpr'][0]
        precision = history['precision'][0]
        f1 = history['f1'][0]
        accuracy = history['accuracy'][0]

        return tpr, fpr, precision, f1, accuracy


