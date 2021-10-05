/* eslint-disable @typescript-eslint/naming-convention */
export interface Metrics {
  accuracy: number;
  balanced_accuracy: number;
  confusion_matrix?: number[][];
  f1_macro: number;
  f1_micro: number;
  f1_weighted: number;
  hamming_loss: number;
  jaccard_macro: number;
  jaccard_micro: number;
  jaccard_weighted: number;
  precision_macro: number;
  precision_micro: number;
  precision_weighted: number;
  recall_macro: number;
  recall_micro: number;
  recall_weighted: number;
  roc_auc_ovo: number;
  roc_auc_ovo_weighted: number;
  roc_auc_ovr: number;
  roc_auc_ovr_weighted: number;
}
