def get_binary_metrics() -> list:
    return ['accuracy', 'balanced_accuracy', 'average_precision', 'neg_brier_score', 'f1', 'neg_log_loss', 'precision',
            'recall', 'jaccard', 'roc_auc', 'confusion_matrix']


def get_multiclass_metrics() -> list:
    return ['accuracy', 'balanced_accuracy', 'precision_micro', 'precision_macro', 'precision_weighted', 'recall_micro',
            'recall_macro', 'recall_weighted', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'f1_micro',
            'f1_macro', 'f1_weighted', 'confusion_matrix', 'hamming_loss', 'roc_auc_ovr', 'roc_auc_ovo',
            'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']


def get_custom_assignation_metrics() -> list:
    return ['accuracy', 'balanced_accuracy', 'precision_micro', 'precision_macro', 'precision_weighted', 'recall_micro',
            'recall_macro', 'recall_weighted', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'f1_micro',
            'f1_macro', 'f1_weighted', 'confusion_matrix', 'hamming_loss']


def get_metrics_by_task(task: str = '') -> list:
    metrics = {
        'prioritization': get_multiclass_metrics(),
        'classification': get_multiclass_metrics(),
        'duplicity': get_binary_metrics(),
        'assignation': get_multiclass_metrics(),
        'custom_assignation': get_custom_assignation_metrics(),
        'similarity': get_binary_metrics()
    }
    return metrics[task]
