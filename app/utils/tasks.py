def get_task_type_text(task: str = 'prioritization', lang: str = 'es') -> str:
    text = {
        'prioritization': {
            'es': 'Priorización de incidencias'
        },
        'classification': {
            'es': 'Clasificación de incidencias'
        },
        'duplicity': {
            'es': 'Detección de incidencias duplicadas'
        },
        'assignation': {
            'es': 'Asignación de incidencias (basada en clasificación)'
        },
        'custom_assignation': {
            'es': 'Asignación de incidencias (basada en carga de trabajo y adecuación del desarrollador)'
        },
        'similarity': {
            'es': 'Recuperación de incidencias similares'
        }
    }
    return text[task][lang]


def get_tasks_by_corpus(corpus: str) -> list:
    tasks = {
        'bugzilla': ['prioritization', 'classification', 'duplicity', 'assignation', 'custom_assignation',
                     'similarity'],
        'eclipse': ['prioritization', 'classification', 'duplicity'],
        'netBeans': ['prioritization', 'classification', 'duplicity'],
        'openOffice': ['prioritization', 'classification', 'duplicity']
    }
    return tasks[corpus]
