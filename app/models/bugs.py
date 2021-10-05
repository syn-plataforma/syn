class Bug(object):
    def __init__(
            self,
            bug_id,
            product,
            bug_severity,
            priority,
            component,
            description
    ):
        self.bug_id = bug_id
        self.product = product
        self.bug_severity = bug_severity
        self.priority = priority
        self.component = component
        self.description = description


class TfidfBug(Bug):
    def __init__(
            self,
            bug_id,
            product,
            bug_severity,
            priority,
            component,
            description,
            description_tfidf
    ):
        super().__init__(bug_id, product, bug_severity, priority, component, description)
        self.description_tfidf = description_tfidf


class VectorizedBug(Bug):
    def __init__(
            self,
            bug_id,
            product,
            bug_severity,
            priority,
            component,
            description,
            product_ohe,
            bug_severity_ohe,
            priority_ohe,
            component_ohe,
            description_tfidf
    ):
        super().__init__(bug_id, product, bug_severity, priority, component, description)
        self.product_ohe = product_ohe
        self.bug_severity_ohe = bug_severity_ohe
        self.priority_ohe = priority_ohe
        self.component_ohe = component_ohe
        self.description_tfidf = description_tfidf
