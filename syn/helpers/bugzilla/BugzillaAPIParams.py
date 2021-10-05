class BugzillaAPIParams:

    def __init__(
            self,
            project="eclipse",
            year=2001,
            start_month=1,
            end_month=12,
            query_limit=0,
            include_fields=None,
            get_comments=True
    ):
        self.project = project
        self.year = year
        self.start_month = start_month
        self.end_month = end_month
        self.query_limit = query_limit
        self.include_fields = include_fields
        self.get_comments = get_comments
