class TextNormalizerAPIParams:

    def __init__(
            self,
            env="local",
            batches=1,
            output_format="mongodb",
            columns_names=None,
            drop_original_columns=False,
            split_new_line=False,
            to_lower_case=True
    ):
        if columns_names is None:
            columns_names = ['short_desc', 'description']
        self.env = env
        self.batches = batches
        self.output_format = output_format
        self.columns_names = columns_names
        self.drop_original_columns = drop_original_columns
        self.split_new_line = split_new_line
        self.to_lower_case = to_lower_case
