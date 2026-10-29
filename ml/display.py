"""Arrow payloads compatible with Streamlit's older browser decoder."""
import pandas as pd
import pyarrow as pa


def display_table(data):
    """Downcast large string/binary columns, including a string row index.

    Pandas 3 can infer Arrow LargeUtf8, which Streamlit 1.28's bundled
    JavaScript decoder cannot read. These dashboard tables are small enough
    for Arrow's ordinary 32-bit string offsets.
    """
    table = pa.Table.from_pandas(pd.DataFrame(data), preserve_index=True)
    fields = []
    for field in table.schema:
        if pa.types.is_large_string(field.type):
            field = field.with_type(pa.string())
        elif pa.types.is_large_binary(field.type):
            field = field.with_type(pa.binary())
        fields.append(field)
    return table.cast(pa.schema(fields, metadata=table.schema.metadata))
