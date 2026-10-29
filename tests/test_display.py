import unittest

import pandas as pd
import pyarrow as pa

from ml.display import display_table


class DisplayTests(unittest.TestCase):
    def test_large_utf8_columns_and_index_are_normalized(self):
        frame = pd.DataFrame({'text': pd.Series(['example', None], dtype='string[pyarrow]'),
                              'number': [1, 2]})
        frame.index = pd.Index(['train', 'test'], dtype='str')
        table = display_table(frame)
        for field in table.schema:
            self.assertFalse(pa.types.is_large_string(field.type))
            self.assertFalse(pa.types.is_large_binary(field.type))
        self.assertEqual(table['text'].to_pylist(), ['example', None])
        self.assertEqual(table['number'].to_pylist(), [1, 2])
        self.assertEqual(table['__index_level_0__'].to_pylist(), ['train', 'test'])

    def test_browser_payloads_use_supported_arrow_types(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.check_frames(app)

    def check_frames(self, app):
        self.assertGreater(len(app.dataframe), 0)
        for item in app.dataframe:
            table = pa.ipc.open_stream(item.proto.data).read_all()
            for field in table.schema:
                self.assertFalse(pa.types.is_large_string(field.type), str(field))
                self.assertFalse(pa.types.is_large_binary(field.type), str(field))
