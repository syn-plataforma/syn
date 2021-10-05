# import unittest
import unittest2 as unittest
from mockupdb import MockupDB, go, Command

from app import app


class BaseTestCase(unittest.TestCase):
    server = None

    @classmethod
    def setUpClass(self):
        self.server = MockupDB(auto_ismaster=True, verbose=True)
        self.server.run()
        # create mongo connection to mock server

        app.app.testing = True
        app.app.config['MONGO_URI'] = self.server.uri
        self.app = app.app.test_client()

    @classmethod
    def tearDownClass(self):
        self.server.stop()

    def test_getDataSources(self):
        # arrange
        future = go(self.app.get, '/dataSources')
        request = self.server.receives(
            Command({'find': 'dataSources', 'filter': {}}, flags=4, namespace='app'))
        request.ok(cursor={'id': 0, 'firstBatch': [
            {'name': 'Google', 'url': 'http://google.com/rest/api'},
            {'name': 'Rest', 'url': 'http://rest.com/rest/api'}]})

        # act
        http_response = future()

        # assert
        data = http_response.get_data(as_text=True)
        self.assertIn('http://google.com/rest/api', data)
        self.assertIn('http://rest.com/rest/api', data)
