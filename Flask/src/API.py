import time
import requests
#import pymongo
class EventAPIClient1:
    """Realtime Events API Client"""
    def __init__(self, first_sequence_number=0,
                 api_url='https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/',
                 api_key='vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC',
                 db=None,
                 interval=30):
        """Initialize the API client."""
        self.next_sequence_number = first_sequence_number
        self.api_url = api_url
        self.api_key = api_key
        self.db = db
        self.interval = 30
    def save_to_database(self, row):
        """Save a data row to the database."""
        print("Received data:\n" + repr(row) + "\n")  # replace this with your code
        return row
    def get_data(self):
        """Fetch data from the API."""
        payload = {'api_key': self.api_key,
                   'sequence_number': self.next_sequence_number}
        response = requests.post(self.api_url, json=payload)
        data = response.json()
        self.next_sequence_number = data['_next_sequence_number']
        return data['data']
    def collect(self):
        """Check for new data from the API periodically."""
        print("Requesting data...")
        data = self.get_data()
        if data:
            print("Saving...")
            for row in data:
                self.save_to_database(row)
        else:
            print("No new data received.")
        return data