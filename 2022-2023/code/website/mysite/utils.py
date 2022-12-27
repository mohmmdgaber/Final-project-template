from pymongo import MongoClient


# client = MongoClient('mongodb+srv://writerid:WriterID123@website.vxaej.mongodb.net/test')
# db = client['writerid']
# userdb = db['users']

# def get_db_handle(db_name, host, port, username, password):
#     client = MongoClient(host=host,
#                          port=int(port),
#                          username="writerid",
#                          password="WriterID123"
#                         )
#     db_handle = client['writerid']
#     return db_handle, client
#
# def get_collection_handle(db_handle,collection_name):
#     return db_handle['users']