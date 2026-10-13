import os
import sys
import certifi
import pymongo

from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY, MONGODB_URI_KEY
from us_visa.exception import USvisaException
from us_visa.logger import logging

ca = certifi.where()


class MongoDBClient:
    """
    MongoDB Client connection manager.
    Connects to MongoDB Atlas using TLS CA cert and environment credentials.
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY) or os.getenv(MONGODB_URI_KEY)

                # Fallback to local atlas-credentials.env or .env if present
                if not mongo_db_url:
                    env_files = ["atlas-credentials.env", ".env"]
                    for env_file in env_files:
                        if os.path.exists(env_file):
                            with open(env_file, "r") as f:
                                for line in f:
                                    line = line.strip()
                                    if line.startswith("MONGODB_URI="):
                                        mongo_db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                                        break
                                    elif line.startswith("MONGODB_URL="):
                                        mongo_db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                                        break
                        if mongo_db_url:
                            break

                if mongo_db_url is None:
                    raise Exception(f"MongoDB URL environment variable not set and credentials file not found.")

                logging.info("Connecting to MongoDB Atlas...")
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca,
                    serverSelectionTimeoutMS=5000,
                )
                # Ping database to verify connection
                MongoDBClient.client.admin.command("ping")

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection established successfully.")
        except Exception as e:
            raise USvisaException(e, sys)