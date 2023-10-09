from typing import Dict, List, Optional, Set
import firebase_admin
from firebase_admin import App, firestore, credentials
from google.cloud.firestore_v1.client import Client
from google.cloud.firestore_v1.collection import CollectionReference
from google.cloud.firestore_v1.document import DocumentReference, DocumentSnapshot
from google.cloud.firestore_v1.query import Query
from google.cloud.firestore_v1.types import WriteResult
from google.cloud.firestore_v1.batch import WriteBatch


def initialize_firestore() -> Client:
    firebase_app: App = firebase_admin.initialize_app()
    db: Client = firestore.client(app=firebase_app)
    return db


class FirestoreAccess:
    firebase_app: App
    db: Client

    def connect(self, credential: Optional[credentials.Certificate] = None):
        try:
            self.firebase_app = firebase_admin.get_app()
        except ValueError:
            self.firebase_app = firebase_admin.initialize_app(credential=credential)

        self.db = firestore.client(app=self.firebase_app)

    def close(self):
        self.db.close()

    def get_houses_collection(self) -> CollectionReference:
        return self.db.collection("houses")

    def get_house_document(self, house_id: str) -> DocumentReference:
        houses_collection: CollectionReference = self.get_houses_collection()
        return houses_collection.document(house_id)

    def get_house_snapshots(self, house_ids: List[str]) -> List[DocumentSnapshot]:
        house_snapshots: List[DocumentSnapshot] = []

        # case when there are more than 30 house_ids (not supported by `in` operator)
        if len(house_ids) > 30:
            beyond_limit: List[str] = house_ids[30:]
            house_snapshots.extend(self.get_house_snapshots(beyond_limit))
            house_ids = house_ids[:30]

        # retrieve house snapshots
        houses_collection: CollectionReference = self.get_houses_collection()
        house_snapshots_query: Query = houses_collection.where("id", "in", house_ids)
        house_snapshots_query_result = house_snapshots_query.get()
        house_snapshots.extend(house_snapshots_query_result)

        return house_snapshots

    def get_non_existant_houses_ids(self, house_ids: List[str]) -> List[str]:
        house_ids_set: Set[str] = set(house_ids)
        house_snapshots: List[DocumentSnapshot] = self.get_house_snapshots(house_ids)
        existant_houses_set: Set[str] = set([house.id for house in house_snapshots])
        non_existant_houses = house_ids_set - existant_houses_set
        return list(non_existant_houses)

    def get_non_existant_houses(self, houses: List[Dict]) -> List[Dict]:
        house_ids: List[str] = [house["id"] for house in houses]
        non_existant_house_ids: List[str] = self.get_non_existant_houses_ids(house_ids)
        non_existant_houses: List[Dict] = [
            house for house in houses if house["id"] in non_existant_house_ids
        ]
        return non_existant_houses

    def add_houses(self, houses: List[Dict]) -> List[WriteResult]:
        houses_collection: CollectionReference = self.get_houses_collection()
        batch: WriteBatch = self.db.batch()
        for house in houses:
            doc_ref: DocumentReference = houses_collection.document(house["id"])
            batch.set(doc_ref, house)

        write_results: List[WriteResult] = batch.commit()
        return write_results
