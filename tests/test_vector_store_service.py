import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unboxed_ai.lib import Services, VectorStoreService, create_engine_from_constants


class TestVectorStoreService(unittest.TestCase):
    def test_ingest_texts_returns_ids_from_store(self) -> None:
        mock_store = MagicMock()
        mock_store.add_texts.return_value = ["id1", "id2"]
        mock_engine = MagicMock()
        with patch.object(mock_engine, "connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            svc = VectorStoreService(store=mock_store, engine=mock_engine)
            ids = svc.ingest_texts(["a", "b"], metadatas=[{}, {}])

        self.assertEqual(ids, ["id1", "id2"])
        mock_store.add_texts.assert_called_once_with(
            ["a", "b"], metadatas=[{}, {}], ids=None
        )

    def test_ingest_documents_returns_ids_from_store(self) -> None:
        from langchain_core.documents import Document

        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["doc-id-1"]
        mock_engine = MagicMock()
        docs = [Document(page_content="x", metadata={})]
        svc = VectorStoreService(store=mock_store, engine=mock_engine)
        ids = svc.ingest_documents(docs)

        self.assertEqual(ids, ["doc-id-1"])
        mock_store.add_documents.assert_called_once_with(docs, ids=None)

    def test_get_retriever_returns_store_retriever(self) -> None:
        mock_retriever = MagicMock()
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_engine = MagicMock()
        svc = VectorStoreService(store=mock_store, engine=mock_engine)
        retriever = svc.get_retriever(k=3, search_type="similarity")

        self.assertIs(retriever, mock_retriever)
        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def test_similarity_search_delegates_to_store(self) -> None:
        from langchain_core.documents import Document

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="hit", metadata={}),
        ]
        mock_engine = MagicMock()
        svc = VectorStoreService(store=mock_store, engine=mock_engine)
        docs = svc.similarity_search("query", k=2)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "hit")
        mock_store.similarity_search.assert_called_once_with("query", k=2)

    def test_ensure_pgvector_ready_executes_extension_sql(self) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_connect = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_connect.return_value.__exit__.return_value = None
        mock_engine.connect.return_value = mock_connect.return_value

        mock_store = MagicMock()
        svc = VectorStoreService(store=mock_store, engine=mock_engine)
        svc.ensure_pgvector_ready()

        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called_once()


class TestCreateEngineFromConstants(unittest.TestCase):
    @patch("unboxed_ai.lib.vector_store.Constants")
    @patch("sqlalchemy.create_engine")
    def test_create_engine_uses_constants_connection(
        self, mock_create_engine: Mock, mock_constants: Mock
    ) -> None:
        mock_constants.PGVECTOR_CONNECTION = "postgresql+psycopg://local/db"
        create_engine_from_constants()
        mock_create_engine.assert_called_once()
        call_kw = mock_create_engine.call_args[1]
        self.assertEqual(call_kw["pool_pre_ping"], True)
        self.assertEqual(
            mock_create_engine.call_args[0][0], "postgresql+psycopg://local/db"
        )


class TestServicesVectorStoreWiring(unittest.TestCase):
    @patch("unboxed_ai.lib.Services.create_engine_from_constants")
    def test_sqlalchemy_engine_calls_create_engine_from_constants(
        self, mock_create_engine: Mock
    ) -> None:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        with patch.object(Services, "pgvector", new_callable=PropertyMock) as mock_pg:
            mock_pg.return_value = MagicMock()
            services = Services()
            _ = services.sqlalchemy_engine
        mock_create_engine.assert_called_once()

    @patch("unboxed_ai.lib.Services.create_engine_from_constants")
    def test_vector_store_returns_vector_store_service(
        self, mock_create_engine: Mock
    ) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_create_engine.return_value = mock_engine
        with patch.object(Services, "pgvector", new_callable=PropertyMock) as mock_pg:
            mock_pg.return_value = mock_store
            services = Services()
            vs = services.vector_store
        self.assertIsInstance(vs, VectorStoreService)
        self.assertIs(vs._store, mock_store)
        self.assertIs(vs._engine, mock_engine)
