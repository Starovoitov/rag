from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.server import app


class TestApiServer(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_build_parser_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "build_parser",
                "argv": ["build_parser", "--output", "data/rag.jsonl"],
                "stdout": "{}\n",
                "stderr": "",
                "result": {},
            }
            response = self.client.post(
                "/build_parser",
                json={"output": "data/rag.jsonl", "min_tokens": 100, "log_json": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "build_parser")

    def test_build_faiss_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "build_faiss",
                "argv": ["build_faiss"],
                "stdout": "{}\n",
                "stderr": "",
                "result": {},
            }
            response = self.client.post(
                "/build_faiss",
                json={"prepare_input": True, "rag_dataset": "data/rag_dataset.jsonl"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "build_faiss")

    def test_demo_retrieval_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "demo_retrieval",
                "argv": ["demo_retrieval"],
                "stdout": "",
                "stderr": "",
                "result": None,
            }
            response = self.client.post("/demo_retrieval", json={"query": "what is rag", "top_k": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "demo_retrieval")

    def test_evaluation_runner_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "evaluation_runner",
                "argv": ["evaluation_runner"],
                "stdout": "",
                "stderr": "",
                "result": None,
            }
            response = self.client.post(
                "/evaluation_runner",
                json={"dataset": "data/eval.jsonl", "retriever": "hybrid", "k_values": "1,3,5"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "evaluation_runner")

    def test_reranker_pipeline_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "reranker_pipeline",
                "argv": ["reranker_pipeline"],
                "stdout": "",
                "stderr": "",
                "result": None,
            }
            response = self.client.post(
                "/reranker_pipeline",
                json={"dataset": "data/eval.jsonl", "train_reranker": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "reranker_pipeline")

    def test_run_rag_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "run_rag",
                "argv": ["run_rag"],
                "stdout": "",
                "stderr": "",
                "result": None,
            }
            response = self.client.post(
                "/run_rag",
                json={"question": "What is RAG?", "provider": "openai", "top_k": 5},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "run_rag")

    def test_cleanup_faiss_endpoint(self) -> None:
        with patch("api.server.execute_cli_command") as execute:
            execute.return_value = {
                "command": "cleanup_faiss",
                "argv": ["cleanup_faiss"],
                "stdout": "{}\n",
                "stderr": "",
                "result": {},
            }
            response = self.client.post(
                "/cleanup_faiss",
                json={"faiss_path": "data/faiss", "drop_persist_directory": True},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["command"], "cleanup_faiss")

    def test_openapi_has_config_backed_example_for_evaluation_runner(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        doc = response.json()
        schema_ref = doc["paths"]["/evaluation_runner"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        schema_name = schema_ref.split("/")[-1]
        schema = doc["components"]["schemas"][schema_name]
        self.assertEqual(
            schema["properties"]["dataset"]["default"], "data/evaluation_with_evidence.jsonl"
        )
        self.assertEqual(schema["properties"]["k_values"]["default"], "1,3,5")
        self.assertEqual(schema["properties"]["retriever"]["default"], "semantic")


if __name__ == "__main__":
    unittest.main()

