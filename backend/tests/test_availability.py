import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

with (
    patch.dict(os.environ, {"DEBUG": "false", "GOOGLE_API_KEY": "unit-test-key"}),
    patch("qdrant_client.QdrantClient"),
):
    from app.main import app


class AvailabilityTests(unittest.TestCase):
    def test_qdrant_outage_does_not_disable_api_or_chat(self):
        with (
            patch("app.main.ensure_collection_exists", side_effect=ConnectionError),
            patch("app.main.qdrant.get_collections", side_effect=ConnectionError),
            TestClient(app) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/health/ready").status_code, 503)
            response = client.get(
                "/api/status",
                headers={"Origin": "https://smart-ops-eight.vercel.app"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers["access-control-allow-origin"],
                "https://smart-ops-eight.vercel.app",
            )
            reply = client.post(
                "/api/ask", data={"question": "hello", "session_id": "availability-test"}
            ).json()
            self.assertEqual(reply["status"], "success")
            self.assertIn("ready to help", reply["answer"])

    def test_failed_pdf_upload_is_reported_as_error(self):
        with (
            patch("app.engines.semantic.ensure_collection_exists", side_effect=ConnectionError),
            TestClient(app) as client,
        ):
            reply = client.post(
                "/api/ask",
                data={"question": "Summarize this", "session_id": "failed-pdf-test"},
                files={"file": ("test.pdf", b"test", "application/pdf")},
            ).json()
        self.assertEqual(reply["status"], "error")
        self.assertIn("PDF storage is temporarily unavailable", reply["error"])
        self.assertIsNone(reply["answer"])

    def test_engine_errors_and_metadata_reach_client(self):
        with (
            patch(
                "app.api.routes.smartops_graph.invoke",
                return_value={
                    "status": "error",
                    "error": "Please upload the CSV again.",
                    "engine": "csv",
                    "chunks_used": 0,
                },
            ),
            TestClient(app) as client,
        ):
            reply = client.post("/api/ask", data={"question": "Count rows"}).json()
        self.assertEqual(reply["error"], "Please upload the CSV again.")
        self.assertEqual(reply["engine_used"], "csv")
        self.assertEqual(reply["chunks_used"], 0)


if __name__ == "__main__":
    unittest.main()
