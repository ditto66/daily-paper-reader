"""Tests for PostgreSQL statement timeout (57014) handling in Supabase RPC calls.

When the database returns a statement timeout error (HTTP 500, code 57014),
the client should:
1. Skip retries in _request_with_retries (retrying won't resolve a server-side limit)
2. Break early in rank_papers_for_queries_via_supabase (subsequent batches will also time out)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supabase_source import (
    _is_statement_timeout,
    _request_with_retries,
    match_papers_by_embedding,
)


class IsStatementTimeoutTest(unittest.TestCase):
    """Tests for _is_statement_timeout helper."""

    def _make_response(self, text: str, status_code: int = 500) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        return resp

    def test_detects_57014_in_json_body(self):
        body = '{"code":"57014","details":null,"hint":null,"message":"canceling statement due to statement timeout"}'
        resp = self._make_response(body)
        self.assertTrue(_is_statement_timeout(resp))

    def test_false_for_other_500_errors(self):
        body = '{"code":"XX000","message":"internal error"}'
        resp = self._make_response(body)
        self.assertFalse(_is_statement_timeout(resp))

    def test_false_for_empty_body(self):
        resp = self._make_response("")
        self.assertFalse(_is_statement_timeout(resp))

    def test_false_for_none_text(self):
        resp = MagicMock()
        resp.text = None
        self.assertFalse(_is_statement_timeout(resp))

    def test_false_for_57014_in_non_code_field(self):
        """Should not match 57014 appearing outside the 'code' field."""
        body = '{"code":"XX000","message":"error 57014 occurred"}'
        resp = self._make_response(body)
        self.assertFalse(_is_statement_timeout(resp))

    def test_false_for_non_json_body(self):
        resp = self._make_response("Internal Server Error")
        self.assertFalse(_is_statement_timeout(resp))


class RequestWithRetriesTimeoutTest(unittest.TestCase):
    """_request_with_retries should not retry on statement timeout."""

    @patch("supabase_source.requests.request")
    def test_no_retry_on_statement_timeout(self, mock_request):
        timeout_body = '{"code":"57014","message":"canceling statement due to statement timeout"}'
        resp = MagicMock()
        resp.status_code = 500
        resp.text = timeout_body
        mock_request.return_value = resp

        result = _request_with_retries(
            "POST",
            "https://example.com/rpc/test",
            headers={"apikey": "test"},
            timeout=20,
            retries=3,
        )
        # Should only be called once (no retries)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(result.status_code, 500)

    @patch("supabase_source.requests.request")
    def test_retries_on_non_timeout_500(self, mock_request):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"message":"internal error"}'
        mock_request.return_value = resp

        result = _request_with_retries(
            "POST",
            "https://example.com/rpc/test",
            headers={"apikey": "test"},
            timeout=20,
            retries=3,
            retry_wait_seconds=0,
        )
        # Should retry all 4 attempts (1 initial + 3 retries)
        self.assertEqual(mock_request.call_count, 4)
        self.assertEqual(result.status_code, 500)


class MatchPapersTimeoutTest(unittest.TestCase):
    """match_papers_by_embedding should propagate 57014 in error message."""

    @patch("supabase_source._request_with_retries")
    def test_error_message_contains_57014(self, mock_req):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"code":"57014","message":"canceling statement due to statement timeout"}'
        mock_req.return_value = resp

        rows, msg = match_papers_by_embedding(
            url="https://example.supabase.co",
            api_key="test-key",
            rpc_name="match_arxiv_papers_exact",
            query_embedding=[0.1, 0.2, 0.3],
            match_count=10,
        )
        self.assertEqual(rows, [])
        self.assertIn("57014", msg)


if __name__ == "__main__":
    unittest.main()
