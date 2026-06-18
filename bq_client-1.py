"""
bq_client.py — BigQuery search client (service account JSON auth).

Builds parameterised queries from a whitelisted field map to prevent SQL
injection. Auth uses a service account JSON file whose path comes from the
BQ_SERVICE_ACCOUNT_FILE env var.

Cloud Run deployment:
  - Store the JSON file content as a secret in Secret Manager.
  - Mount it as a volume in Cloud Run (--set-secrets=/secrets/sa.json=BQ_SA_KEY:latest).
  - Set BQ_SERVICE_ACCOUNT_FILE=/secrets/sa.json.
  - The file never lives in git or in the container image.

Local dev:
  - Place serviceaccount.json anywhere on disk.
  - Set BQ_SERVICE_ACCOUNT_FILE=/full/path/to/serviceaccount.json in .env.
"""
from __future__ import annotations

import os
from typing import Any

from google.cloud import bigquery
from google.oauth2 import service_account

import config
from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Field whitelist — only these can be filtered on. Anything else is rejected
# before it reaches BigQuery, eliminating SQL injection risk via field names.
# ---------------------------------------------------------------------------

FIELD_TYPES: dict[str, str] = {
    "XQCBA1": "STRING",
    "OGBEAC": "STRING",
    "OGBEA1": "STRING",
    "OGBEA2": "STRING",
    "OGBEA3": "STRING",
    "OGBEA4": "STRING",
    "OGT521": "STRING",
    "OGTRNO": "STRING",
    "OGTMZ1": "STRING",
    "OGCPDT": "INT64",
    "OGCPTM": "INT64",
    "OGAVDT": "INT64",
    "OGAVTM": "INT64",
    "OGPYAM": "FLOAT64",
    "OGPALE": "FLOAT64",
    "OGPYCY": "STRING",
    "OGNAR1": "STRING",
    "OGNAR2": "STRING",
    "OGNAR3": "STRING",
    "OGNAR4": "STRING",
}

ALLOWED_FIELDS = set(FIELD_TYPES.keys())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class BigQueryError(Exception):
    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BigQueryClient:
    """Thin wrapper around bigquery.Client with parameterised query helper.

    The underlying google-cloud-bigquery Client is thread-safe and pools
    connections internally, so we keep one instance per process.
    """

    def __init__(self) -> None:
        self._validate_config()

        creds = service_account.Credentials.from_service_account_file(
            config.BQ_SERVICE_ACCOUNT_FILE
        )
        self._client = bigquery.Client(
            credentials=creds,
            project=config.BQ_PROJECT_ID,
        )

        log.info(
            "BigQueryClient ready | project=%s dataset=%s table=%s "
            "sa_file=%s cmek=%s",
            config.BQ_PROJECT_ID,
            config.BQ_DATASET_ID,
            config.BQ_TABLE_ID,
            config.BQ_SERVICE_ACCOUNT_FILE,
            "enabled" if config.BQ_KMS_KEY else "disabled",
        )

    @staticmethod
    def _validate_config() -> None:
        missing = [
            name for name, val in {
                "BQ_PROJECT_ID":         config.BQ_PROJECT_ID,
                "BQ_DATASET_ID":         config.BQ_DATASET_ID,
                "BQ_TABLE_ID":           config.BQ_TABLE_ID,
                "BQ_SERVICE_ACCOUNT_FILE": config.BQ_SERVICE_ACCOUNT_FILE,
            }.items() if not val
        ]
        if missing:
            raise RuntimeError(
                f"BigQueryClient misconfigured — missing env vars: {', '.join(missing)}"
            )

        if not os.path.isfile(config.BQ_SERVICE_ACCOUNT_FILE):
            raise RuntimeError(
                f"Service account file not found at: {config.BQ_SERVICE_ACCOUNT_FILE}. "
                f"Check BQ_SERVICE_ACCOUNT_FILE env var and that the file is mounted/readable."
            )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        """Run a SELECT * with equality filters on whitelisted fields.

        Returns a list of dicts (one per row). Raises BigQueryError on failure.
        """
        where_clauses: list[str] = []
        query_params: list[bigquery.ScalarQueryParameter] = []

        for field, value in filters.items():
            if field not in ALLOWED_FIELDS:
                # Should never happen — Pydantic already enforces this — but
                # defence in depth.
                log.warning("Skipping unknown field: %s", field)
                continue

            param_type = FIELD_TYPES[field]
            # Backticks around the field name are safe here because the field
            # comes from our whitelist, not user input.
            where_clauses.append(f"`{field}` = @{field}")
            query_params.append(
                bigquery.ScalarQueryParameter(field, param_type, value)
            )

        query_params.append(
            bigquery.ScalarQueryParameter("row_limit", "INT64", limit)
        )

        table_ref = f"`{config.BQ_PROJECT_ID}.{config.BQ_DATASET_ID}.{config.BQ_TABLE_ID}`"
        query = f"SELECT * FROM {table_ref}"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " LIMIT @row_limit"

        job_config_kwargs: dict[str, Any] = {"query_parameters": query_params}
        if config.BQ_KMS_KEY:
            job_config_kwargs["destination_encryption_configuration"] = (
                bigquery.EncryptionConfiguration(kms_key_name=config.BQ_KMS_KEY)
            )
        job_config = bigquery.QueryJobConfig(**job_config_kwargs)

        # Log query shape but NOT parameter values (those may be PII).
        log.info("BQ -> query=%s | param_count=%d limit=%d",
                 query, len(query_params) - 1, limit)

        try:
            job = self._client.query(query, job_config=job_config)
            rows = [dict(row) for row in job.result()]
        except Exception as e:
            log.exception("BQ query failed: %s", e)
            raise BigQueryError(
                f"BigQuery query failed: {e}", status_code=500
            ) from e

        log.info("BQ <- rows=%d", len(rows))
        return rows

    # ------------------------------------------------------------------
    # Reconcile: from a given TRNREF, walk forward in capture date+time
    # order accumulating OGPYAM until the running total reaches target_amount.
    # The transaction that crosses the target IS included.
    # ------------------------------------------------------------------

    def reconcile_from_trnref(
        self,
        trnref: str,
        target_amount: float,
    ) -> dict[str, Any]:
        """Find the anchor txn by OGTRNO, then return all transactions from
        that point forward (ordered by capture date+time) until the cumulative
        OGPYAM reaches target_amount.

        Returns a dict with the matched rows, running totals, and whether the
        target was fully reconciled.
        """
        table_ref = f"`{config.BQ_PROJECT_ID}.{config.BQ_DATASET_ID}.{config.BQ_TABLE_ID}`"

        # Single query:
        #   1. CTE `anchor` finds the capture date+time of the given TRNREF.
        #   2. Main query returns every row at-or-after that (date, time),
        #      ordered ascending so Python can accumulate in sequence.
        # Ordering key is (OGCPDT, OGCPTM) — capture date then capture time.
        query = f"""
            WITH anchor AS (
                SELECT OGCPDT AS a_dt, OGCPTM AS a_tm
                FROM {table_ref}
                WHERE OGTRNO = @trnref
                ORDER BY OGCPDT, OGCPTM
                LIMIT 1
            )
            SELECT t.*
            FROM {table_ref} t, anchor
            WHERE (t.OGCPDT > anchor.a_dt)
               OR (t.OGCPDT = anchor.a_dt AND t.OGCPTM >= anchor.a_tm)
            ORDER BY t.OGCPDT, t.OGCPTM
        """

        query_params = [
            bigquery.ScalarQueryParameter("trnref", "STRING", trnref),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)

        log.info("BQ reconcile -> trnref=%s target=%.2f", trnref, target_amount)

        try:
            job = self._client.query(query, job_config=job_config)
            all_rows = [dict(row) for row in job.result()]
        except Exception as e:
            log.exception("BQ reconcile query failed: %s", e)
            raise BigQueryError(
                f"BigQuery reconcile query failed: {e}", status_code=500
            ) from e

        if not all_rows:
            raise BigQueryError(
                f"No transaction found with OGTRNO={trnref}", status_code=404
            )

        # Walk forward accumulating OGPYAM until we reach/exceed target.
        # The crossing transaction is included (per requirement).
        matched: list[dict[str, Any]] = []
        running_total = 0.0
        reconciled = False

        for row in all_rows:
            amount = row.get("OGPYAM")
            # Skip rows with no payment amount — they can't contribute.
            if amount is None:
                matched.append(row)
                continue

            running_total += float(amount)
            matched.append(row)

            if running_total >= target_amount:
                reconciled = True
                break

        log.info(
            "BQ reconcile <- matched=%d running_total=%.2f reconciled=%s",
            len(matched), running_total, reconciled,
        )

        return {
            "trnref":          trnref,
            "target_amount":   target_amount,
            "reconciled":      reconciled,
            "running_total":   round(running_total, 2),
            "shortfall":       round(max(0.0, target_amount - running_total), 2),
            "row_count":       len(matched),
            "rows":            matched,
        }
