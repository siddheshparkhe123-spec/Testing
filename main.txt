"""
main.py — FastAPI gateway. UPI + HSBC Hub + BigQuery search.

Routes
  GET  /health
  POST /api/v1/upi/secure
  GET  /api/v1/transactions/history?accountNumber=... + body sensitiveData
  POST /api/v1/accounts/apply-hold   (body forwarded as-is)
  GET  /api/v1/accounts/demand-deposit?hubCustomerNumber=... + body sensitiveData
  POST /api/v1/search                (BigQuery filtered search)
  POST /api/v1/reconcile             (BigQuery amount reconciliation from TRNREF)

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from bq_client import BigQueryClient, BigQueryError, FIELD_TYPES
from hsbc_client import HSBCError, HSBCHubClient
from logger import get_logger
from upi import AESCipher, CryptoError, UPIClient, UPIError

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Starting HSBC MHA Gateway ===")
    log.info("VERIFY : False (HSBC internal CA)")
    log.info("TIMEOUT: connect=%.0fs read=%.0fs",
             config.HTTP_CONNECT_TIMEOUT, config.HTTP_READ_TIMEOUT)

    http = httpx.AsyncClient(
        verify=False,
        timeout=httpx.Timeout(
            connect=config.HTTP_CONNECT_TIMEOUT,
            read=config.HTTP_READ_TIMEOUT,
            write=config.HTTP_READ_TIMEOUT,
            pool=config.HTTP_CONNECT_TIMEOUT,
        ),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    hsbc = HSBCHubClient()
    bq   = BigQueryClient()

    app.state.upi  = UPIClient(http=http, cipher=AESCipher(config.UPI_AES_KEY_HEX))
    app.state.hsbc = hsbc
    app.state.bq   = bq
    app.state.http = http

    log.info("=== Gateway ready ===")
    try:
        yield
    finally:
        await http.aclose()
        hsbc.close()
        bq.close()
        log.info("=== Gateway shut down ===")


app = FastAPI(title="HSBC MHA Gateway", version="1.0.0", lifespan=lifespan, docs_url="/docs")


@app.exception_handler(Exception)
async def _unhandled(_req: Request, exc: Exception):
    log.exception("Unhandled: %s", exc)
    return JSONResponse(
        {"error": "internal_error", "message": "An unexpected error occurred"},
        status_code=500,
    )


def _hsbc_error_detail(req_id: str, e: HSBCError) -> dict[str, Any]:
    return {
        "requestId":      req_id,
        "error":          "upstream_error",
        "message":        str(e),
        "upstreamStatus": e.upstream_status,
        "upstreamBody":   e.upstream_body,
    }


class SensitiveDataBody(BaseModel):
    sensitiveData: list[dict[str, str]] = Field(
        ...,
        description='key = value used in accountNumber/hubCustomerNumber query param, '
                    'value = real HSBC internal account number',
        examples=[[{"key": "abc123", "value": "INHSBC500021738001"}]],
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "hsbc-mha-gateway"}


# ---------------------------------------------------------------------------
# UPI Secure
# ---------------------------------------------------------------------------

@app.post("/api/v1/upi/secure", tags=["UPI"])
async def upi_secure(
    request: Request,
    payload: dict[str, Any] = Body(..., examples=[{"upiId": "user@bank", "amount": 100.5}]),
):
    req_id = str(uuid.uuid4())
    log.info("[%s] POST /upi/secure", req_id)
    upi: UPIClient = request.app.state.upi
    try:
        result = await upi.send_secure(payload)
        return {"requestId": req_id, "data": result}
    except CryptoError as e:
        log.error("[%s] crypto: %s", req_id, e)
        raise HTTPException(502, detail={
            "requestId": req_id, "error": "crypto_error", "message": str(e),
        })
    except UPIError as e:
        log.error("[%s] upi: %s", req_id, e)
        raise HTTPException(e.status_code, detail={
            "requestId": req_id, "error": "upstream_error",
            "message": str(e), "upstreamStatus": e.upstream_status,
        })


# ---------------------------------------------------------------------------
# Transaction History
# ---------------------------------------------------------------------------

@app.get("/api/v1/transactions/history", tags=["Transactions"])
async def get_transaction_history(
    request:        Request,
    account_number: str           = Query(..., alias="accountNumber"),
    from_date:      Optional[str] = Query(default=None, alias="fromDate", description="YYYY-MM-DD"),
    to_date:        Optional[str] = Query(default=None, alias="toDate",   description="YYYY-MM-DD"),
    body:           SensitiveDataBody = Body(...),
):
    req_id = str(uuid.uuid4())
    log.info("[%s] GET /transactions/history accountNumber=%s", req_id, account_number)
    sensitive_data = json.dumps(body.sensitiveData, separators=(",", ":"))
    hub: HSBCHubClient = request.app.state.hsbc
    try:
        result = await asyncio.to_thread(
            hub.get_transaction_history, account_number, from_date, to_date, sensitive_data
        )
        return {"requestId": req_id, "data": result}
    except HSBCError as e:
        log.error("[%s] %s", req_id, e)
        raise HTTPException(e.status_code, detail=_hsbc_error_detail(req_id, e))


# ---------------------------------------------------------------------------
# Apply Account Hold
# ---------------------------------------------------------------------------

@app.post("/api/v1/accounts/apply-hold", tags=["Accounts"])
async def apply_account_hold(
    request: Request,
    body:    dict[str, Any] = Body(..., examples=[{
        "applyHoldDetails": {
            "accountNumber":       "INHSBC011019072006",
            "accountCurrencyCode": "INR",
            "holdTillExpiredDate": "2028-12-31",
            "holdType":            "B",
            "holdCurrency":        "INR",
            "holdAmount":          "28.28",
        }
    }]),
):
    req_id = str(uuid.uuid4())
    details = body.get("applyHoldDetails", body) if isinstance(body, dict) else {}
    log.info("[%s] POST /accounts/apply-hold accountNumber=%s amount=%s",
             req_id, details.get("accountNumber"), details.get("holdAmount"))
    hub: HSBCHubClient = request.app.state.hsbc
    try:
        result = await asyncio.to_thread(hub.apply_account_hold, body)
        return {"requestId": req_id, "data": result}
    except HSBCError as e:
        log.error("[%s] %s", req_id, e)
        raise HTTPException(e.status_code, detail=_hsbc_error_detail(req_id, e))


# ---------------------------------------------------------------------------
# Demand Deposit Account Enquiry
# ---------------------------------------------------------------------------

@app.get("/api/v1/accounts/demand-deposit", tags=["Accounts"])
async def get_demand_deposit_account(
    request:             Request,
    hub_customer_number: str = Query(..., alias="hubCustomerNumber"),
    body:                SensitiveDataBody = Body(...),
):
    req_id = str(uuid.uuid4())
    log.info("[%s] GET /accounts/demand-deposit hubCustomerNumber=%s", req_id, hub_customer_number)
    sensitive_data = json.dumps(body.sensitiveData, separators=(",", ":"))
    hub: HSBCHubClient = request.app.state.hsbc
    try:
        result = await asyncio.to_thread(
            hub.get_demand_deposit_account, hub_customer_number, sensitive_data
        )
        return {"requestId": req_id, "data": result}
    except HSBCError as e:
        log.error("[%s] %s", req_id, e)
        raise HTTPException(e.status_code, detail=_hsbc_error_detail(req_id, e))


# ---------------------------------------------------------------------------
# BigQuery Search
# All filter fields are Optional[str] — Pydantic still enforces presence in
# the whitelist by virtue of being declared here; anything else is rejected
# at the FastAPI boundary before reaching bq_client.
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    XQCBA1: Optional[str] = Field(None, description="Correspondent Bank/Branch Name and Address")
    OGBEAC: Optional[str] = Field(None, description="Beneficiary A/C No")
    OGBEA1: Optional[str] = Field(None, description="Beneficiary Name and Address Line 1")
    OGBEA2: Optional[str] = Field(None, description="Beneficiary Name and Address Line 2")
    OGBEA3: Optional[str] = Field(None, description="Beneficiary Name and Address Line 3")
    OGBEA4: Optional[str] = Field(None, description="Beneficiary Name and Address Line 4")
    OGT521: Optional[str] = Field(None, description="Tag52 Ordering Institution Line 1")
    OGTRNO: Optional[str] = Field(None, description="Transaction Reference No")
    OGTMZ1: Optional[str] = Field(None, description="Time Stamp")
    OGCPDT: Optional[int] = Field(None, description="Capture Date (e.g. 20260926)")
    OGCPTM: Optional[int] = Field(None, description="Capture Time (e.g. 133101)")
    OGAVDT: Optional[int] = Field(None, description="Approval/Verification Date")
    OGAVTM: Optional[int] = Field(None, description="Approval/Verification Time")
    OGPYAM: Optional[float] = Field(None, description="Payment Amount")
    OGPALE: Optional[float] = Field(None, description="Lcy Payment Amount")
    OGPYCY: Optional[str] = Field(None, description="Payment Currency")
    OGNAR1: Optional[str] = Field(None, description="Narrative 1")
    OGNAR2: Optional[str] = Field(None, description="Narrative 2")
    OGNAR3: Optional[str] = Field(None, description="Narrative 3")
    OGNAR4: Optional[str] = Field(None, description="Narrative 4")
    limit:  Optional[int] = Field(100, ge=1, le=10000, description="Max rows to return")


@app.post("/api/v1/search", tags=["Search"])
async def search(request: Request, req: SearchRequest):
    req_id = str(uuid.uuid4())

    # Build filters dict from non-null fields, excluding `limit`
    filters = {
        k: v for k, v in req.model_dump(exclude_none=True).items()
        if k != "limit" and k in FIELD_TYPES
    }

    if not filters:
        raise HTTPException(
            status_code=400,
            detail={"requestId": req_id, "error": "bad_request",
                    "message": "Provide at least one field to filter on."},
        )

    limit = req.limit or 100
    log.info("[%s] POST /search filters=%s limit=%d",
             req_id, list(filters.keys()), limit)

    bq: BigQueryClient = request.app.state.bq
    try:
        rows = await asyncio.to_thread(bq.search, filters, limit)
        return {
            "requestId":     req_id,
            "filters_used":  filters,
            "row_count":     len(rows),
            "rows":          rows,
        }
    except BigQueryError as e:
        log.error("[%s] %s", req_id, e)
        raise HTTPException(e.status_code, detail={
            "requestId": req_id, "error": "bigquery_error", "message": str(e),
        })


# ---------------------------------------------------------------------------
# BigQuery Reconcile
# From a transaction reference number, walk forward in capture date+time order
# accumulating OGPYAM (Payment Amount) until the running total reaches the
# target amount. The transaction that crosses the target is included.
# ---------------------------------------------------------------------------

class ReconcileRequest(BaseModel):
    trnref:        str   = Field(..., description="Transaction Reference No (OGTRNO) of the anchor txn",
                                 examples=["LP BOM600010HIB"])
    target_amount: float = Field(..., gt=0, alias="targetAmount",
                                 description="Amount to reconcile up to, e.g. 500",
                                 examples=[500.0])


@app.post("/api/v1/reconcile", tags=["Search"])
async def reconcile(request: Request, req: ReconcileRequest):
    req_id = str(uuid.uuid4())
    log.info("[%s] POST /reconcile trnref=%s target=%.2f",
             req_id, req.trnref, req.target_amount)

    bq: BigQueryClient = request.app.state.bq
    try:
        result = await asyncio.to_thread(
            bq.reconcile_from_trnref, req.trnref, req.target_amount
        )
        return {"requestId": req_id, **result}
    except BigQueryError as e:
        log.error("[%s] %s", req_id, e)
        raise HTTPException(e.status_code, detail={
            "requestId": req_id, "error": "bigquery_error", "message": str(e),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
