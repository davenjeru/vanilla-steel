-- Slice 0: platform-wide reference data and tenants.
-- Tenant-owned tables (enquiries, rfqs, quotes, traces) arrive with their slices.

CREATE TABLE IF NOT EXISTS tenants (
    id      TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    country TEXT NOT NULL
);

-- Slice 3: each tenant calls the API with its own key. Only a hash is stored, never the key.
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash  TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id)
);

CREATE TABLE IF NOT EXISTS suppliers (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,
    city      TEXT NOT NULL,
    country   TEXT NOT NULL,
    latitude  REAL NOT NULL,
    longitude REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_items (
    id              TEXT PRIMARY KEY,
    supplier_id     TEXT NOT NULL REFERENCES suppliers (id),
    grade           TEXT NOT NULL,
    form            TEXT NOT NULL CHECK (form IN ('sheet', 'plate', 'coil')),
    rolling         TEXT NOT NULL CHECK (rolling IN ('hot_rolled', 'cold_rolled')),
    thickness_mm    REAL NOT NULL CHECK (thickness_mm > 0),
    width_mm        REAL NOT NULL CHECK (width_mm > 0),
    length_mm       REAL CHECK (length_mm IS NULL OR length_mm > 0),
    quantity_t      REAL NOT NULL CHECK (quantity_t > 0),
    coating         TEXT,
    quality         TEXT NOT NULL CHECK (quality IN ('prime', 'second_choice')),
    price_eur_per_t TEXT NOT NULL  -- Decimal stored as text to avoid float rounding
);

-- Slice 1: enquiries are tenant-owned. Every query filters by tenant_id.
CREATE TABLE IF NOT EXISTS enquiries (
    id             TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL REFERENCES tenants (id),
    sender         TEXT NOT NULL,
    subject        TEXT NOT NULL,
    body           TEXT NOT NULL,
    status_history TEXT NOT NULL,  -- JSON list of {status, at}; the last entry is the current status
    spec           TEXT,           -- JSON MaterialSpec once extracted
    warnings       TEXT NOT NULL,  -- JSON list of strings
    failure_reason TEXT,
    matches        TEXT NOT NULL,  -- JSON list of ranked matches; internal, holds supplier IDs
    fingerprint    TEXT NOT NULL,  -- same for resent or forwarded enquiries (Slice 4)
    escalation_reason TEXT,        -- why a human must take over (Slice 4)
    escalation_detail TEXT,        -- internal explanation from the agent
    language       TEXT,           -- the customer's ISO 639-1 code, for the quote letter (Slice 5)
    duplicate_of   TEXT,           -- the same tenant's earlier enquiry this one repeats (Slice 3)
    is_correction  INTEGER NOT NULL DEFAULT 0,  -- the customer corrects an earlier enquiry (F-012)
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS enquiries_by_tenant ON enquiries (tenant_id);
CREATE INDEX IF NOT EXISTS enquiries_by_fingerprint ON enquiries (tenant_id, fingerprint);

CREATE TABLE IF NOT EXISTS enquiry_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    enquiry_id   TEXT NOT NULL REFERENCES enquiries (id),
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content      BLOB NOT NULL
);

-- Slice 3: a retried POST with the same Idempotency-Key replays the stored response.
-- The key is reserved before the enquiry is handled (response still NULL), so two simultaneous
-- retries can never both run the agent.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    tenant_id       TEXT NOT NULL REFERENCES tenants (id),
    idempotency_key TEXT NOT NULL,
    request_hash    TEXT NOT NULL,  -- SHA-256 of the request body; a different body under the same key is refused
    response        TEXT,           -- JSON response body; NULL while the first request is still running
    PRIMARY KEY (tenant_id, idempotency_key)
);

-- Slice 4: at most one RFQ per enquiry, enforced by the database, not only by the agent.
CREATE TABLE IF NOT EXISTS rfqs (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL REFERENCES tenants (id),
    enquiry_id    TEXT NOT NULL UNIQUE REFERENCES enquiries (id),
    stock_item_id TEXT NOT NULL REFERENCES stock_items (id),
    created_at    TEXT NOT NULL
);

-- Slice 5: at most one quote per RFQ and per enquiry, enforced by the database.
CREATE TABLE IF NOT EXISTS quotes (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants (id),
    enquiry_id   TEXT NOT NULL UNIQUE REFERENCES enquiries (id),
    rfq_id       TEXT NOT NULL UNIQUE REFERENCES rfqs (id),
    priced       TEXT NOT NULL,  -- JSON PricedQuote: lines and terms, no supplier fields
    cover_letter TEXT NOT NULL,  -- written by the LLM, scanned for supplier leaks before storing
    language     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

-- Slice 4: one agent run per enquiry, with every LLM turn and tool call. Internal only.
CREATE TABLE IF NOT EXISTS agent_runs (
    enquiry_id TEXT PRIMARY KEY REFERENCES enquiries (id),
    tenant_id  TEXT NOT NULL REFERENCES tenants (id),
    run        TEXT NOT NULL  -- JSON AgentRun
);
