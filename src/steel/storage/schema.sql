-- Slice 0: platform-wide reference data and tenants.
-- Tenant-owned tables (enquiries, rfqs, quotes, traces) arrive with their slices.

CREATE TABLE IF NOT EXISTS tenants (
    id      TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    country TEXT NOT NULL
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
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS enquiries_by_tenant ON enquiries (tenant_id);

CREATE TABLE IF NOT EXISTS enquiry_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    enquiry_id   TEXT NOT NULL REFERENCES enquiries (id),
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content      BLOB NOT NULL
);
