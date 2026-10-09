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
    thickness_mm    REAL NOT NULL CHECK (thickness_mm > 0),
    width_mm        REAL NOT NULL CHECK (width_mm > 0),
    length_mm       REAL CHECK (length_mm IS NULL OR length_mm > 0),
    quantity_t      REAL NOT NULL CHECK (quantity_t > 0),
    coating         TEXT,
    quality         TEXT NOT NULL CHECK (quality IN ('prime', 'second_choice')),
    price_eur_per_t TEXT NOT NULL  -- Decimal stored as text to avoid float rounding
);
