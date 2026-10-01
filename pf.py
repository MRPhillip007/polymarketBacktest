"""Helpers for the pendulumflow Polymarket archive (v3 hours)."""
import duckdb, datetime as dt, re

BASE = 'https://archive.pendulumflow.com/v3'

def url(h: dt.datetime) -> str:
    return f"{BASE}/{h:%Y-%m-%d}/{h:%H}/{h:%Y-%m-%dT%H}.parquet"

def con():
    c = duckdb.connect()
    c.sql("SET TimeZone='UTC'")
    c.sql("INSTALL httpfs; LOAD httpfs;")
    c.sql("SET enable_http_metadata_cache=true; SET enable_object_cache=true;")
    return c

SLUG_RE = re.compile(r'^(?P<coin>[a-z0-9]+)-updown-(?P<tf>\d+m)-(?P<start>\d{10})$')
