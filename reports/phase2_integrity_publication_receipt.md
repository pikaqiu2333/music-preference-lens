# Phase 2 Integrity Amendment Publication Receipt

- Local implementation commit: `5255885ac78364011fe65b8ef999a7accec16108`
- Local machine-readable receipt commit: `2b217eb`
- Public Gist: https://gist.github.com/a56668283b095f59f0eacf0527395b58
- Exact method-code publication history: `d8930a7f4cda8a9d62f3bd8668cb691fb257a408`
- Latest history after correcting the Chinese report transport encoding: `6444e8d73633e833bab3daf6568b24dd7c518891`
- Latest publication timestamp: `2026-07-13T13:04:48Z`

The verifier, exporter, mechanism analysis, GPU runner, shard merger, scorer,
English amendment, and machine-readable receipt were downloaded through their
Gist raw URLs and compared with local bytes before the v2 catalog replay. All
method files and receipts matched exactly. The Chinese report was initially
corrupted by a PowerShell native-pipeline encoding conversion, was immediately
republished with ASCII-escaped JSON transport, and then matched local UTF-8
bytes exactly at the latest history version above.

The v2 catalog replay started only after these checks. Its formal outputs must
therefore name verifier version `phase2_catalog_v2_complete_alias_audit` and are
bound to verifier SHA-256
`c9659b932ac3f1064e75e0b5b03c8cad5aa34cf7ca84efa2946c8dd9a9bb9867`.
