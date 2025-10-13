# 4. Data Models & Schemas

## Volume Metadata
Includes filename, shape, voxel_size, intensity stats.

## Upload Session & File Record
Upload tracked by upload_id; assembled into FileRecord.

## Job Record
Tracks job_id, state, params, artifacts, errors.

## Preview Message
JSON with scale, shape, dtype, base64 payload.

## Artifact Stats
JSON of label voxel counts, physical volume, surface area.

## Auth Tokens
Bearer tokens with scopes, expiry.

---
