# Group 3 release checklist

This checklist is for the final three-group integration. Group 3 owns the central API boundary, frontend, deployment configuration, and integration tests. It does not modify Group 2 implementation files.

## Local verification

Backend:

```bash
cd group3
source .venv/bin/activate
PYTHONPATH=. pytest -q
```

Frontend:

```bash
cd group3/frontend
npm install
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Docker:

```bash
cd group3
docker compose up --build
curl http://localhost:8000/health
docker compose ps
```

The expected API health response is `status=ok` with service `central-api`. PostgreSQL and central-api should report healthy in `docker compose ps`.

## Three-group integration

1. Merge the shared contracts and Group 1 workflow changes into the integration branch.
2. Keep Group 2 on its own service and configure `GROUP2_API_URL` to its reachable URL. Do not copy or edit Group 2 implementation files in Group 3.
3. Start Group 2 separately if live evidence, document indexing, or analytics are being demonstrated.
4. Open the Meridian dashboard and create the Harbor National Bank engagement.
5. Run the analysis and answer the clarification gate with client-confirmed or explicitly synthetic inputs.
6. Verify the manager approval gate, report output, evidence ledger, and Audit Trail.
7. Confirm that no unsupported market claims are presented as verified facts.

## GitHub merge checks

- Push each group's feature branch.
- Open pull requests into the agreed integration branch.
- Resolve contract conflicts by preserving the `v1` identifiers and enum values.
- Require `.github/workflows/group3-ci.yml` to pass before merging Group 3.
- Run the full three-group smoke test after all merges.
- Set production `JWT_SECRET`, `DATABASE_URL`, and `GROUP2_API_URL` through repository or deployment secrets; never commit `.env`.
