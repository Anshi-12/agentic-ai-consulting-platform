# Group 3 — Central API, Authentication, and Consulting Dashboard

This branch integrates Group 2 without changing its notebook-first implementation. The central API owns demo authentication, role checks, project isolation, audit events, and a stable proxy contract for Group 2 knowledge/research/analytics tools.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --app-dir group3
```

The API is available at `http://localhost:8000/docs`.

Run the required Next.js frontend in a second terminal:

```bash
cd group3/frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The Meridian dashboard uses Axios for API calls, Zustand for workspace state, and Tailwind CSS for layout and responsive styling. The main actions create a demo engagement, start the Group 1 workflow, and submit the manager approval decision through the central API.

The Group 2 adapter is configured with `GROUP2_API_URL`. Its development-only notebook API must remain bound to localhost. In production, replace the adapter with the shared authenticated service boundary.

Group 1 is now merged into the shared branch. To enable the full agent workflow locally, install both dependency sets from the repository root:

```bash
pip install -r requirements.txt
pip install -r group3/requirements.txt
```

The central API exposes the Group 1 engine through authenticated routes:

- `POST /api/v1/projects/{project_id}/agent-workflows`
- `GET /api/v1/agent-workflows/{execution_id}`
- `GET /api/v1/agent-workflows/{execution_id}/outputs`
- `POST /api/v1/agent-workflows/{execution_id}/resume`

The original Group 1 development app remains available at `backend/api/app.py` for its own workflow and WebSocket smoke tests. Group 3 owns the production router boundary.

## Continuous integration

Group 3 CI is defined in `.github/workflows/group3-ci.yml`. On Group 3 pull requests and pushes it runs the backend pytest suite, frontend Vitest tests, the Next.js production build, and Playwright Chromium tests. Playwright reports are uploaded when the browser suite fails.

## Versioned contracts

The canonical Group 3 contract package is `group3/contracts/v1.py`. It defines the shared role, project, task status, agent result, evidence, citation, review, audit, error, and WebSocket event models. `group3/api/contracts.py` is a compatibility export, so existing central API imports continue to use the same versioned models. New contract changes should be additive and preserve existing enum values and identifiers.

## Roles

`administrator` and `engagement_manager` can create projects, manage membership, and approve deliverables. Consultants and client executives must be project members (or owners); client executives can read only approved projects. JWTs expire according to `JWT_EXPIRES_MINUTES` (default 60 minutes), and `JWT_SECRET` must be configured for a deployed environment.

## Persistence

Group 3 stores users, projects, and audit events through SQLAlchemy. Local
development defaults to `sqlite:///./group3_runtime.db`; configure
`DATABASE_URL=postgresql+psycopg://...` for PostgreSQL. The Docker Compose
configuration provisions PostgreSQL with pgvector and points the central API at
it. The database is initialized when the Group 3 API starts, so a restart does
not discard the project workspace.

The Compose deployment builds the central API from the repository root so the existing Group 1 engine is available in-process. Group 2 remains an external service boundary; set `GROUP2_API_URL` to the separately running Group 2 service when using live research or analytics.
