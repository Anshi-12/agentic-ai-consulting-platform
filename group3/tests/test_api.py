from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from uuid import UUID

from api.main import app

client=TestClient(app)

def headers(role="engagement_manager"):
    return {"Authorization":"Bearer "+client.post('/api/v1/auth/demo-login',params={'role':role}).json()['access_token']}

def test_project_isolation_and_approval():
    h=headers(); p=client.post('/api/v1/projects',headers=h,json={'name':'Growth diagnostic','client_name':'Northstar Retail','industry':'Retail','objectives':['Retention']}); assert p.status_code==200; pid=p.json()['id']
    client_login=client.post('/api/v1/auth/demo-login',params={'role':'client_executive'}).json()
    assert client.get(f'/api/v1/projects/{pid}',headers={'Authorization':'Bearer '+client_login['access_token']}).status_code==403
    member_id=client_login['user']['id']
    assert client.post(f'/api/v1/projects/{pid}/members',headers=h,json={'user_id':member_id}).status_code==200
    assert client.get(f'/api/v1/projects/{pid}',headers={'Authorization':'Bearer '+client_login['access_token']}).status_code==403
    assert client.post(f'/api/v1/projects/{pid}/status',headers=h,json={'decision':'approve','comment':'Reviewed'}).json()['status']=='approved'
    assert client.get(f'/api/v1/projects/{pid}',headers={'Authorization':'Bearer '+client_login['access_token']}).status_code==200

def test_validation_and_auth():
    assert client.get('/api/v1/projects').status_code==401
    assert client.post('/api/v1/projects',headers=headers(),json={'name':'x'}).status_code==422

def test_unassigned_user_cannot_read_project():
    project = client.post('/api/v1/projects',headers=headers(),json={'name':'Access diagnostic','client_name':'Northstar Retail','industry':'Retail','objectives':['Retention']}).json()
    other = client.post('/api/v1/auth/demo-login',params={'role':'consultant'}).json()
    assert client.get(f"/api/v1/projects/{project['id']}",headers={'Authorization':'Bearer '+other['access_token']}).status_code == 403

def test_audit_log_is_manager_only():
    manager_headers = headers('engagement_manager')
    assert client.get('/api/v1/audit', headers=manager_headers).status_code == 200
    consultant_headers = headers('consultant')
    assert client.get('/api/v1/audit', headers=consultant_headers).status_code == 403

def test_project_is_written_to_persistent_store():
    response = client.post(
        '/api/v1/projects',
        headers=headers('consultant'),
        json={
            'name': 'Persistence diagnostic',
            'client_name': 'Harbor National Bank',
            'industry': 'Banking',
            'objectives': ['Improve deposit retention'],
            'constraints': ['90-day decision window'],
        },
    )
    assert response.status_code == 200
    project_id = response.json()['id']

    from api.store import projects

    persisted = projects.get(UUID(project_id))
    assert persisted is not None
    assert persisted.client_name == 'Harbor National Bank'
    assert persisted.status == 'active'

def test_websocket_rejects_missing_token():
    project = client.post(
        '/api/v1/projects',
        headers=headers(),
        json={'name':'WebSocket diagnostic','client_name':'Northstar Retail','industry':'Retail','objectives':['Retention']},
    ).json()

    try:
        with client.websocket_connect(f"/ws/projects/{project['id']}/workflows/wf_missing"):
            assert False, 'Expected the unauthenticated websocket to be rejected'
    except WebSocketDisconnect as exc:
        assert exc.code == 4401

def test_websocket_replays_project_scoped_events(monkeypatch):
    project = client.post(
        '/api/v1/projects',
        headers=headers(),
        json={'name':'Event diagnostic','client_name':'Northstar Retail','industry':'Retail','objectives':['Retention']},
    ).json()
    token = client.post('/api/v1/auth/demo-login', params={'role':'engagement_manager'}).json()['access_token']

    class FakeStatus:
        project_id = project['id']

    class FakeService:
        def get_status(self, execution_id, principal):
            assert execution_id == 'wf_test'
            return FakeStatus()

        def get_events(self, execution_id, principal):
            return [{
                'channel': 'workflow:wf_test',
                'event': {'type': 'workflow.started', 'execution_id': execution_id},
            }]

        def subscribe(self, execution_id, principal, callback):
            return lambda: None

    import api.group1_adapter as group1_adapter

    monkeypatch.setattr(group1_adapter, '_execution_principal', lambda user, execution_id: (FakeService(), object()))

    with client.websocket_connect(f"/ws/projects/{project['id']}/workflows/wf_test?token={token}") as websocket:
        message = websocket.receive_json()
        assert message['event']['type'] == 'workflow.started'
        assert message['event']['execution_id'] == 'wf_test'
