from uuid import uuid4

from contracts.v1 import AgentExecutionResult, Citation, Evidence, TaskStatus, WebSocketEvent


def test_task_status_values_are_stable():
    assert [status.value for status in TaskStatus] == [
        'queued', 'running', 'awaiting_review', 'completed', 'failed', 'cancelled'
    ]


def test_agent_result_serializes_evidence_and_citations():
    project_id = uuid4()
    evidence = Evidence(
        evidence_id='ev_001',
        project_id=project_id,
        source='client-upload',
        title='Deposit cohort extract',
        content='Illustrative client-provided evidence.',
        source_type='dataset',
        reliability=0.9,
    )
    result = AgentExecutionResult(
        task_id='task_001',
        project_id=project_id,
        agent_name='diagnostics',
        status=TaskStatus.completed,
        structured_output={'retention_gap': 0.06},
        evidence=[evidence],
        citations=[Citation(citation_id='cit_001', evidence_id='ev_001', claim='Retention gap identified')],
        confidence=0.82,
    )

    payload = result.model_dump(mode='json')
    assert payload['status'] == 'completed'
    assert payload['evidence'][0]['evidence_id'] == 'ev_001'
    assert payload['citations'][0]['evidence_id'] == 'ev_001'


def test_websocket_event_is_json_schema_compatible():
    event = WebSocketEvent(
        event_type='workflow.started',
        task_id='task_001',
        project_id=uuid4(),
        payload={'stage': 'discovery'},
    )
    assert event.model_dump(mode='json')['event_type'] == 'workflow.started'
