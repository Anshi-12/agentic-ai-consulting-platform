import { afterEach, describe, expect, it } from 'vitest';
import { useWorkspace } from '../../lib/workspace';

afterEach(() => {
  useWorkspace.setState({
    token: '',
    project: null,
    workflow: null,
    outputs: null,
    evidence: [],
  });
});

describe('workspace store', () => {
  it('stores the authenticated project context', () => {
    const project = {
      id: 'project_001',
      name: 'Deposit diagnostic',
      client_name: 'Harbor National Bank',
      industry: 'Banking',
      objectives: ['Improve retention'],
      constraints: ['90-day window'],
      status: 'active',
      owner_id: 'user_001',
      created_at: '2026-01-01T00:00:00Z',
    };

    useWorkspace.getState().setToken('demo-token');
    useWorkspace.getState().setProject(project);

    expect(useWorkspace.getState().token).toBe('demo-token');
    expect(useWorkspace.getState().project).toEqual(project);
  });

  it('updates workflow, outputs, and evidence independently', () => {
    useWorkspace.getState().setWorkflow({
      execution_id: 'wf_001',
      project_id: 'project_001',
      status: 'awaiting_input',
      current_stage: 'human_approval',
      pending_interrupt: { type: 'human_approval' },
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    });
    useWorkspace.getState().setOutputs({
      execution_id: 'wf_001',
      status: 'completed',
      report_markdown: '# Retention plan',
      warnings: [],
      security_flags: [],
      usage: {},
    });
    useWorkspace.getState().setEvidence([{ evidence_id: 'ev_001', title: 'Cohort extract' }]);

    expect(useWorkspace.getState().workflow?.current_stage).toBe('human_approval');
    expect(useWorkspace.getState().outputs?.report_markdown).toBe('# Retention plan');
    expect(useWorkspace.getState().evidence[0].evidence_id).toBe('ev_001');
  });
});
