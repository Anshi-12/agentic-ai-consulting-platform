import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockedAxios = vi.hoisted(() => {
  const instance = {
    defaults: { headers: { common: {} as Record<string, string> } },
    get: vi.fn(),
    post: vi.fn(),
  };
  return { instance, create: vi.fn(() => instance) };
});

vi.mock('axios', () => ({ default: { create: mockedAxios.create } }));

import { createProject, demoLogin, resumeWorkflow, setToken } from '../../lib/api';

describe('central API client', () => {
  beforeEach(() => {
    mockedAxios.instance.get.mockReset();
    mockedAxios.instance.post.mockReset();
    mockedAxios.instance.defaults.headers.common = {};
  });

  it('stores the bearer token after demo login', async () => {
    mockedAxios.instance.post.mockResolvedValueOnce({
      data: { access_token: 'jwt-demo-token', token_type: 'bearer' },
    });

    const result = await demoLogin('engagement_manager');

    expect(mockedAxios.instance.post).toHaveBeenCalledWith(
      '/api/v1/auth/demo-login',
      null,
      { params: { role: 'engagement_manager' } },
    );
    expect(result.access_token).toBe('jwt-demo-token');
    expect(mockedAxios.instance.defaults.headers.common.Authorization).toBe('Bearer jwt-demo-token');
  });

  it('sends project creation data through the versioned API', async () => {
    mockedAxios.instance.post.mockResolvedValueOnce({ data: { id: 'project_001', status: 'active' } });
    const input = {
      name: 'Deposit diagnostic',
      client_name: 'Harbor National Bank',
      industry: 'Banking',
      objectives: ['Improve retention'],
      constraints: ['90-day window'],
    };

    const result = await createProject(input);

    expect(mockedAxios.instance.post).toHaveBeenCalledWith('/api/v1/projects', input);
    expect(result.id).toBe('project_001');
  });

  it('wraps workflow clarification and approval payloads consistently', async () => {
    mockedAxios.instance.post.mockResolvedValueOnce({ data: { status: 'awaiting_input' } });

    await setToken('jwt-demo-token');
    await resumeWorkflow('wf_001', {
      type: 'clarification',
      answers: { question: 'Increase 90-day retention to 70 percent.' },
    });

    expect(mockedAxios.instance.post).toHaveBeenCalledWith(
      '/api/v1/agent-workflows/wf_001/resume',
      { payload: { type: 'clarification', answers: { question: 'Increase 90-day retention to 70 percent.' } } },
    );
  });
});
