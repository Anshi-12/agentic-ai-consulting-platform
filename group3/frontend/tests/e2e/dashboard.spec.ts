import { expect, test } from '@playwright/test';

test('opens the Meridian engagement workspace', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Deposit growth diagnostic')).toBeVisible();
  await page.getByRole('button', { name: 'Open engagement' }).click();
  await expect(page.getByRole('button', { name: 'Engagement open' })).toBeVisible();
});

test('exposes the human-in-the-loop workflow action', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Manager review gate')).toBeVisible();
  await expect(page.getByText('Evidence-backed workspace')).toBeVisible();
});

test('completes clarification, approval, and report generation', async ({ page }) => {
  let workflowPhase: 'clarification' | 'approval' | 'completed' = 'clarification';
  const project = {
    id: 'project_demo_001',
    name: 'Harbor National deposit growth diagnostic',
    client_name: 'Harbor National Bank',
    industry: 'Banking',
    objectives: ['Improve primary deposit retention'],
    constraints: ['90-day decision window', 'Protect branch relationships'],
    status: 'active',
    owner_id: 'user_demo',
    created_at: '2026-01-01T00:00:00Z',
  };

  await page.route('**/api/v1/auth/demo-login**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ access_token: 'demo-token', token_type: 'bearer' }),
    });
  });
  await page.route('**/api/v1/projects', async route => {
    if (route.request().method() === 'POST') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(project) });
      return;
    }
    await route.continue();
  });
  await page.route('**/api/v1/projects/project_demo_001/agent-workflows', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      execution_id: 'wf_demo_001',
      project_id: project.id,
      status: 'awaiting_input',
      current_stage: 'clarification',
      pending_interrupt: {
        type: 'clarification',
        questions: ['What metric and target define success for this banking engagement?'],
      },
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    }) });
  });
  await page.route('**/api/v1/agent-workflows/wf_demo_001', async route => {
    const status = workflowPhase === 'clarification'
      ? {
          execution_id: 'wf_demo_001', project_id: project.id, status: 'awaiting_input', current_stage: 'clarification',
          pending_interrupt: { type: 'clarification', questions: ['What metric and target define success?'] },
        }
      : workflowPhase === 'approval'
        ? {
            execution_id: 'wf_demo_001', project_id: project.id, status: 'awaiting_input', current_stage: 'human_approval',
            pending_interrupt: { type: 'human_approval', questions: [] },
          }
        : {
            execution_id: 'wf_demo_001', project_id: project.id, status: 'completed', current_stage: 'report',
            pending_interrupt: null,
          };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      ...status, created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }) });
  });
  await page.route('**/api/v1/agent-workflows/wf_demo_001/resume', async route => {
    workflowPhase = workflowPhase === 'clarification' ? 'approval' : 'completed';
    const completed = workflowPhase === 'completed';
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      execution_id: 'wf_demo_001', project_id: project.id, status: completed ? 'completed' : 'awaiting_input',
      current_stage: completed ? 'report' : 'human_approval', pending_interrupt: completed ? null : { type: 'human_approval', questions: [] },
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }) });
  });
  await page.route('**/api/v1/agent-workflows/wf_demo_001/outputs', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      execution_id: 'wf_demo_001', status: 'completed', report_markdown: '# Harbor National retention plan',
      warnings: [], security_flags: [], usage: {},
    }) });
  });

  await page.goto('/');
  await page.getByRole('button', { name: 'Open engagement' }).click();
  await page.getByRole('button', { name: 'Run next analysis' }).click();
  await expect(page.getByText(/awaiting input/i)).toBeVisible();

  await page.getByPlaceholder(/Type your answer/i).fill('Increase 90-day primary deposit retention from 64% to 70% within 90 days.');
  await page.getByRole('button', { name: 'Send direction' }).click();
  await expect(page.getByText(/human[_ ]approval/i)).toBeVisible();
  await page.getByRole('button', { name: 'Approve workflow' }).click();
  await expect(page.getByText(/completed/i)).toBeVisible();
  await expect(page.getByText('Harbor National retention plan')).toBeVisible();
});

test('navigates the consulting workflow views', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Projects/ }).click();
  await expect(page.getByRole('heading', { name: 'Project workspace' })).toBeVisible();
  await page.getByRole('button', { name: /Client intake/ }).click();
  await expect(page.getByRole('heading', { name: 'Client intake' })).toBeVisible();
  await page.getByRole('button', { name: /Audit trail/ }).click();
  await expect(page.getByRole('heading', { name: 'Audit trail' })).toBeVisible();
});
