'use client';

import type { AuditEvent, EvidenceItem, Project, WorkflowOutputs, WorkflowStatus } from '../lib/api';

export type WorkspaceView = 'overview' | 'projects' | 'intake' | 'scope' | 'evidence' | 'review' | 'deliverables' | 'audit';

type Props = {
  view: Exclude<WorkspaceView, 'overview'>;
  project: Project | null;
  workflow: WorkflowStatus | null;
  outputs: WorkflowOutputs | null;
  evidence: EvidenceItem[];
  auditEvents: AuditEvent[];
};

const viewCopy: Record<Props['view'], { eyebrow: string; title: string; description: string }> = {
  projects: { eyebrow: 'ENGAGEMENTS', title: 'Project workspace', description: 'Keep the client portfolio, ownership, and current stage visible at a glance.' },
  intake: { eyebrow: 'DISCOVERY', title: 'Client intake', description: 'Turn the first conversation into a decision-ready consulting brief.' },
  scope: { eyebrow: 'WORKPLAN', title: 'Scope and work plan', description: 'Make the work explicit before the agents begin testing hypotheses.' },
  evidence: { eyebrow: 'EVIDENCE LEDGER', title: 'Evidence review', description: 'Separate client-provided facts, retrieved sources, and working assumptions.' },
  review: { eyebrow: 'MANAGER GATE', title: 'Recommendation review', description: 'Review confidence, warnings, and approval state before anything reaches the client.' },
  deliverables: { eyebrow: 'DELIVERY', title: 'Executive deliverables', description: 'Package approved analysis into material a banking leadership team can use.' },
  audit: { eyebrow: 'CONTROL ROOM', title: 'Audit trail', description: 'Track the actions that changed the engagement and the decisions that need an owner.' },
};

export function WorkspaceView({ view, project, workflow, outputs, evidence, auditEvents }: Props) {
  const copy = viewCopy[view];
  return <div className="mt-7">
    <div className="mb-6 border-b border-meridian-line pb-5">
      <div className="text-[10px] font-extrabold uppercase tracking-[.14em] text-meridian-cyan">{copy.eyebrow}</div>
      <h1 className="mt-2 text-[30px] font-extrabold tracking-[-.045em]">{copy.title}</h1>
      <p className="mt-2 text-sm text-meridian-muted">{copy.description}</p>
    </div>
    {view === 'projects' && <Projects project={project} />}
    {view === 'intake' && <Intake project={project} />}
    {view === 'scope' && <Scope project={project} workflow={workflow} />}
    {view === 'evidence' && <Evidence evidence={evidence} />}
    {view === 'review' && <Review workflow={workflow} outputs={outputs} />}
    {view === 'deliverables' && <Deliverables outputs={outputs} workflow={workflow} />}
    {view === 'audit' && <Audit project={project} workflow={workflow} auditEvents={auditEvents} />}
  </div>;
}

function Card({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return <section className={`rounded-lg border border-meridian-line bg-white p-5 ${className}`}><h2 className="mb-4 text-sm font-extrabold">{title}</h2>{children}</section>;
}

function Label({ children }: { children: React.ReactNode }) { return <div className="mb-1 text-[10px] font-extrabold uppercase tracking-[.12em] text-meridian-muted">{children}</div>; }
function EmptyState({ title, detail }: { title: string; detail: string }) { return <div className="rounded-md border border-dashed border-[#cbd9dd] bg-[#f8fbfb] p-8 text-center"><div className="text-sm font-extrabold">{title}</div><p className="mx-auto mt-2 max-w-md text-xs leading-relaxed text-meridian-muted">{detail}</p></div>; }

function Projects({ project }: { project: Project | null }) {
  return <div className="grid gap-4 md:grid-cols-[1.2fr_.8fr]">
    <Card title="Active engagements">
      {project ? <div className="rounded-md border border-meridian-line p-4"><div className="flex items-start justify-between gap-4"><div><div className="text-[10px] font-extrabold uppercase tracking-[.12em] text-meridian-cyan">BANKING · ACTIVE</div><h3 className="mt-2 text-base font-extrabold">{project.name}</h3><p className="mt-1 text-xs text-meridian-muted">{project.client_name} · {project.industry}</p></div><span className="rounded bg-[#e8f5f3] px-2 py-1 text-[10px] font-extrabold uppercase text-[#287f82]">{project.status}</span></div><div className="mt-5 grid grid-cols-2 gap-3 border-t border-meridian-line pt-4"><div><Label>Objectives</Label><div className="text-sm font-bold">{project.objectives.length}</div></div><div><Label>Team access</Label><div className="text-sm font-bold">{project.member_ids?.length || 0} members</div></div></div></div> : <EmptyState title="No engagement is open" detail="Use the Open engagement action in the header to connect a client workspace." />}
    </Card>
    <Card title="Workspace signals"><div className="space-y-4"><Signal label="Current client" value={project?.client_name || 'Not connected'} /><Signal label="Industry" value={project?.industry || '—'} /><Signal label="Control posture" value="Human approval required" /></div></Card>
  </div>;
}

function Intake({ project }: { project: Project | null }) {
  return <div className="grid gap-4 md:grid-cols-2"><Card title="Client context"><div className="space-y-4"><Field label="Client" value={project?.client_name || 'Harbor National Bank'} /><Field label="Industry" value={project?.industry || 'Banking'} /><Field label="Engagement question" value="How can the bank improve primary deposit retention without broad pricing concessions?" multiline /></div></Card><Card title="What success looks like"><div className="space-y-3"><Check text="Improve primary deposit retention" /><Check text="Protect branch relationships" /><Check text="Create a decision within 90 days" /><div className="mt-5 rounded-md bg-[#fff9e7] p-3 text-xs leading-relaxed text-[#766343]">Working assumptions remain illustrative until the client confirms the target metric, time horizon, and scope.</div></div></Card></div>;
}

function Scope({ project, workflow }: { project: Project | null; workflow: WorkflowStatus | null }) {
  return <div className="space-y-4"><Card title="Agreed workstreams"><div className="grid gap-3 md:grid-cols-3"><Workstream number="01" title="Diagnose" detail="Map the retention gap by acquisition path and customer cohort." /><Workstream number="02" title="Test" detail="Compare onboarding and incentive interventions against available evidence." /><Workstream number="03" title="Decide" detail="Prioritize an executable 90-day commercial discipline program." /></div></Card><div className="grid gap-4 md:grid-cols-2"><Card title="Guardrails"><Check text="Use project-scoped evidence only" /><Check text="Label illustrative or simulated data" /><Check text="Pause for manager approval before delivery" /></Card><Card title="Execution status"><Signal label="Engagement" value={project?.status || 'Not started'} /><Signal label="Agent workflow" value={workflow?.current_stage || 'Not started'} /><Signal label="Decision window" value="90 days" /></Card></div></div>;
}

function Evidence({ evidence }: { evidence: EvidenceItem[] }) {
  return <Card title="Project-scoped evidence"><div className="mb-4 flex items-center justify-between text-xs text-meridian-muted"><span>{evidence.length ? `${evidence.length} retrieved items` : 'No live evidence retrieved'}</span><span className="rounded bg-[#fff5d5] px-2 py-1 text-[10px] font-extrabold text-[#876320]">UNTRUSTED INPUTS</span></div>{evidence.length ? <div className="space-y-3">{evidence.map((item, index) => <div className="rounded-md border border-meridian-line p-4" key={item.evidence_id || index}><div className="flex items-center justify-between gap-3"><strong className="text-sm">{item.title || item.source || `Evidence ${index + 1}`}</strong><span className="text-[10px] font-extrabold uppercase text-meridian-cyan">{item.source_type || 'evidence'}</span></div><p className="mt-2 text-xs leading-relaxed text-meridian-muted">{item.content || 'No content returned.'}</p>{item.injection_suspected && <div className="mt-2 text-[10px] font-extrabold text-meridian-orange">Flagged for prompt-injection review</div>}</div>)}</div> : <EmptyState title="Evidence ledger is empty" detail="Retrieve evidence from the dashboard after opening an engagement. Retrieved content is treated as evidence, never as instructions." />}</Card>;
}

function Review({ workflow, outputs }: { workflow: WorkflowStatus | null; outputs: WorkflowOutputs | null }) {
  return <div className="grid gap-4 md:grid-cols-[1.1fr_.9fr]"><Card title="Manager decision"><div className="rounded-md bg-[#fff9e7] p-4"><Label>Current gate</Label><div className="mt-1 text-lg font-extrabold">{workflow?.status === 'completed' ? 'Approved and completed' : workflow?.status === 'awaiting_input' ? 'Awaiting human input' : 'Not yet reached'}</div><p className="mt-2 text-xs leading-relaxed text-[#766343]">Consequential recommendations remain paused until an Engagement Manager reviews the evidence, assumptions, and warnings.</p></div></Card><Card title="Quality signals"><Signal label="Workflow stage" value={workflow?.current_stage || 'Not started'} /><Signal label="QA iterations" value={String(workflow?.qa_iterations ?? '—')} /><Signal label="Warnings" value={outputs?.warnings?.length ? `${outputs.warnings.length} open` : 'None returned'} /></Card></div>;
}

function Deliverables({ outputs, workflow }: { outputs: WorkflowOutputs | null; workflow: WorkflowStatus | null }) {
  if (!outputs?.report_markdown) return <EmptyState title="No approved deliverable yet" detail={workflow?.status === 'completed' ? 'The workflow completed without a report payload.' : 'Complete the workflow and manager review gate before generating client-ready material.'} />;
  return <div className="space-y-4"><Card title="Executive report"><div className="rounded-md bg-[#f5f8f9] p-5"><div className="mb-3 flex items-center justify-between"><span className="text-[10px] font-extrabold uppercase tracking-[.12em] text-meridian-cyan">REPORT OUTPUT</span><span className="rounded bg-[#e8f5f3] px-2 py-1 text-[10px] font-extrabold text-[#287f82]">WORKFLOW COMPLETE</span></div><div className="whitespace-pre-wrap text-sm leading-relaxed">{outputs.report_markdown}</div></div></Card><div className="grid gap-4 md:grid-cols-3"><Card title="Executive summary"><p className="text-xs leading-relaxed text-meridian-muted">Ready for manager review and client editing.</p></Card><Card title="Presentation outline"><p className="text-xs leading-relaxed text-meridian-muted">Source-backed storyline can be assembled from the approved report.</p></Card><Card title="Evidence appendix"><p className="text-xs leading-relaxed text-meridian-muted">Citations and warnings remain attached to the workflow record.</p></Card></div></div>;
}

function Audit({ project, workflow, auditEvents }: { project: Project | null; workflow: WorkflowStatus | null; auditEvents: AuditEvent[] }) {
  const events = auditEvents.length ? auditEvents.map(event => ({ label: event.action, detail: `${event.resource_type}${event.resource_id ? ` · ${event.resource_id.slice(0, 8)}` : ''}`, time: event.created_at })) : [
    project && { label: 'Engagement context', detail: 'Project workspace connected', time: project.created_at },
    workflow && { label: 'Agent workflow', detail: `Execution ${workflow.execution_id.slice(0, 8)} · ${workflow.status}`, time: workflow.updated_at },
  ].filter(Boolean) as Array<{ label: string; detail: string; time: string }>;
  return <Card title="Recent activity">{events.length ? <div className="divide-y divide-meridian-line">{events.map((event, index) => <div className="flex items-center justify-between gap-4 py-4" key={`${event.label}-${event.time}-${index}`}><div><div className="text-xs font-extrabold">{event.label}</div><div className="mt-1 text-xs text-meridian-muted">{event.detail}</div></div><time className="text-[10px] text-meridian-muted">{new Date(event.time).toLocaleString()}</time></div>)}</div> : <EmptyState title="No audit events in this workspace" detail="Project and workflow events will appear here as the engagement progresses." />}</Card>;
}

function Signal({ label, value }: { label: string; value: string }) { return <div className="border-b border-meridian-line pb-3 last:border-0 last:pb-0"><Label>{label}</Label><div className="text-sm font-bold">{value}</div></div>; }
function Field({ label, value, multiline = false }: { label: string; value: string; multiline?: boolean }) { return <div><Label>{label}</Label><div className={`rounded-md border border-meridian-line bg-[#f8fbfb] px-3 py-2 text-xs leading-relaxed ${multiline ? 'min-h-16' : ''}`}>{value}</div></div>; }
function Check({ text }: { text: string }) { return <div className="flex items-start gap-2 text-xs"><span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-[#e8f5f3] text-[10px] font-black text-[#287f82]">✓</span><span>{text}</span></div>; }
function Workstream({ number, title, detail }: { number: string; title: string; detail: string }) { return <div className="rounded-md border border-meridian-line p-4"><div className="text-[10px] font-extrabold text-meridian-cyan">{number}</div><div className="mt-2 text-sm font-extrabold">{title}</div><p className="mt-2 text-xs leading-relaxed text-meridian-muted">{detail}</p></div>; }
