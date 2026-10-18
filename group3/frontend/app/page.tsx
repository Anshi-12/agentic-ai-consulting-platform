'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import toast from 'react-hot-toast';
import {
  approveProject, connectWorkflowSocket, createProject, demoLogin, getAuditEvents,
  getWorkflowOutputs, getWorkflowStatus, resumeWorkflow, searchEvidence,
  startAgentWorkflow, uploadDocument,
} from '../lib/api';
import { useWorkspace } from '../lib/workspace';
import { WorkspaceView as WorkspaceViewPanel, type WorkspaceView } from './workspace-views';

const stages = [
  ['Discovery', 'Clarify the decision the bank needs to make.'],
  ['Evidence review', 'Read project materials and retrieved sources.'],
  ['Diagnostics', 'Test retention, onboarding, and commercial hypotheses.'],
  ['Strategy', 'Shape options, trade-offs, and implementation choices.'],
  ['Quality review', 'Check assumptions, warnings, and evidence coverage.'],
  ['Executive brief', 'Prepare the decision-ready output for review.'],
] as const;
const queue = [
  ['A', 'Where does retention break?', 'Compare 30/60/90-day balance behavior by acquisition path', 'READY'],
  ['B', 'Does pricing pay back?', 'Run incentive sensitivity by customer segment', 'QUEUED'],
  ['C', 'What can branches own?', 'Validate the handoff with frontline interviews', 'QUEUED'],
];
const fallbackSources = [
  ['Customer cohort extract', '90-day balance behavior by acquisition path', 'PROJECT DATA'],
  ['Branch leader interviews', 'Four interviews; themes coded by analyst', 'CLIENT MATERIAL'],
  ['Digital funnel report', 'Drop-off concentrated at product selection', 'PROJECT REPORT'],
];

export default function Page() {
  const workspace = useWorkspace();
  const { project, workflow, outputs, evidence, token } = workspace;
  const [view, setView] = useState<WorkspaceView>('overview');
  const [busy, setBusy] = useState(false);
  const [evidenceBusy, setEvidenceBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [approved, setApproved] = useState(false);
  const [question, setQuestion] = useState('');
  const [auditEvents, setAuditEvents] = useState<import('../lib/api').AuditEvent[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const interruptType = workflow?.pending_interrupt?.type as string | undefined;
  const questions = Array.isArray(workflow?.pending_interrupt?.questions) ? workflow.pending_interrupt.questions as string[] : [];

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocket | undefined;
    const executionId = workflow?.execution_id;
    if (!executionId || !project || !token) return;
    const refresh = async () => {
      try {
        const status = await getWorkflowStatus(executionId);
        workspace.setWorkflow(status);
        if (['completed', 'failed', 'rejected', 'cancelled'].includes(status.status)) {
          workspace.setOutputs(await getWorkflowOutputs(executionId));
          if (status.status === 'completed') toast.success('The consulting brief is ready for review');
          if (timer) clearInterval(timer);
        }
      } catch { /* preserve the last known state while the local API restarts */ }
    };
    socket = connectWorkflowSocket(project.id, executionId, token, () => { void refresh(); });
    void refresh();
    timer = setInterval(() => void refresh(), 5000);
    return () => { if (timer) clearInterval(timer); socket?.close(); };
  }, [workflow?.execution_id, project?.id, token]);

  useEffect(() => {
    if (view === 'audit' && token) getAuditEvents().then(setAuditEvents).catch(() => toast.error('Audit service is unavailable'));
  }, [view, token]);

  async function openEngagement() {
    setBusy(true);
    try {
      const auth = await demoLogin();
      workspace.setToken(auth.access_token);
      const currentProject = project || await createProject({ name: 'Harbor National deposit growth diagnostic', client_name: 'Harbor National Bank', industry: 'Banking', objectives: ['Improve primary deposit retention'], constraints: ['90-day decision window', 'Protect branch relationships'] });
      workspace.setProject(currentProject);
      toast.success('Engagement connected to the central API');
    } catch { toast.error('Central API is not available'); }
    finally { setBusy(false); }
  }

  async function runAnalysis() {
    setBusy(true);
    try {
      if (!project) await openEngagement();
      const currentProject = useWorkspace.getState().project;
      if (!currentProject) return;
      const result = await startAgentWorkflow(currentProject.id);
      workspace.setWorkflow(result); workspace.setOutputs(null);
      toast.success('The consulting team has started working');
    } catch { toast.error('Workflow could not be started'); }
    finally { setBusy(false); }
  }

  async function submitClarification() {
    if (!workflow?.execution_id || !question.trim()) { toast.error('Add a short answer before continuing'); return; }
    setBusy(true);
    try {
      workspace.setWorkflow(await resumeWorkflow(workflow.execution_id, { type: 'clarification', answers: { question: question.trim() } }));
      setQuestion(''); toast.success('Your direction was sent to the consulting team');
    } catch { toast.error('The clarification could not be submitted'); }
    finally { setBusy(false); }
  }

  async function approve() {
    if (!project) { toast.error('Open the engagement first'); return; }
    setBusy(true);
    try {
      if (workflow?.execution_id && workflow.status === 'awaiting_input') {
        if (interruptType === 'clarification' || workflow.current_stage === 'clarification') await submitClarification();
        else {
          workspace.setWorkflow(await resumeWorkflow(workflow.execution_id, { type: 'approval', decision: 'approve', reviewer: 'Morgan Manager', comments: 'Evidence and assumptions reviewed by Engagement Manager.' }));
          toast.success('Recommendation approved; preparing the executive brief');
        }
      } else { await approveProject(project.id); setApproved(true); toast.success('Project approved by Engagement Manager'); }
    } catch { toast.error('The review action failed'); }
    finally { setBusy(false); }
  }

  async function refreshEvidence() {
    const currentProject = useWorkspace.getState().project;
    if (!currentProject) { toast.error('Open the engagement first'); return; }
    setEvidenceBusy(true);
    try { const result = await searchEvidence(currentProject.id, 'deposit retention, customer acquisition, and bank growth risks'); workspace.setEvidence(result.evidence || []); toast.success(`${result.evidence?.length || 0} evidence items retrieved`); }
    catch { toast.error('Evidence service is unavailable — no results were fabricated'); }
    finally { setEvidenceBusy(false); }
  }

  async function handleUpload(file?: File) {
    const currentProject = useWorkspace.getState().project;
    if (!currentProject || !file) return;
    setUploadBusy(true);
    try { await uploadDocument(currentProject.id, file); toast.success(`${file.name} uploaded for indexing`); await refreshEvidence(); }
    catch { toast.error('Document upload failed — it was not marked as indexed'); }
    finally { setUploadBusy(false); }
  }

  return <div className="min-h-screen bg-[#f6f8fa] text-meridian-ink"><div className="grid min-h-screen grid-cols-[235px_1fr] max-[950px]:block">
    <aside className="relative bg-meridian-navy px-[18px] py-6 text-slate-200 max-[950px]:hidden"><div className="flex items-center gap-2 text-xl font-extrabold tracking-tight"><div className="grid h-7 w-7 place-items-center rounded-md bg-meridian-lime text-meridian-navy">M</div> meridian</div><div className="ml-9 mt-1 text-[9px] font-bold tracking-[.13em] text-[#89a6b4]">ADVISORY OPERATING SYSTEM</div><nav className="mt-10"><div className="mb-2 px-2 text-[10px] font-extrabold uppercase tracking-[.13em] text-[#7897a7]">Workspace</div><NavButton active={view === 'overview'} onClick={() => setView('overview')}>✦ &nbsp; Copilot</NavButton><NavButton active={view === 'projects'} onClick={() => setView('projects')}>▣ &nbsp; Projects</NavButton><NavButton active={view === 'intake'} onClick={() => setView('intake')}>◎ &nbsp; Client intake</NavButton><NavButton active={view === 'scope'} onClick={() => setView('scope')}>◒ &nbsp; Scope and work plan</NavButton><NavButton active={view === 'evidence'} onClick={() => setView('evidence')}>▤ &nbsp; Evidence ledger</NavButton><div className="mb-2 mt-5 px-2 text-[10px] font-extrabold uppercase tracking-[.13em] text-[#7897a7]">Delivery</div><NavButton active={view === 'review'} onClick={() => setView('review')}>↗ &nbsp; Review queue</NavButton><NavButton active={view === 'deliverables'} onClick={() => setView('deliverables')}>□ &nbsp; Executive brief</NavButton><NavButton active={view === 'audit'} onClick={() => setView('audit')}>◷ &nbsp; Audit trail</NavButton></nav><div className="absolute bottom-5 left-5 right-5 border-t border-[#2a4a5e] pt-3 text-[11px] leading-relaxed text-[#89a6b4]">CURRENT CLIENT<br /><b className="text-slate-100">Harbor National Bank</b><br />Retail deposits · Q4 diagnostic</div></aside>
    <main className="w-full max-w-[1450px] px-8 py-6 max-[950px]:px-4"><div className="flex items-center justify-between border-b border-meridian-line pb-4"><div className="text-[11px] text-meridian-muted">PROJECTS / <b className="text-meridian-ink">HARBOR NATIONAL BANK</b> / {view === 'overview' ? 'COPILOT' : view.toUpperCase()}</div><div className="flex items-center gap-2"><span className="rounded-full border border-[#cce8e2] bg-[#edf9f6] px-2.5 py-1.5 text-[10px] font-extrabold text-[#147b77]">CONTROLLED WORKSPACE</span><button disabled={busy} className="rounded border border-meridian-blue bg-white px-3 py-2 text-xs font-extrabold text-meridian-blue disabled:opacity-50" onClick={openEngagement}>{project ? 'Engagement open' : 'Open engagement'}</button><button className="rounded bg-meridian-blue px-3 py-2 text-xs font-extrabold text-white" onClick={() => toast('Brief link copied')}>Share brief</button></div></div>{view === 'overview' ? <Overview project={project} workflow={workflow} outputs={outputs} evidence={evidence} busy={busy} evidenceBusy={evidenceBusy} uploadBusy={uploadBusy} approved={approved} fileInput={fileInput} question={question} questions={questions} onQuestion={setQuestion} onRun={runAnalysis} onSubmit={submitClarification} onApprove={approve} onRetrieve={refreshEvidence} onUpload={handleUpload} /> : <WorkspaceViewPanel view={view} project={project} workflow={workflow} outputs={outputs} evidence={evidence} auditEvents={auditEvents} />}</main>
  </div></div>;
}

function NavButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) { return <button onClick={onClick} className={`mb-1 block w-full rounded-md px-2 py-3 text-left text-sm ${active ? 'border-l-[3px] border-meridian-lime bg-[#123e59] font-bold text-white' : 'text-[#abc1ca] hover:bg-[#123e59] hover:text-white'}`}>{children}</button>; }

type OverviewProps = { project: ReturnType<typeof useWorkspace.getState>['project']; workflow: ReturnType<typeof useWorkspace.getState>['workflow']; outputs: ReturnType<typeof useWorkspace.getState>['outputs']; evidence: ReturnType<typeof useWorkspace.getState>['evidence']; busy: boolean; evidenceBusy: boolean; uploadBusy: boolean; approved: boolean; fileInput: React.RefObject<HTMLInputElement>; question: string; questions: string[]; onQuestion: (value: string) => void; onRun: () => void; onSubmit: () => void; onApprove: () => void; onRetrieve: () => void; onUpload: (file?: File) => void };

function Overview({ project, workflow, outputs, evidence, busy, evidenceBusy, uploadBusy, approved, fileInput, question, questions, onQuestion, onRun, onSubmit, onApprove, onRetrieve, onUpload }: OverviewProps) {
  const isClarification = workflow?.status === 'awaiting_input' && (workflow.current_stage === 'clarification' || workflow.pending_interrupt?.type === 'clarification');
  const isApproval = workflow?.status === 'awaiting_input' && !isClarification;
  return <div><div className="my-7 flex items-end justify-between gap-5 max-[700px]:block"><div><div className="text-[10px] font-extrabold uppercase tracking-[.15em] text-meridian-cyan">CONSULTING COPILOT</div><h1 className="m-0 mt-2 text-[32px] font-extrabold leading-tight tracking-[-.045em]">Let’s work through the retention gap.</h1><p className="mt-2 text-meridian-muted">Ask a question, review the evidence, and guide the next decision.</p></div><div className="rounded-full bg-[#e9f7f4] px-3 py-2 text-[10px] font-extrabold uppercase text-[#147b77]">{workflow ? workflow.status.replace('_', ' ') : 'Ready for direction'}</div></div>
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_310px]"><section className="overflow-hidden rounded-xl border border-meridian-line bg-white shadow-[0_14px_40px_rgba(18,42,64,.06)]"><div className="border-b border-meridian-line px-6 py-5"><div className="text-[10px] font-extrabold uppercase tracking-[.14em] text-meridian-cyan">MERIDIAN CONSULTING COPILOT</div><div className="mt-1 text-base font-extrabold">Harbor National Bank · Deposit growth diagnostic</div></div><div className="space-y-5 px-6 py-6"><ChatMessage label="Meridian consulting copilot" badge="M">I’ve started a diagnostic on Harbor National’s 90-day deposit retention gap. I’m comparing customer cohorts, branch handoffs, and digital funnel performance.</ChatMessage><AgentActivity workflow={workflow} /><ChatMessage label="You" badge="MM" user>Focus the next pass on measurable commercial impact and branch execution.</ChatMessage><ChatMessage label="Meridian consulting copilot" badge="M">Understood. I’ll prioritize impact sizing, implementation effort, and branch-level dependencies. I’ll pause before approval so you can review the evidence and assumptions.</ChatMessage><div className="border-t border-meridian-line pt-5">{isClarification && <div className="mb-3 rounded-lg border border-[#ead7a8] bg-[#fff9e7] p-3 text-xs leading-relaxed text-[#765f2d]"><b>One clarification will improve the analysis:</b><br />{questions.join(' ') || 'What metric and target define success for this banking engagement?'}</div>}<textarea value={question} onChange={event => onQuestion(event.target.value)} placeholder={isClarification ? 'Type your answer for the consulting team…' : 'Ask a follow-up or clarify the next analysis…'} className="min-h-[86px] w-full resize-y rounded-xl border border-[#cfdbe2] bg-white p-3 text-sm outline-none transition focus:border-meridian-cyan focus:ring-4 focus:ring-[#e4f6f3]" /><div className="mt-2 flex items-center justify-between gap-3"><span className="text-[11px] text-meridian-muted">Grounded in project evidence · human judgment remains in the loop</span><div className="flex gap-2"><button className="rounded-lg border border-meridian-line px-3 py-2 text-xs font-extrabold" onClick={() => onQuestion('')}>Clear</button><button disabled={busy || (isClarification && !question.trim())} className="rounded-lg bg-meridian-cyan px-3 py-2 text-xs font-extrabold text-white disabled:opacity-50" onClick={isClarification ? onSubmit : onRun}>{isClarification ? 'Send direction' : 'Run next analysis'}</button></div></div></div><div className="flex items-center justify-between gap-4 rounded-xl border border-[#f0d99b] bg-[#fff9e7] p-4 max-[650px]:block"><div><strong className="text-[13px] text-[#765a22]">{isApproval ? 'Manager review required' : 'Human review gate'}</strong><p className="m-0 mt-1 text-[11px] text-[#766343]">{isApproval ? 'Review the evidence, assumptions, and warnings before approving.' : 'Consequential recommendations remain paused until an authorized manager reviews them.'}</p></div><button disabled={busy || approved} className="rounded-lg bg-meridian-orange px-3 py-2 text-xs font-extrabold text-white disabled:bg-[#23856c] max-[650px]:mt-3" onClick={onApprove}>{approved || workflow?.status === 'completed' ? 'Approved ✓' : isApproval ? 'Approve workflow' : 'Review when ready'}</button></div></div></section>
      <aside className="space-y-4"><section className="rounded-xl border border-meridian-line bg-white p-5"><div className="flex items-center justify-between"><h2 className="m-0 text-sm font-extrabold">Evidence at a glance</h2><span className="text-[9px] font-extrabold uppercase text-meridian-muted">Project sources</span></div><div className="mt-4 space-y-3"><Metric label="90-day retention" value="64%" note="Opportunity identified" /><Metric label="Digital completion" value="71%" note="4.2 pts below prior" orange /></div></section><section className="rounded-xl border border-meridian-line bg-white p-5"><div className="mb-3 flex items-center justify-between"><h2 className="m-0 text-sm font-extrabold">Evidence ledger</h2><span className="text-[9px] font-extrabold uppercase text-meridian-muted">{evidence.length ? 'Retrieved' : 'Awaiting retrieval'}</span></div><div className="mb-3 flex gap-2"><input ref={fileInput} type="file" accept=".pdf,.docx,.csv,.xlsx,.txt,.json" className="hidden" onChange={event => onUpload(event.target.files?.[0])} /><button disabled={uploadBusy || !project} onClick={() => fileInput.current?.click()} className="rounded border border-meridian-line px-2 py-1.5 text-[10px] font-extrabold text-meridian-blue disabled:opacity-50">{uploadBusy ? 'Uploading…' : 'Upload source'}</button><button disabled={evidenceBusy || !project} onClick={onRetrieve} className="rounded border border-meridian-line px-2 py-1.5 text-[10px] font-extrabold text-meridian-blue disabled:opacity-50">{evidenceBusy ? 'Searching…' : 'Retrieve evidence'}</button></div>{(evidence.length ? evidence : fallbackSources).map((item, index) => { const live = item as import('../lib/api').EvidenceItem; const fallback = item as string[]; const title = evidence.length ? live.title || live.source || `Evidence ${index + 1}` : fallback[0]; const detail = evidence.length ? live.content || 'Retrieved from the project-scoped evidence service.' : fallback[1]; const tag = evidence.length ? live.source_type || 'EVIDENCE' : fallback[2]; return <div className="border-t border-meridian-line py-3 first:border-0" key={title}><div className="text-xs font-extrabold">{title}</div><p className="m-0 mt-1 text-[10px] leading-relaxed text-meridian-muted">{detail}</p><span className="mt-2 inline-block rounded bg-[#e8f5f3] px-1.5 py-1 text-[9px] font-extrabold text-[#287f82]">{tag}</span></div>; })}</section></aside></div>
    <div className="mt-5 rounded-xl border border-meridian-line bg-white p-5"><div className="mb-3 flex items-center justify-between"><h2 className="m-0 text-sm font-extrabold">Open questions</h2><span className="text-[10px] font-extrabold uppercase text-meridian-muted">3 work items</span></div>{queue.map(([letter, title, desc, status]) => <div key={letter} className="grid grid-cols-[25px_1fr_auto] items-center gap-3 border-t border-meridian-line py-3 first:border-0"><div className="grid h-6 w-6 place-items-center rounded bg-[#e8f4f3] text-xs font-extrabold text-[#168983]">{letter}</div><div><b className="text-xs">{title}</b><p className="m-0 mt-1 text-[10px] text-meridian-muted">{desc}</p></div><span className={`text-[9px] font-black ${status === 'READY' ? 'text-meridian-cyan' : 'text-[#c58b2a]'}`}>{status}</span></div>)}</div>{outputs?.report_markdown && <ReportOutput markdown={outputs.report_markdown} />}<div className="mt-5 flex justify-between border-t border-meridian-line pt-3 text-[10px] text-meridian-muted"><span>Evidence-backed workspace · human judgment remains in the loop</span><span className="max-[650px]:hidden">No unsupported market claims</span></div></div>;
}

function ReportOutput({ markdown }: { markdown: string }) {
  const clean = markdown.replace(/\\([#*_])/g, '$1').replace(/\r/g, '');
  const lines = clean.split('\n').map(line => line.trim()).filter(Boolean);
  return <div className="mt-5 rounded-xl border border-[#cce8e2] bg-[#f1fbf8] p-5"><div className="flex items-start justify-between gap-4"><div><div className="text-[10px] font-extrabold uppercase tracking-[.12em] text-meridian-cyan">LATEST REPORT OUTPUT</div><h2 className="mt-1 text-xl font-extrabold tracking-[-.03em]">Executive decision brief</h2></div><span className="rounded-full bg-white px-2.5 py-1 text-[9px] font-extrabold uppercase text-[#287f82]">Human approved</span></div><div className="mt-5 space-y-2 text-sm leading-relaxed text-[#314e52]">{lines.map((line, index) => { if (line.startsWith('# ')) return <h3 className="pt-3 text-lg font-extrabold text-meridian-ink" key={index}>{inlineReport(line.slice(2))}</h3>; if (line.startsWith('## ')) return <h4 className="pt-3 text-sm font-extrabold uppercase tracking-[.08em] text-[#1f5f62]" key={index}>{inlineReport(line.slice(3))}</h4>; if (/^\d+\.\s/.test(line)) return <div className="rounded-lg border border-[#dcebea] bg-white p-3" key={index}>{inlineReport(line.replace(/^\d+\.\s/, ''))}</div>; if (line.startsWith('- ')) return <div className="flex gap-2" key={index}><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-meridian-cyan" /><span>{inlineReport(line.slice(2))}</span></div>; if (line.startsWith('|')) return null; return <p className="m-0" key={index}>{inlineReport(line)}</p>; })}</div></div>;
}

function inlineReport(value: string): ReactNode { const parts = value.split(/(\*\*[^*]+\*\*|⚠)/g); return parts.map((part, index) => part.startsWith('**') && part.endsWith('**') ? <strong key={index}>{part.slice(2, -2)}</strong> : part === '⚠' ? <span className="font-extrabold text-meridian-orange" key={index}>⚠</span> : part); }

function ChatMessage({ label, badge, children, user = false }: { label: string; badge: string; children: ReactNode; user?: boolean }) { return <div className={`flex gap-3 ${user ? 'justify-end' : ''}`}><div className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg text-[11px] font-extrabold ${user ? 'order-2 bg-[#d8efed] text-[#168682]' : 'bg-meridian-navy text-white'}`}>{badge}</div><div className={`max-w-[78%] rounded-2xl px-4 py-3 text-[13px] leading-relaxed ${user ? 'rounded-tr-md bg-meridian-navy text-white' : 'rounded-tl-md bg-[#f3f7f9]'}`}><small className={`mb-1 block text-[10px] font-extrabold uppercase tracking-[.08em] ${user ? 'text-[#9fc1cc]' : 'text-meridian-muted'}`}>{label}</small>{children}</div></div>; }

function AgentActivity({ workflow }: { workflow: ReturnType<typeof useWorkspace.getState>['workflow'] }) {
  const stage = workflow?.current_stage || '';
  const stageIndex: Record<string, number> = {
    intake: 0, problem_framing: 0, clarification: 0,
    research: 1, research_task: 1, financial_analysis: 1,
    hypothesis_evaluation: 2,
    strategy: 3,
    qa: 4, human_approval: 4,
    synthesis: 5, finalize: 5, done: 5,
  };
  const index = workflow?.status === 'completed' ? 5 : (stageIndex[stage] ?? -1);
  return <div className="rounded-xl border border-[#dcebea] bg-[#f4fbfa] p-4"><div className="mb-4 flex items-center justify-between"><div><div className="text-sm font-extrabold text-[#1f5f62]">Consulting team activity</div><div className="mt-1 text-[11px] text-[#5b8c8d]">Progress reflects the latest stage reported by the workflow.</div></div><span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-extrabold uppercase text-[#287f82]">{workflow ? workflow.status.replace('_', ' ') : 'Not started'}</span></div><div className="grid gap-2.5">{stages.map(([name, detail], i) => { const complete = index > i; const current = index === i; return <div className={`grid grid-cols-[28px_1fr_auto] items-center gap-2.5 text-xs ${current ? 'font-extrabold text-meridian-ink' : 'text-[#527072]'}`} key={name}><div className={`grid h-6 w-6 place-items-center rounded-full text-[10px] font-extrabold ${complete ? 'bg-meridian-cyan text-white' : current ? 'border-2 border-meridian-orange bg-white text-meridian-orange' : 'bg-[#d9efec] text-meridian-cyan'}`}>{complete ? '✓' : i + 1}</div><div><div>{name}</div><div className="mt-0.5 text-[10px] font-normal text-[#779091]">{detail}</div></div><span className={`text-[9px] font-extrabold uppercase tracking-[.08em] ${current ? 'text-meridian-orange' : complete ? 'text-meridian-cyan' : 'text-[#91a8a9]'}`}>{current ? 'Working' : complete ? 'Complete' : 'Queued'}</span></div>; })}</div></div>;
}
function Metric({ label, value, note, orange = false }: { label: string; value: string; note: string; orange?: boolean }) { return <div className="rounded-lg border border-meridian-line bg-[#fbfcfd] p-3"><small className="block text-[10px] uppercase tracking-[.1em] text-meridian-muted">{label}</small><strong className="my-1 block text-[24px] font-extrabold tracking-[-.05em]">{value}</strong><footer className={`text-[10px] font-extrabold ${orange ? 'text-meridian-orange' : 'text-meridian-cyan'}`}>{note}</footer></div>; }
