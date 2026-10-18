'use client';

import { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import {
  approveProject,
  connectWorkflowSocket,
  createProject,
  demoLogin,
  getWorkflowOutputs,
  getWorkflowStatus,
  getAuditEvents,
  resumeWorkflow,
  searchEvidence,
  startAgentWorkflow,
  uploadDocument,
} from '../lib/api';
import { useWorkspace } from '../lib/workspace';
import { WorkspaceView as WorkspaceViewPanel, type WorkspaceView } from './workspace-views';

const queue = [
  ['A', 'Where does retention break?', 'Compare 30/60/90-day balance behavior by acquisition path', 'READY'],
  ['B', 'Does pricing pay back?', 'Run incentive sensitivity by customer segment', 'QUEUED'],
  ['C', 'What can branches own?', 'Validate the handoff with frontline interviews', 'QUEUED'],
];

const sources = [
  ['Customer cohort extract', '90-day balance behavior by acquisition path', 'DATASET · VERIFIED'],
  ['Branch leader interviews', 'Four interviews; themes coded by analyst', 'INTERVIEWS · REVIEWED'],
  ['Digital funnel report', 'Drop-off concentrated at product selection', 'REPORT · VERIFIED'],
];

export default function Page() {
  const { project, workflow, outputs, evidence, token, setProject, setWorkflow, setOutputs, setEvidence, setToken } = useWorkspace();
  const [view, setView] = useState<WorkspaceView>('overview');
  const [busy, setBusy] = useState(false);
  const [evidenceBusy, setEvidenceBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [approved, setApproved] = useState(false);
  const [auditEvents, setAuditEvents] = useState<import('../lib/api').AuditEvent[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const interruptType = workflow?.pending_interrupt?.type as string | undefined;
  const questions = Array.isArray(workflow?.pending_interrupt?.questions)
    ? (workflow.pending_interrupt.questions as string[])
    : [];

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocket | undefined;
    const executionId = workflow?.execution_id;
    if (!executionId || !project || !token) return;

    const refresh = async () => {
      try {
        const status = await getWorkflowStatus(executionId);
        setWorkflow(status);
        if (['completed', 'failed', 'rejected', 'cancelled'].includes(status.status)) {
          setOutputs(await getWorkflowOutputs(executionId));
          if (status.status === 'completed') toast.success('Workflow completed and outputs are ready');
          if (timer) clearInterval(timer);
        }
      } catch {
        // Polling is best-effort while the local API is restarted.
      }
    };

    socket = connectWorkflowSocket(project.id, executionId, token, refresh);
    refresh();
    timer = setInterval(refresh, 5000);
    return () => {
      if (timer) clearInterval(timer);
      socket?.close();
    };
  }, [workflow?.execution_id, project?.id, token, setWorkflow, setOutputs]);

  useEffect(() => {
    if (view !== 'audit' || !token) return;
    getAuditEvents().then(setAuditEvents).catch(() => toast.error('Audit service is unavailable'));
  }, [view, token]);

  async function openEngagement() {
    setBusy(true);
    try {
      const auth = await demoLogin();
      setToken(auth.access_token);
      const currentProject = project || await createProject({
        name: 'Harbor National deposit growth diagnostic',
        client_name: 'Harbor National Bank',
        industry: 'Banking',
        objectives: ['Improve primary deposit retention'],
        constraints: ['90-day decision window', 'Protect branch relationships'],
      });
      setProject(currentProject);
      toast.success('Engagement connected to the central API');
    } catch {
      toast.error('Central API is not available');
    } finally {
      setBusy(false);
    }
  }

  async function runAnalysis() {
    setBusy(true);
    try {
      if (!project) await openEngagement();
      const currentProject = useWorkspace.getState().project;
      if (currentProject) {
        const result = await startAgentWorkflow(currentProject.id);
        setWorkflow(result);
        setOutputs(null);
        toast.success('Agent workflow started');
      }
    } catch {
      toast.error('Workflow could not be started');
    } finally {
      setBusy(false);
    }
  }

  async function refreshEvidence() {
    const currentProject = useWorkspace.getState().project;
    if (!currentProject) {
      toast.error('Open the engagement first');
      return;
    }
    setEvidenceBusy(true);
    try {
      const result = await searchEvidence(currentProject.id, 'deposit retention, customer acquisition, and bank growth risks');
      setEvidence(result.evidence || []);
      toast.success(`${result.evidence?.length || 0} evidence items retrieved`);
    } catch {
      toast.error('Evidence service is unavailable — no results were fabricated');
    } finally {
      setEvidenceBusy(false);
    }
  }

  async function handleUpload(file?: File) {
    const currentProject = useWorkspace.getState().project;
    if (!currentProject || !file) return;
    setUploadBusy(true);
    try {
      await uploadDocument(currentProject.id, file);
      toast.success(`${file.name} uploaded for indexing`);
      await refreshEvidence();
    } catch {
      toast.error('Document upload failed — it was not marked as indexed');
    } finally {
      setUploadBusy(false);
    }
  }

  async function approve() {
    if (!project) {
      toast.error('Open the engagement first');
      return;
    }
    setBusy(true);
    try {
      if (workflow?.execution_id && workflow.status === 'awaiting_input') {
        const isClarification = interruptType === 'clarification' || workflow.current_stage === 'clarification';
        if (isClarification) {
          const answer = window.prompt(questions.join(' ') || 'Please provide the clarification needed by the consulting workflow.');
          if (!answer?.trim()) {
            toast.error('A clarification answer is required');
            return;
          }
          setWorkflow(await resumeWorkflow(workflow.execution_id, {
            type: 'clarification',
            answers: { question: answer.trim() },
          }));
          toast.success('Clarification submitted to the workflow');
        } else {
          setWorkflow(await resumeWorkflow(workflow.execution_id, {
            type: 'approval',
            decision: 'approve',
            reviewer: 'Morgan Manager',
            comments: 'Cohort definition reviewed by Engagement Manager.',
          }));
          toast.success('Workflow approval submitted');
        }
      } else {
        await approveProject(project.id);
        setApproved(true);
        toast.success('Project approved by Engagement Manager');
      }
    } catch (error) {
      const responseError = error as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
      const detail = responseError.response?.data?.detail;
      const message = typeof detail === 'string' ? detail : detail?.message || responseError.message || 'The workflow action failed';
      toast.error(message);
    } finally {
      setBusy(false);
    }
  }

  return <div className="min-h-screen bg-meridian-bg text-meridian-ink">
    <div className="grid min-h-screen grid-cols-[235px_1fr] max-[950px]:block">
      <aside className="relative bg-meridian-navy px-[18px] py-6 text-slate-200 max-[950px]:hidden">
        <div className="flex items-center gap-2 text-xl font-extrabold tracking-tight"><div className="grid h-7 w-7 place-items-center rounded-md bg-meridian-lime text-meridian-navy">M</div> meridian</div>
        <div className="ml-9 mt-1 text-[9px] font-bold tracking-[.13em] text-[#89a6b4]">ADVISORY OPERATING SYSTEM</div>
        <nav className="mt-10">
          <div className="mb-2 px-2 text-[10px] font-extrabold uppercase tracking-[.13em] text-[#7897a7]">Workspace</div>
          <NavButton active={view === 'overview'} onClick={() => setView('overview')}>▦ &nbsp; Client overview</NavButton>
          <NavButton active={view === 'projects'} onClick={() => setView('projects')}>▣ &nbsp; Projects</NavButton>
          <NavButton active={view === 'intake'} onClick={() => setView('intake')}>◎ &nbsp; Client intake</NavButton>
          <NavButton active={view === 'scope'} onClick={() => setView('scope')}>◒ &nbsp; Scope and work plan</NavButton>
          <NavButton active={view === 'evidence'} onClick={() => setView('evidence')}>▤ &nbsp; Evidence ledger</NavButton>
          <div className="mb-2 mt-5 px-2 text-[10px] font-extrabold uppercase tracking-[.13em] text-[#7897a7]">Delivery</div>
          <NavButton active={view === 'review'} onClick={() => setView('review')}>↗ &nbsp; Review queue</NavButton>
          <NavButton active={view === 'deliverables'} onClick={() => setView('deliverables')}>□ &nbsp; Executive brief</NavButton>
          <NavButton active={view === 'audit'} onClick={() => setView('audit')}>◷ &nbsp; Audit trail</NavButton>
        </nav>
        <div className="absolute bottom-5 left-5 right-5 border-t border-[#2a4a5e] pt-3 text-[11px] leading-relaxed text-[#89a6b4]">CURRENT CLIENT<br /><b className="text-slate-100">Harbor National Bank</b><br />Retail deposits · Q4 diagnostic</div>
      </aside>

      <main className="w-full max-w-[1450px] px-8 py-6 max-[950px]:px-4">
        <div className="flex items-center justify-between border-b border-meridian-line pb-4">
          <div className="text-[11px] text-meridian-muted">CLIENTS / <b className="text-meridian-ink">HARBOR NATIONAL</b> / {view.toUpperCase()}</div>
          <div className="flex items-center gap-2"><span className="rounded border border-[#efd99d] bg-[#fff5d5] px-2 py-1.5 text-[10px] font-extrabold text-[#876320]">ILLUSTRATIVE DATA</span><button disabled={busy} className="rounded border border-meridian-blue bg-white px-3 py-2 text-xs font-extrabold text-meridian-blue disabled:opacity-50" onClick={openEngagement}>{project ? 'Engagement open' : 'Open engagement'}</button><button className="rounded border border-meridian-line bg-white px-3 py-2 text-xs font-extrabold" onClick={() => toast('Review link copied')}>Share</button><button className="rounded bg-meridian-blue px-3 py-2 text-xs font-extrabold text-white" onClick={() => toast('Brief export queued')}>Export brief</button></div>
        </div>

        {view === 'overview' ? <Overview project={project} workflow={workflow} outputs={outputs} evidence={evidence} busy={busy} evidenceBusy={evidenceBusy} uploadBusy={uploadBusy} approved={approved} fileInput={fileInput} onRun={runAnalysis} onApprove={approve} onRetrieve={refreshEvidence} onUpload={handleUpload} /> : <WorkspaceViewPanel view={view} project={project} workflow={workflow} outputs={outputs} evidence={evidence} auditEvents={auditEvents} />}
      </main>
    </div>
  </div>;
}

function NavButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button onClick={onClick} className={`mb-1 block w-full rounded-md px-2 py-3 text-left text-sm ${active ? 'border-l-[3px] border-meridian-lime bg-[#123e59] font-bold text-white' : 'text-[#abc1ca] hover:bg-[#123e59] hover:text-white'}`}>{children}</button>;
}

function Overview({ project, workflow, outputs, evidence, busy, evidenceBusy, uploadBusy, approved, fileInput, onRun, onApprove, onRetrieve, onUpload }: { project: ReturnType<typeof useWorkspace.getState>['project']; workflow: ReturnType<typeof useWorkspace.getState>['workflow']; outputs: ReturnType<typeof useWorkspace.getState>['outputs']; evidence: ReturnType<typeof useWorkspace.getState>['evidence']; busy: boolean; evidenceBusy: boolean; uploadBusy: boolean; approved: boolean; fileInput: React.RefObject<HTMLInputElement>; onRun: () => void; onApprove: () => void; onRetrieve: () => void; onUpload: (file?: File) => void }) {
  return <div>
    <div className="my-7 flex items-end justify-between max-[700px]:block"><div><h1 className="m-0 text-[32px] font-extrabold leading-tight tracking-[-.045em]">Deposit growth diagnostic</h1><p className="mt-2 text-meridian-muted">The numbers are moving. Now we need to understand why.</p></div><div className="text-right text-[11px] text-meridian-muted max-[700px]:mt-4 max-[700px]:text-left"><b className="block text-sm text-meridian-ink">Leadership meeting</b>Friday · 10:00 AM · 3 decisions open</div></div>
    <section className="grid overflow-hidden rounded-lg bg-meridian-navy text-white md:grid-cols-[1.25fr_.75fr]"><div className="p-7"><div className="text-[10px] font-extrabold uppercase tracking-[.13em] text-[#8fd6d2]">WORKING THESIS · UPDATED 08:42 PT</div><h2 className="my-2 text-2xl font-extrabold leading-tight tracking-[-.03em]">Make the first 90 days feel more useful — not just more rewarding.</h2><p className="max-w-xl leading-relaxed text-[#bdd0d6]">Early evidence suggests that a simpler onboarding journey may protect primary deposits more efficiently than a broad pricing incentive. The team is testing that hypothesis before it reaches the client deck.</p><div className="mt-5 flex gap-2"><button disabled={busy} className="rounded bg-meridian-lime px-3 py-2 text-xs font-extrabold text-meridian-navy disabled:opacity-50" onClick={onRun}>{busy ? 'Working…' : 'Run next analysis'}</button><button className="rounded border border-[#7193a1] bg-transparent px-3 py-2 text-xs font-extrabold text-white" onClick={() => toast('Question added to review queue')}>Add question</button></div></div><div className="flex flex-col justify-center bg-gradient-to-br from-[#124968] to-[#0c3248] p-6"><div className="text-[11px] font-extrabold uppercase tracking-[.12em] text-[#a4c1cc]">Primary deposits</div><div className="my-1 text-[43px] font-extrabold tracking-[-.06em]">$8.4B</div><div className="text-xs font-extrabold text-meridian-lime">↑ 4.7% year over year</div><div className="mt-4 flex h-11 items-end gap-1">{[6, 7, 5, 8, 7, 10, 9, 11].map((height, index) => <i key={index} style={{ height: `${height * 4}px` }} className="w-3 rounded-t bg-meridian-cyan" />)}</div></div></section>
    <div className="my-6 flex items-center justify-between"><h2 className="text-[17px] font-extrabold">Executive indicators</h2><span className="text-[11px] text-meridian-muted">Last refreshed from project workspace · 08:42 PT</span></div><div className="grid grid-cols-4 gap-3 max-[800px]:grid-cols-2"><Metric label="Net interest margin" value="3.42%" note="↑ 0.18 pts vs prior" /><Metric label="90-day retention" value="64%" note="Opportunity identified" /><Metric label="Digital completion" value="71%" note="↓ 4.2 pts vs prior" orange /><Metric label="CET1 ratio" value="12.6%" note="Within target range" /></div>
    <div className="mt-4 grid gap-4 md:grid-cols-[1.15fr_.85fr]"><Panel title="Analysis queue" meta="3 OPEN QUESTIONS">{queue.map(([letter, title, desc, status]) => <div key={letter} className="grid grid-cols-[24px_1fr_auto] items-center gap-2.5 border-b border-[#edf1f3] py-3 last:border-0"><div className="grid h-6 w-6 place-items-center rounded bg-[#e8f4f3] text-xs font-extrabold text-[#168983]">{letter}</div><div><b className="text-xs">{title}</b><p className="m-0 mt-1 text-[10px] text-meridian-muted">{desc}</p></div><span className={`text-[9px] font-black ${status === 'READY' ? 'text-meridian-cyan' : 'text-[#c58b2a]'}`}>{status}</span></div>)}</Panel><Panel title="Evidence ledger" meta={evidence.length ? `${evidence.length} RETRIEVED · GROUP 2` : 'ILLUSTRATIVE UNTIL RETRIEVED'}><div className="mb-2 flex justify-end gap-2"><input ref={fileInput} type="file" accept=".pdf,.docx,.csv,.xlsx,.txt,.json" className="hidden" onChange={event => onUpload(event.target.files?.[0])} /><button disabled={uploadBusy || !project} onClick={() => fileInput.current?.click()} className="rounded border border-meridian-line px-2 py-1 text-[10px] font-extrabold text-meridian-blue disabled:opacity-50">{uploadBusy ? 'Uploading…' : 'Upload source'}</button><button disabled={evidenceBusy || !project} onClick={onRetrieve} className="rounded border border-meridian-line px-2 py-1 text-[10px] font-extrabold text-meridian-blue disabled:opacity-50">{evidenceBusy ? 'Searching…' : 'Retrieve evidence'}</button></div>{(evidence.length ? evidence : sources).map((item, index) => { const live = item as import('../lib/api').EvidenceItem; const staticItem = item as string[]; const title = evidence.length ? (live.title || live.source || `Evidence ${index + 1}`) : staticItem[0]; const desc = evidence.length ? (live.content || 'Retrieved from the project-scoped evidence service.') : staticItem[1]; const tag = evidence.length ? `${live.source_type || 'EVIDENCE'}${live.reliability ? ` · ${Math.round(live.reliability * 100)}% RELIABLE` : ''}` : staticItem[2]; return <div key={evidence.length ? (live.evidence_id || index) : title} className="flex gap-2.5 border-b border-meridian-line py-2.5 last:border-0"><i className="mt-1 h-2 w-2 shrink-0 rounded-full bg-meridian-cyan" /><div><strong className="text-xs">{title}</strong><p className="m-0 my-1 line-clamp-3 text-[10px] text-meridian-muted">{desc}</p><code className="rounded bg-[#e8f5f3] px-1 py-0.5 text-[9px] text-[#287f82]">{tag}</code></div></div>; })}</Panel></div>
    <div className="mt-4 rounded border border-meridian-line bg-white p-4"><div className="flex items-center justify-between"><div><strong className="text-[13px]">Live agent workflow</strong><p className="m-0 mt-1 text-[11px] text-meridian-muted">{workflow ? `Execution ${workflow.execution_id.slice(0, 8)} · ${workflow.current_stage || 'queued'}` : 'No workflow has been started yet.'}</p></div>{workflow && <span className="rounded bg-[#e8f5f3] px-2 py-1 text-[10px] font-extrabold uppercase text-[#287f82]">{workflow.status.replace('_', ' ')}</span>}</div>{workflow && <div className="mt-3 h-2 overflow-hidden rounded bg-[#e9eff1]"><div className={`h-full rounded bg-meridian-cyan transition-all ${workflow.status === 'completed' ? 'w-full' : workflow.status === 'failed' ? 'w-1/2' : 'w-2/3'}`} /></div>}{outputs?.report_markdown && <div className="mt-3 rounded bg-[#f5f8f9] p-3 text-[11px] leading-relaxed text-meridian-ink"><b>Latest report output</b><p className="m-0 mt-1 line-clamp-3 whitespace-pre-wrap">{outputs.report_markdown}</p></div>}{(outputs?.warnings || []).length > 0 && <p className="mt-2 text-[10px] text-meridian-orange">Warnings: {(outputs?.warnings || []).join(' · ')}</p>}</div>
    <div className="mt-4 flex items-center justify-between gap-4 rounded border border-[#f0d99b] bg-[#fff9e7] p-4 max-[650px]:block"><div><strong className="text-[13px]">Manager review gate</strong><p className="m-0 mt-1 text-[11px] text-[#766343]">{workflow?.status === 'awaiting_input' ? 'The workflow is paused for a human decision. Review the evidence before approving.' : 'This is still a working hypothesis. Confirm the cohort definition before approving the next analysis.'}</p></div><button disabled={busy || approved} className="rounded bg-meridian-orange px-3 py-2 text-xs font-extrabold text-white disabled:bg-[#23856c] max-[650px]:mt-3" onClick={onApprove}>{approved || workflow?.status === 'completed' ? 'Approved' : workflow?.status === 'awaiting_input' ? 'Approve workflow' : 'Approve next analysis'}</button></div>
    <div className="mt-5 flex justify-between border-t border-meridian-line pt-3 text-[10px] text-meridian-muted"><span>Evidence-backed workspace · human judgment remains in the loop</span><span className="max-[650px]:hidden">No unsupported market claims</span></div>
  </div>;
}

function Metric({ label, value, note, orange = false }: { label: string; value: string; note: string; orange?: boolean }) { return <div className="rounded-lg border border-meridian-line bg-white p-4"><small className="block text-[11px] text-meridian-muted">{label}</small><strong className="my-2 block text-[25px] font-extrabold tracking-[-.05em]">{value}</strong><footer className={`text-[10px] font-extrabold ${orange ? 'text-meridian-orange' : 'text-meridian-cyan'}`}>{note}</footer></div>; }
function Panel({ title, meta, children }: { title: string; meta: string; children: React.ReactNode }) { return <section className="rounded-lg border border-meridian-line bg-white p-[18px]"><div className="mb-2 flex items-start justify-between border-b border-meridian-line pb-3"><h3 className="m-0 text-sm font-extrabold">{title}</h3><small className="text-[10px] text-meridian-muted">{meta}</small></div>{children}</section>; }
