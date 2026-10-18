import axios from 'axios';

export type Role='administrator'|'engagement_manager'|'consultant'|'client_executive';
export type Project={id:string;name:string;client_name:string;industry:string;objectives:string[];constraints:string[];status:string;owner_id:string;member_ids?:string[];created_at:string};
export type WorkflowStatus={execution_id:string;project_id:string;status:string;current_stage:string|null;pending_interrupt:Record<string,unknown>|null;final_status?:string|null;qa_iterations?:number;stage_history?:Array<Record<string,unknown>>;error?:Record<string,unknown>|null;created_at:string;updated_at:string;completed_at?:string|null};
export type WorkflowOutputs={execution_id:string;status:string;strategy?:Record<string,unknown>|null;qa?:Record<string,unknown>|null;report?:Record<string,unknown>|null;report_markdown?:string|null;warnings:string[];security_flags:Array<Record<string,unknown>>;usage:Record<string,unknown>};
export type EvidenceItem={evidence_id?:string;source?:string;title?:string;content?:string;source_type?:string;url?:string|null;reliability?:number;injection_suspected?:boolean};
export type AuditEvent={id:string;actor_id:string;action:string;resource_type:string;resource_id?:string|null;created_at:string};

export const api=axios.create({baseURL:process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000',headers:{'Content-Type':'application/json'}});
export function setToken(token:string){api.defaults.headers.common.Authorization=`Bearer ${token}`}
export async function demoLogin(role:Role='engagement_manager'){const {data}=await api.post('/api/v1/auth/demo-login',null,{params:{role}});setToken(data.access_token);return data}
export async function createProject(input:{name:string;client_name:string;industry:string;objectives:string[];constraints:string[]}){const {data}=await api.post<Project>('/api/v1/projects',input);return data}
export async function approveProject(id:string){const {data}=await api.post<Project>(`/api/v1/projects/${id}/status`,{decision:'approve',comment:'Cohort definition reviewed by Engagement Manager.'});return data}
export async function startAgentWorkflow(id:string){const {data}=await api.post(`/api/v1/projects/${id}/agent-workflows`,{client_brief:'Evaluate the deposit growth opportunity and identify a testable 90-day retention intervention.',client_name:'Harbor National Bank',industry:'Banking',objectives:['Improve primary deposit retention'],constraints:['90-day decision window','Protect branch relationships']});return data}
export async function getWorkflowStatus(id:string){const {data}=await api.get<WorkflowStatus>(`/api/v1/agent-workflows/${id}`);return data}
export async function getWorkflowOutputs(id:string){const {data}=await api.get<WorkflowOutputs>(`/api/v1/agent-workflows/${id}/outputs`);return data}
export async function resumeWorkflow(id:string, payload:Record<string,unknown>){const {data}=await api.post<WorkflowStatus>(`/api/v1/agent-workflows/${id}/resume`,{payload});return data}
export async function searchEvidence(projectId:string, question:string){const {data}=await api.post<{query:string;evidence:EvidenceItem[];retrieval_method:string}>(`/api/v1/projects/${projectId}/research`,{question,domain:'sec',top_k:5});return data}
export async function uploadDocument(projectId:string,file:File){const form=new FormData();form.append('file',file);const {data}=await api.post(`/api/v1/projects/${projectId}/documents?domain=sec`,form,{headers:{'Content-Type':'multipart/form-data'}});return data}
export async function getAuditEvents(){const {data}=await api.get<AuditEvent[]>('/api/v1/audit');return data}
export function connectWorkflowSocket(projectId:string,executionId:string,token:string,onEvent:(event:Record<string,unknown>)=>void){const apiUrl=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000';const wsUrl=apiUrl.replace(/^http/,'ws')+`/ws/projects/${projectId}/workflows/${executionId}?token=${encodeURIComponent(token)}`;const socket=new WebSocket(wsUrl);socket.onmessage=message=>{try{const envelope=JSON.parse(message.data);if(envelope.event)onEvent(envelope.event)}catch{/* Ignore malformed event frames. */}};return socket}
