import {create} from 'zustand';
import type {EvidenceItem,Project,WorkflowOutputs,WorkflowStatus} from './api';

type WorkspaceState={token:string;project:Project|null;workflow:WorkflowStatus|null;outputs:WorkflowOutputs|null;evidence:EvidenceItem[];setToken:(token:string)=>void;setProject:(project:Project)=>void;setWorkflow:(workflow:WorkflowStatus|null)=>void;setOutputs:(outputs:WorkflowOutputs|null)=>void;setEvidence:(evidence:EvidenceItem[])=>void};
export const useWorkspace=create<WorkspaceState>((set)=>({token:'',project:null,workflow:null,outputs:null,evidence:[],setToken:(token)=>set({token}),setProject:(project)=>set({project}),setWorkflow:(workflow)=>set({workflow}),setOutputs:(outputs)=>set({outputs}),setEvidence:(evidence)=>set({evidence})}));
