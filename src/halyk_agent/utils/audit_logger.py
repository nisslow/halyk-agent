import os
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, base_dir: str = "data/audit"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def log_state(self, state, node_name: str):
        """
        Log the current AgentState (dict or BaseModel) at a given node.
        We create a directory for the case_id, and inside it a file for the iteration and node.
        """
        if isinstance(state, dict):
            case_id = state.get("case_id", "unknown_case")
            iteration = state.get("iteration", 0)
            state_dict = state
        else:
            case_id = getattr(state, "case_id", "unknown_case") or "unknown_case"
            iteration = getattr(state, "iteration", 0)
            state_dict = state.model_dump() if hasattr(state, "model_dump") else state.dict()
            
        case_dir = os.path.join(self.base_dir, str(case_id))
        os.makedirs(case_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iter_{iteration}_{node_name}_{timestamp}.md"
        filepath = os.path.join(case_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Audit Log: {node_name} (Iteration {iteration})\n\n")
            f.write(f"**Case ID:** {case_id}\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            
            # Retrieval
            if node_name in ["retriever", "resolver"]:
                f.write("## Retrieved Chunks\n")
                chunks = state_dict.get("retrieved_chunks", [])
                for idx, chunk in enumerate(chunks):
                    f.write(f"### Chunk {idx+1}\n")
                    f.write(f"**Doc ID:** {chunk.get('doc_id') if isinstance(chunk, dict) else getattr(chunk, 'doc_id', '')}\n")
                    text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', '')
                    f.write(f"```text\n{text}\n```\n\n")
            
            # Calculations & Verifications
            if node_name in ["calculator", "verifier"]:
                f.write("## Calculations\n")
                calc = state_dict.get("calculations", {})
                f.write("```json\n")
                f.write(json.dumps(calc, indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
                
                f.write("## Verification Results (Z3)\n")
                verif = state_dict.get("verification_results", {})
                f.write("```json\n")
                f.write(json.dumps(verif, indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
                
            # Synthesizer & Validator
            if node_name in ["synthesizer", "validator"]:
                f.write("## Synthesized Decision\n")
                f.write(f"**Decision:** {state_dict.get('decision')}\n")
                f.write(f"**Confidence:** {state_dict.get('confidence')}\n\n")
                
                f.write("## Reasoning Trace\n")
                trace = state_dict.get("reasoning_trace", [])
                for t in trace:
                    if isinstance(t, dict):
                        claim = t.get("claim", "")
                    else:
                        claim = getattr(t, "claim", "")
                    f.write(f"- {claim}\n")
                f.write("\n")
                
                if node_name == "validator":
                    f.write("## Critic Validation\n")
                    passed = state_dict.get("validation_passed")
                    f.write(f"**Passed:** {passed}\n")
                    if not passed:
                        f.write("### Errors\n")
                        errors = state_dict.get("validation_errors", [])
                        for err in errors:
                            f.write(f"- {err}\n")
                        
            # Dump full state as fallback
            f.write("\n---\n## Full State Dump\n")
            f.write("<details>\n<summary>Click to expand</summary>\n\n")
            f.write("```json\n")
            def default_serializer(obj):
                return str(obj)
            try:
                f.write(json.dumps(state_dict, indent=2, ensure_ascii=False, default=default_serializer))
            except Exception as e:
                f.write(f"Error serializing state: {e}")
            f.write("\n```\n</details>\n")
