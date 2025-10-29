"""
DSPy + GEPA: Optimizing Inter-Agent Communication Protocols

This example demonstrates a novel approach to multi-agent systems:
Instead of optimizing individual agent prompts, we optimize the COMMUNICATION
PROTOCOLS between agents using DSPy + GEPA + MAST.

Key innovations:
1. Communication protocols as DSPy signatures
2. MAST (Multi-Agent System Failure Taxonomy) for identifying failures
3. Model Assertions for runtime monitoring
4. GEPA for optimizing dialogue structure and information exchange

Based on research from UC Berkeley Sky Computing Lab:
- MAST: https://sky.cs.berkeley.edu/project/mast/
- Model Assertions: Kang et al. (2019)
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field

import dspy
from dspy.teleprompt import GEPA

# ----------------------------
# Configuration
# ----------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

# ----------------------------
# MAST: Multi-Agent System Failure Taxonomy
# ----------------------------

class MASTFailureCategory(str, Enum):
    """
    MAST identifies 3 main categories of failures in multi-agent systems.
    Based on: "Why Do Multi-Agent LLM Systems Fail?" (arXiv:2503.13657)
    """
    SYSTEM_DESIGN = "system_design"           # Issues with architecture/roles
    INTER_AGENT_MISALIGNMENT = "misalignment"  # Communication breakdowns
    TASK_VERIFICATION = "verification"         # Incorrect output validation


class MASTFailureMode(str, Enum):
    """
    14 unique failure modes identified by MAST taxonomy.
    We focus on communication-related failures for this example.
    """
    # System Design Issues
    ROLE_CONFUSION = "role_confusion"                   # Agent unclear about its role
    INSUFFICIENT_CONTEXT = "insufficient_context"       # Missing critical information
    INEFFICIENT_ROUTING = "inefficient_routing"         # Task sent to wrong agent

    # Inter-Agent Misalignment
    INFORMATION_LOSS = "information_loss"               # Data lost in handoff
    AMBIGUOUS_REQUEST = "ambiguous_request"             # Unclear communication
    REDUNDANT_EXCHANGE = "redundant_exchange"           # Unnecessary back-and-forth
    CONFLICTING_OBJECTIVES = "conflicting_objectives"   # Agents have different goals
    POOR_NEGOTIATION = "poor_negotiation"               # Failed conflict resolution

    # Task Verification
    INCOMPLETE_RESPONSE = "incomplete_response"         # Missing required information
    FORMAT_MISMATCH = "format_mismatch"                 # Wrong response structure
    UNVERIFIED_HANDOFF = "unverified_handoff"           # No confirmation of receipt
    SPEC_MISUNDERSTANDING = "spec_misunderstanding"     # Misinterpreted requirements


@dataclass
class MASTFailureRecord:
    """Record of a detected failure in the system."""
    timestamp: datetime
    failure_mode: MASTFailureMode
    category: MASTFailureCategory
    from_agent: str
    to_agent: Optional[str]
    context: Dict[str, Any]
    severity: float  # 0-1, where 1 is critical
    message: str


class MASTMonitor:
    """
    Monitors multi-agent communications and categorizes failures
    using the MAST taxonomy.
    """
    def __init__(self):
        self.failures: List[MASTFailureRecord] = []
        self.stats: Dict[MASTFailureMode, int] = {mode: 0 for mode in MASTFailureMode}

    def record_failure(
        self,
        failure_mode: MASTFailureMode,
        from_agent: str,
        to_agent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: float = 0.5,
        message: str = ""
    ):
        """Record a detected failure."""
        category = self._get_category(failure_mode)
        record = MASTFailureRecord(
            timestamp=datetime.now(),
            failure_mode=failure_mode,
            category=category,
            from_agent=from_agent,
            to_agent=to_agent,
            context=context or {},
            severity=severity,
            message=message
        )
        self.failures.append(record)
        self.stats[failure_mode] += 1

    def _get_category(self, mode: MASTFailureMode) -> MASTFailureCategory:
        """Map failure mode to category."""
        system_design = {
            MASTFailureMode.ROLE_CONFUSION,
            MASTFailureMode.INSUFFICIENT_CONTEXT,
            MASTFailureMode.INEFFICIENT_ROUTING
        }
        misalignment = {
            MASTFailureMode.INFORMATION_LOSS,
            MASTFailureMode.AMBIGUOUS_REQUEST,
            MASTFailureMode.REDUNDANT_EXCHANGE,
            MASTFailureMode.CONFLICTING_OBJECTIVES,
            MASTFailureMode.POOR_NEGOTIATION
        }

        if mode in system_design:
            return MASTFailureCategory.SYSTEM_DESIGN
        elif mode in misalignment:
            return MASTFailureCategory.INTER_AGENT_MISALIGNMENT
        else:
            return MASTFailureCategory.TASK_VERIFICATION

    def get_failure_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of failures."""
        total = len(self.failures)
        if total == 0:
            return {"total_failures": 0}

        category_counts = {}
        for category in MASTFailureCategory:
            count = sum(1 for f in self.failures if f.category == category)
            category_counts[category.value] = count

        avg_severity = sum(f.severity for f in self.failures) / total

        # Top failure modes
        top_modes = sorted(
            self.stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_failures": total,
            "avg_severity": round(avg_severity, 2),
            "by_category": category_counts,
            "top_failure_modes": [
                {"mode": mode.value, "count": count}
                for mode, count in top_modes if count > 0
            ],
            "optimization_priority": self._get_optimization_priority()
        }

    def _get_optimization_priority(self) -> List[str]:
        """Identify which communication protocols need optimization most."""
        priorities = []

        # Calculate weighted scores for different areas
        misalignment_score = sum(
            f.severity for f in self.failures
            if f.category == MASTFailureCategory.INTER_AGENT_MISALIGNMENT
        )

        verification_score = sum(
            f.severity for f in self.failures
            if f.category == MASTFailureCategory.TASK_VERIFICATION
        )

        system_score = sum(
            f.severity for f in self.failures
            if f.category == MASTFailureCategory.SYSTEM_DESIGN
        )

        # Prioritize based on scores
        scores = [
            ("task_request_protocol", misalignment_score),
            ("data_handoff_protocol", verification_score),
            ("routing_protocol", system_score)
        ]

        priorities = [name for name, score in sorted(scores, key=lambda x: x[1], reverse=True) if score > 0]

        return priorities

# ----------------------------
# Model Assertions
# ----------------------------

class CommunicationAssertion:
    """
    Model assertions for monitoring communication quality.
    Based on: Kang et al. "Model Assertions for Monitoring ML Models"

    Assertions are boolean functions over model inputs/outputs that
    indicate when errors may be occurring.
    """

    @staticmethod
    def assert_clarity(request: str, threshold: float = 0.5) -> Tuple[bool, float, str]:
        """
        Assert that a communication request is clear and actionable.
        Returns: (passed, score, message)
        """
        score = 0.0
        issues = []

        # Check for specific details
        if len(request.split()) < 10:
            issues.append("Too brief - lacks detail")
        else:
            score += 0.3

        # Check for question words or action verbs
        action_words = ['analyze', 'collect', 'find', 'synthesize', 'compare', 'evaluate']
        if any(word in request.lower() for word in action_words):
            score += 0.3
        else:
            issues.append("No clear action verb")

        # Check for context
        context_indicators = ['because', 'in order to', 'for', 'regarding', 'about']
        if any(word in request.lower() for word in context_indicators):
            score += 0.4
        else:
            issues.append("Missing context/purpose")

        passed = score >= threshold
        message = "; ".join(issues) if issues else "Clear communication"

        return passed, score, message

    @staticmethod
    def assert_completeness(response: str, request: str, threshold: float = 0.6) -> Tuple[bool, float, str]:
        """
        Assert that a response contains necessary information to address request.
        """
        score = 0.0
        issues = []

        # Check minimum length
        if len(response.split()) < 20:
            issues.append("Response too brief")
        else:
            score += 0.3

        # Check for structured content
        has_structure = any(marker in response for marker in ['1.', '2.', '-', '*', ':'])
        if has_structure:
            score += 0.3
        else:
            issues.append("Lacks structure")

        # Check for specificity (numbers, names, etc.)
        has_specifics = any(char.isdigit() for char in response)
        if has_specifics:
            score += 0.4
        else:
            issues.append("Lacks specific details")

        passed = score >= threshold
        message = "; ".join(issues) if issues else "Complete response"

        return passed, score, message

    @staticmethod
    def assert_routing_accuracy(
        task_type: str,
        assigned_agent: str,
        agent_specialties: Dict[str, List[str]],
        threshold: float = 0.7
    ) -> Tuple[bool, float, str]:
        """
        Assert that a task was routed to an appropriate specialist.
        """
        if assigned_agent not in agent_specialties:
            return False, 0.0, f"Unknown agent: {assigned_agent}"

        specialties = agent_specialties[assigned_agent]
        task_lower = task_type.lower()

        # Check for keyword matches
        matches = sum(1 for specialty in specialties if specialty.lower() in task_lower)
        score = min(matches / len(specialties), 1.0) if specialties else 0.0

        passed = score >= threshold
        message = f"Routing score: {score:.2f}" if passed else f"Poor routing - better fit: {', '.join(specialties)}"

        return passed, score, message

# ----------------------------
# Agent Roles and Specialties
# ----------------------------

class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    DATA_COLLECTOR = "data_collector"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


AGENT_SPECIALTIES = {
    AgentRole.COORDINATOR: ["routing", "task decomposition", "coordination"],
    AgentRole.DATA_COLLECTOR: ["search", "retrieval", "data gathering", "information collection"],
    AgentRole.ANALYZER: ["analysis", "evaluation", "comparison", "assessment"],
    AgentRole.SYNTHESIZER: ["synthesis", "summarization", "integration", "combination"]
}

# ----------------------------
# DSPy Communication Protocol Signatures
# ----------------------------

class TaskRequestProtocol(dspy.Signature):
    """
    Protocol for requesting help from a specialist agent.
    This signature is optimized by GEPA to minimize ambiguity and maximize routing accuracy.
    """
    task_description = dspy.InputField(desc="What needs to be done")
    requesting_agent_role = dspy.InputField(desc="Role of the requesting agent")
    target_agent_specialty = dspy.InputField(desc="Specialty area needed")
    context = dspy.InputField(desc="Background information and constraints")

    # Optimized output: clear, actionable request
    optimized_request = dspy.OutputField(
        desc="Clear, specific request that the target agent can execute. Include: (1) specific action needed, (2) required deliverables, (3) success criteria"
    )
    priority_level = dspy.OutputField(desc="Priority: high, medium, or low")


class DataHandoffProtocol(dspy.Signature):
    """
    Protocol for handing off data/results between agents.
    Optimized to minimize information loss and ensure completeness.
    """
    data_summary = dspy.InputField(desc="Summary of the data being handed off")
    source_agent_role = dspy.InputField(desc="Role of the sending agent")
    target_agent_role = dspy.InputField(desc="Role of the receiving agent")
    processing_context = dspy.InputField(desc="What was done and what's next")

    # Optimized output: structured handoff
    structured_handoff = dspy.OutputField(
        desc="Structured data handoff including: (1) key findings, (2) data format/location, (3) quality notes, (4) next steps recommended"
    )
    verification_checklist = dspy.OutputField(
        desc="List of items the receiving agent should verify"
    )


class RoutingProtocol(dspy.Signature):
    """
    Protocol for routing tasks to the most appropriate specialist.
    Optimized to maximize first-time-right routing.
    """
    task_description = dspy.InputField(desc="Description of the task to route")
    available_specialists = dspy.InputField(desc="List of available specialists and their capabilities")
    task_complexity = dspy.InputField(desc="Simple, moderate, or complex")

    # Optimized output: routing decision with rationale
    selected_specialist = dspy.OutputField(desc="The chosen specialist agent")
    routing_rationale = dspy.OutputField(
        desc="Brief explanation of why this specialist is the best fit"
    )
    confidence_level = dspy.OutputField(desc="Confidence in routing decision: high, medium, low")


class NegotiationProtocol(dspy.Signature):
    """
    Protocol for resolving conflicts when agents disagree.
    Optimized for consensus quality and efficiency.
    """
    agent_a_position = dspy.InputField(desc="Agent A's viewpoint and reasoning")
    agent_b_position = dspy.InputField(desc="Agent B's viewpoint and reasoning")
    conflict_context = dspy.InputField(desc="What the disagreement is about")

    # Optimized output: resolution strategy
    consensus_approach = dspy.OutputField(
        desc="Strategy to resolve the disagreement: compromise, defer to expert, gather more data, or escalate"
    )
    synthesis = dspy.OutputField(
        desc="Combined perspective that addresses both viewpoints"
    )
    action_items = dspy.OutputField(desc="Specific next steps to move forward")


# Initialize DSPy modules
task_requester = dspy.ChainOfThought(TaskRequestProtocol)
data_handoff = dspy.ChainOfThought(DataHandoffProtocol)
router = dspy.Predict(RoutingProtocol)
negotiator = dspy.ChainOfThought(NegotiationProtocol)

# ----------------------------
# Multi-Agent System with Monitoring
# ----------------------------

class CommunicationMetrics(BaseModel):
    """Metrics for evaluating communication quality."""
    total_exchanges: int = 0
    failed_assertions: int = 0
    successful_routings: int = 0
    total_routings: int = 0
    avg_clarity_score: float = 0.0
    avg_completeness_score: float = 0.0
    avg_routing_accuracy: float = 0.0


@dataclass
class AgentMessage:
    """A message exchanged between agents."""
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str  # request, response, handoff, negotiation
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MultiAgentSystem:
    """
    Multi-agent research system with monitored communication.

    This system demonstrates how communication protocols can be optimized
    separately from individual agent capabilities.
    """

    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.mast_monitor = MASTMonitor()
        self.metrics = CommunicationMetrics()
        self.message_history: List[AgentMessage] = []

    def request_task(
        self,
        from_role: AgentRole,
        to_specialty: str,
        task: str,
        context: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Request a task from another agent using optimized protocol.
        Returns: (optimized_request, metrics)
        """
        with dspy.context(lm=self.lm):
            result = task_requester(
                task_description=task,
                requesting_agent_role=from_role.value,
                target_agent_specialty=to_specialty,
                context=context
            )

        # Assert clarity
        passed, clarity_score, msg = CommunicationAssertion.assert_clarity(
            result.optimized_request
        )

        self.metrics.total_exchanges += 1
        self.metrics.avg_clarity_score = (
            (self.metrics.avg_clarity_score * (self.metrics.total_exchanges - 1) + clarity_score)
            / self.metrics.total_exchanges
        )

        if not passed:
            self.metrics.failed_assertions += 1
            self.mast_monitor.record_failure(
                failure_mode=MASTFailureMode.AMBIGUOUS_REQUEST,
                from_agent=from_role.value,
                to_agent=to_specialty,
                severity=1.0 - clarity_score,
                message=f"Clarity assertion failed: {msg}"
            )

        # Record message
        self.message_history.append(
            AgentMessage(
                from_agent=from_role,
                to_agent=AgentRole.COORDINATOR,  # Will be routed
                message_type="request",
                content=result.optimized_request,
                metadata={"priority": result.priority_level, "clarity_score": clarity_score}
            )
        )

        return result.optimized_request, {
            "priority": result.priority_level,
            "clarity_score": clarity_score,
            "clarity_passed": passed
        }

    def route_task(
        self,
        task: str,
        complexity: str = "moderate"
    ) -> Tuple[AgentRole, str, Dict[str, Any]]:
        """
        Route a task to the most appropriate specialist.
        Returns: (selected_role, rationale, metrics)
        """
        # Format specialist information
        specialist_info = "\n".join([
            f"- {role.value}: {', '.join(AGENT_SPECIALTIES[role])}"
            for role in AgentRole if role != AgentRole.COORDINATOR
        ])

        with dspy.context(lm=self.lm):
            result = router(
                task_description=task,
                available_specialists=specialist_info,
                task_complexity=complexity
            )

        # Parse selected specialist
        selected_role = None
        for role in AgentRole:
            if role.value in result.selected_specialist.lower():
                selected_role = role
                break

        if not selected_role:
            # Default to coordinator if parsing failed
            selected_role = AgentRole.COORDINATOR

        # Assert routing accuracy
        self.metrics.total_routings += 1

        if selected_role != AgentRole.COORDINATOR:
            passed, routing_score, msg = CommunicationAssertion.assert_routing_accuracy(
                task_type=task,
                assigned_agent=selected_role.value,
                agent_specialties={r.value: AGENT_SPECIALTIES[r] for r in AgentRole}
            )

            self.metrics.avg_routing_accuracy = (
                (self.metrics.avg_routing_accuracy * (self.metrics.total_routings - 1) + routing_score)
                / self.metrics.total_routings
            )

            if passed:
                self.metrics.successful_routings += 1
            else:
                self.mast_monitor.record_failure(
                    failure_mode=MASTFailureMode.INEFFICIENT_ROUTING,
                    from_agent=AgentRole.COORDINATOR.value,
                    to_agent=selected_role.value,
                    severity=1.0 - routing_score,
                    message=f"Routing assertion failed: {msg}"
                )

        return selected_role, result.routing_rationale, {
            "confidence": result.confidence_level,
            "routing_score": routing_score if selected_role != AgentRole.COORDINATOR else 0.5
        }

    def handoff_data(
        self,
        from_role: AgentRole,
        to_role: AgentRole,
        data_summary: str,
        context: str
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Hand off data between agents using optimized protocol.
        Returns: (structured_handoff, verification_checklist, metrics)
        """
        with dspy.context(lm=self.lm):
            result = data_handoff(
                data_summary=data_summary,
                source_agent_role=from_role.value,
                target_agent_role=to_role.value,
                processing_context=context
            )

        # Parse verification checklist
        checklist_text = result.verification_checklist
        checklist = [
            item.strip('- ').strip()
            for item in checklist_text.split('\n')
            if item.strip() and not item.strip().startswith('#')
        ]

        # Assert completeness
        passed, completeness_score, msg = CommunicationAssertion.assert_completeness(
            response=result.structured_handoff,
            request=data_summary
        )

        self.metrics.total_exchanges += 1
        self.metrics.avg_completeness_score = (
            (self.metrics.avg_completeness_score * (self.metrics.total_exchanges - 1) + completeness_score)
            / self.metrics.total_exchanges
        )

        if not passed:
            self.metrics.failed_assertions += 1
            self.mast_monitor.record_failure(
                failure_mode=MASTFailureMode.INCOMPLETE_RESPONSE,
                from_agent=from_role.value,
                to_agent=to_role.value,
                severity=1.0 - completeness_score,
                message=f"Completeness assertion failed: {msg}"
            )

        # Record message
        self.message_history.append(
            AgentMessage(
                from_agent=from_role,
                to_agent=to_role,
                message_type="handoff",
                content=result.structured_handoff,
                metadata={"completeness_score": completeness_score, "checklist": checklist}
            )
        )

        return result.structured_handoff, checklist, {
            "completeness_score": completeness_score,
            "completeness_passed": passed
        }

    def negotiate(
        self,
        agent_a_role: AgentRole,
        agent_b_role: AgentRole,
        position_a: str,
        position_b: str,
        conflict_context: str
    ) -> Tuple[str, str, List[str]]:
        """
        Negotiate between agents with conflicting viewpoints.
        Returns: (approach, synthesis, action_items)
        """
        with dspy.context(lm=self.lm):
            result = negotiator(
                agent_a_position=position_a,
                agent_b_position=position_b,
                conflict_context=conflict_context
            )

        # Parse action items
        action_text = result.action_items
        actions = [
            item.strip('- ').strip()
            for item in action_text.split('\n')
            if item.strip() and not item.strip().startswith('#')
        ]

        # Check if negotiation was successful (has concrete actions)
        if len(actions) < 2:
            self.mast_monitor.record_failure(
                failure_mode=MASTFailureMode.POOR_NEGOTIATION,
                from_agent=agent_a_role.value,
                to_agent=agent_b_role.value,
                severity=0.7,
                message="Negotiation produced insufficient action items"
            )

        # Record message
        self.message_history.append(
            AgentMessage(
                from_agent=agent_a_role,
                to_agent=agent_b_role,
                message_type="negotiation",
                content=result.synthesis,
                metadata={"approach": result.consensus_approach, "actions": actions}
            )
        )

        return result.consensus_approach, result.synthesis, actions

    def get_communication_report(self) -> Dict[str, Any]:
        """Generate comprehensive communication quality report."""
        mast_summary = self.mast_monitor.get_failure_summary()

        return {
            "metrics": {
                "total_exchanges": self.metrics.total_exchanges,
                "total_routings": self.metrics.total_routings,
                "routing_accuracy": (
                    f"{self.metrics.successful_routings}/{self.metrics.total_routings}"
                    if self.metrics.total_routings > 0 else "N/A"
                ),
                "avg_clarity_score": round(self.metrics.avg_clarity_score, 2),
                "avg_completeness_score": round(self.metrics.avg_completeness_score, 2),
                "avg_routing_accuracy": round(self.metrics.avg_routing_accuracy, 2),
                "failed_assertions": self.metrics.failed_assertions
            },
            "mast_analysis": mast_summary,
            "message_count": len(self.message_history)
        }

# ----------------------------
# GEPA Optimization for Communication Protocols
# ----------------------------

def communication_protocol_metric(gold, pred, trace=None) -> float:
    """
    Metric for GEPA to optimize communication protocols.

    Unlike typical DSPy metrics that measure task accuracy, this measures
    COMMUNICATION QUALITY: clarity, completeness, routing accuracy.
    """
    score = 0.0

    # Check for TaskRequestProtocol outputs
    if hasattr(pred, 'optimized_request'):
        request = pred.optimized_request or ""
        passed, clarity_score, _ = CommunicationAssertion.assert_clarity(request)
        score = clarity_score

    # Check for DataHandoffProtocol outputs
    elif hasattr(pred, 'structured_handoff'):
        handoff = pred.structured_handoff or ""
        # Simple heuristic: check structure
        has_sections = sum(1 for marker in ['(1)', '(2)', '(3)', '(4)'] if marker in handoff)
        has_checklist = bool(pred.verification_checklist)
        score = (has_sections * 0.15) + (0.4 if has_checklist else 0.0)

    # Check for RoutingProtocol outputs
    elif hasattr(pred, 'selected_specialist'):
        specialist = pred.selected_specialist or ""
        has_rationale = len(pred.routing_rationale or "") > 20
        has_confidence = bool(pred.confidence_level)
        # Check if specialist is actually a valid role
        valid_specialist = any(role.value in specialist.lower() for role in AgentRole)
        score = (0.4 if valid_specialist else 0.0) + (0.3 if has_rationale else 0.0) + (0.3 if has_confidence else 0.0)

    # Check for NegotiationProtocol outputs
    elif hasattr(pred, 'consensus_approach'):
        has_approach = len(pred.consensus_approach or "") > 10
        has_synthesis = len(pred.synthesis or "") > 30
        has_actions = len(pred.action_items or "") > 20
        score = (0.3 if has_approach else 0.0) + (0.4 if has_synthesis else 0.0) + (0.3 if has_actions else 0.0)

    return max(0.0, min(1.0, score))


def create_training_data() -> Dict[str, List[dspy.Example]]:
    """
    Create training examples for each communication protocol.

    These examples represent GOOD communication patterns that GEPA should learn.
    """

    # TaskRequestProtocol training data
    task_request_examples = [
        dspy.Example(
            task_description="Find recent papers on transformer architectures",
            requesting_agent_role="coordinator",
            target_agent_specialty="data collection and search",
            context="Working on a literature review about neural architecture innovations from 2023-2024. Need peer-reviewed sources.",
            optimized_request="Please search for peer-reviewed papers on transformer architectures published between 2023-2024. Deliverables: (1) List of 5-10 papers with titles, authors, and publication venues, (2) Brief abstract summary for each. Success criteria: Papers should be from reputable ML conferences (NeurIPS, ICML, ICLR, etc.) and focus specifically on architectural innovations.",
            priority_level="high"
        ).with_inputs('task_description', 'requesting_agent_role', 'target_agent_specialty', 'context'),

        dspy.Example(
            task_description="Compare the performance of three ML models",
            requesting_agent_role="coordinator",
            target_agent_specialty="analysis and evaluation",
            context="Have benchmark results from Model A, B, and C across 3 datasets. Need statistical analysis.",
            optimized_request="Please analyze and compare the performance of Models A, B, and C using the provided benchmark results. Deliverables: (1) Statistical significance tests (t-test or ANOVA), (2) Performance ranking with confidence intervals, (3) Identify which model is best for each dataset. Success criteria: Include p-values, effect sizes, and clear recommendation for model selection.",
            priority_level="medium"
        ).with_inputs('task_description', 'requesting_agent_role', 'target_agent_specialty', 'context'),
    ]

    # DataHandoffProtocol training data
    data_handoff_examples = [
        dspy.Example(
            data_summary="Collected 15 research papers on transformer architectures, with metadata and abstracts",
            source_agent_role="data_collector",
            target_agent_role="analyzer",
            processing_context="Completed literature search using multiple academic databases. Papers filtered for relevance and publication date. Next step: extract key innovations from each paper.",
            structured_handoff="Data handoff from Data Collector to Analyzer:\n\n(1) Key findings: Successfully retrieved 15 peer-reviewed papers from top ML conferences (NeurIPS, ICML, ICLR). Date range: 2023-2024. Primary themes: attention mechanisms, efficiency improvements, multi-modal transformers.\n\n(2) Data format/location: JSON file containing title, authors, year, venue, abstract, and DOI for each paper. Located at ./papers_data.json\n\n(3) Quality notes: All papers are peer-reviewed and from tier-1 venues. Two papers are pre-prints (arXiv) but highly cited. No duplicate papers found.\n\n(4) Next steps recommended: Extract and categorize architectural innovations from each paper. Focus on novel attention mechanisms and efficiency techniques. Create comparison table.",
            verification_checklist="Verify all 15 papers have complete metadata\nCheck that JSON file is properly formatted\nConfirm all papers are within 2023-2024 date range\nEnsure abstracts are complete and readable"
        ).with_inputs('data_summary', 'source_agent_role', 'target_agent_role', 'processing_context'),

        dspy.Example(
            data_summary="Analysis complete: Model A outperforms B and C on datasets 1 and 2, but not dataset 3",
            source_agent_role="analyzer",
            target_agent_role="synthesizer",
            processing_context="Conducted statistical tests on benchmark results. Found significant differences between models. Ready for final synthesis and recommendations.",
            structured_handoff="Data handoff from Analyzer to Synthesizer:\n\n(1) Key findings: Model A shows statistically significant improvements over B and C on datasets 1 and 2 (p < 0.01), with 12% and 8% accuracy gains respectively. However, Model C performs best on dataset 3 (p < 0.05).\n\n(2) Data format/location: Analysis results in ./analysis_results.csv including mean accuracy, std dev, p-values, and effect sizes for all model-dataset combinations.\n\n(3) Quality notes: Statistical tests properly conducted with appropriate corrections for multiple comparisons. Sample sizes sufficient for reliable conclusions. One outlier detected in dataset 2 but kept after verification.\n\n(4) Next steps recommended: Synthesize findings into actionable recommendations. Consider dataset characteristics to explain Model C's performance on dataset 3. Provide model selection guidance based on use case.",
            verification_checklist="Confirm all p-values and effect sizes are present\nCheck that statistical test assumptions were met\nVerify dataset characteristics are documented\nEnsure outlier treatment is explained"
        ).with_inputs('data_summary', 'source_agent_role', 'target_agent_role', 'processing_context'),
    ]

    # RoutingProtocol training data
    routing_examples = [
        dspy.Example(
            task_description="Search for recent papers on graph neural networks",
            available_specialists="- data_collector: search, retrieval, data gathering, information collection\n- analyzer: analysis, evaluation, comparison, assessment\n- synthesizer: synthesis, summarization, integration, combination",
            task_complexity="simple",
            selected_specialist="data_collector",
            routing_rationale="This is a search and retrieval task that requires gathering papers from academic databases. The data_collector specializes in information collection and search, making it the ideal choice for this task.",
            confidence_level="high"
        ).with_inputs('task_description', 'available_specialists', 'task_complexity'),

        dspy.Example(
            task_description="Combine insights from multiple analysis reports into a coherent summary",
            available_specialists="- data_collector: search, retrieval, data gathering, information collection\n- analyzer: analysis, evaluation, comparison, assessment\n- synthesizer: synthesis, summarization, integration, combination",
            task_complexity="moderate",
            selected_specialist="synthesizer",
            routing_rationale="This task requires integrating information from multiple sources and creating a unified summary. The synthesizer agent specializes in synthesis, summarization, and combination of information, making it the perfect fit.",
            confidence_level="high"
        ).with_inputs('task_description', 'available_specialists', 'task_complexity'),
    ]

    # NegotiationProtocol training data
    negotiation_examples = [
        dspy.Example(
            agent_a_position="We should use Model A because it has the highest overall accuracy (85%) and is well-established in the field.",
            agent_b_position="We should use Model C because it's specifically optimized for our dataset 3, which is the most important for our use case, achieving 88% accuracy compared to Model A's 79%.",
            conflict_context="Selecting which ML model to deploy in production. Dataset 3 represents 60% of expected real-world traffic.",
            consensus_approach="defer to expert - Since dataset 3 represents the majority of real-world traffic, performance on this dataset should be weighted more heavily. The data supports Model C's superior performance on the most critical dataset.",
            synthesis="Both agents raise valid points. While Model A has better overall accuracy across all datasets, Model C's specialized performance on dataset 3 is more relevant given the expected traffic distribution. The 9% accuracy improvement on the primary use case (dataset 3) outweighs Model A's better performance on the less-used datasets 1 and 2.",
            action_items="Deploy Model C as the primary model for production\nMonitor real-world performance to validate the decision\nKeep Model A as a fallback for edge cases similar to datasets 1-2\nSet up A/B testing to compare models on actual traffic\nReview decision in 3 months with production data"
        ).with_inputs('agent_a_position', 'agent_b_position', 'conflict_context'),
    ]

    return {
        "task_request": task_request_examples,
        "data_handoff": data_handoff_examples,
        "routing": routing_examples,
        "negotiation": negotiation_examples
    }


def optimize_communication_protocols(reflection_lm: dspy.LM):
    """
    Use GEPA to optimize all communication protocols.

    This is the key innovation: we're not optimizing agents for task performance,
    we're optimizing the LANGUAGE they use to communicate with each other.
    """
    print("\n" + "="*80)
    print("GEPA OPTIMIZATION: Communication Protocols")
    print("="*80)
    print("\nOptimizing inter-agent communication using GEPA...")
    print("Goal: Learn optimal dialogue patterns, not task-specific prompts\n")

    training_data = create_training_data()

    # Create GEPA optimizer
    gepa = GEPA(
        metric=communication_protocol_metric,
        auto="light",
        reflection_lm=reflection_lm,
        track_stats=False
    )

    # Optimize each protocol
    protocols = {
        "TaskRequestProtocol": (task_requester, training_data["task_request"]),
        "DataHandoffProtocol": (data_handoff, training_data["data_handoff"]),
        "RoutingProtocol": (router, training_data["routing"]),
        "NegotiationProtocol": (negotiator, training_data["negotiation"])
    }

    optimized = {}
    for name, (module, trainset) in protocols.items():
        print(f"\n[GEPA] Optimizing {name}...")
        print(f"[GEPA] Training examples: {len(trainset)}")
        try:
            optimized_module = gepa.compile(student=module, trainset=trainset)
            optimized[name] = optimized_module
            print(f"[GEPA] ✓ {name} optimization complete")
        except Exception as e:
            print(f"[GEPA] ⚠ {name} optimization failed: {e}")
            optimized[name] = module

    print("\n" + "="*80)
    print("Communication protocol optimization complete!")
    print("="*80 + "\n")

    return optimized

# ----------------------------
# Example Usage and Evaluation
# ----------------------------

async def run_research_scenario(
    system: MultiAgentSystem,
    scenario_name: str
) -> Dict[str, Any]:
    """Run a research scenario through the multi-agent system."""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}\n")

    if scenario_name == "literature_review":
        # Step 1: Coordinator requests data collection
        print("[1] Coordinator → Data Collector: Request literature search")
        request, req_metrics = system.request_task(
            from_role=AgentRole.COORDINATOR,
            to_specialty="data gathering and search",
            task="Find recent papers on multi-agent systems",
            context="Need to understand current state of multi-agent LLM research for a technical report"
        )
        print(f"    Request: {request[:100]}...")
        print(f"    Metrics: {req_metrics}\n")

        # Step 2: Route the task
        print("[2] Coordinator: Route task to specialist")
        assigned_role, rationale, route_metrics = system.route_task(
            task=request,
            complexity="moderate"
        )
        print(f"    Assigned to: {assigned_role.value}")
        print(f"    Rationale: {rationale}")
        print(f"    Metrics: {route_metrics}\n")

        # Step 3: Data collector hands off to analyzer
        print("[3] Data Collector → Analyzer: Hand off collected papers")
        handoff, checklist, handoff_metrics = system.handoff_data(
            from_role=AgentRole.DATA_COLLECTOR,
            to_role=AgentRole.ANALYZER,
            data_summary="Collected 12 papers on multi-agent LLM systems from ACL, NeurIPS, and arXiv",
            context="Papers focus on communication patterns, coordination strategies, and failure modes"
        )
        print(f"    Handoff: {handoff[:150]}...")
        print(f"    Checklist items: {len(checklist)}")
        print(f"    Metrics: {handoff_metrics}\n")

        # Step 4: Analyzer hands off to synthesizer
        print("[4] Analyzer → Synthesizer: Hand off analysis results")
        handoff2, checklist2, handoff_metrics2 = system.handoff_data(
            from_role=AgentRole.ANALYZER,
            to_role=AgentRole.SYNTHESIZER,
            data_summary="Identified 5 key themes across the papers: communication protocols, failure handling, emergent behaviors, coordination strategies, and evaluation methods",
            context="Each theme has 3-5 supporting papers with specific examples"
        )
        print(f"    Handoff: {handoff2[:150]}...")
        print(f"    Checklist items: {len(checklist2)}")
        print(f"    Metrics: {handoff_metrics2}\n")

    elif scenario_name == "model_comparison":
        # Step 1: Request analysis
        print("[1] Coordinator → Analyzer: Request model comparison")
        request, req_metrics = system.request_task(
            from_role=AgentRole.COORDINATOR,
            to_specialty="analysis and evaluation",
            task="Compare performance of GPT-4, Claude, and Gemini on reasoning tasks",
            context="Have benchmark results from MMLU, GSM8K, and HumanEval"
        )
        print(f"    Request: {request[:100]}...")
        print(f"    Metrics: {req_metrics}\n")

        # Step 2: Route task
        print("[2] Coordinator: Route comparison task")
        assigned_role, rationale, route_metrics = system.route_task(
            task=request,
            complexity="moderate"
        )
        print(f"    Assigned to: {assigned_role.value}")
        print(f"    Rationale: {rationale}\n")

        # Step 3: Simulate disagreement between analyzer and synthesizer
        print("[3] Analyzer ↔ Synthesizer: Negotiation on conclusions")
        approach, synthesis, actions = system.negotiate(
            agent_a_role=AgentRole.ANALYZER,
            agent_b_role=AgentRole.SYNTHESIZER,
            position_a="GPT-4 shows the most consistent performance across all three benchmarks, making it the safest choice for deployment.",
            position_b="While GPT-4 is consistent, Claude shows superior performance on reasoning tasks (GSM8K), which is our primary use case. We should optimize for the main workflow.",
            conflict_context="Choosing which LLM to use for a reasoning-heavy application"
        )
        print(f"    Approach: {approach}")
        print(f"    Synthesis: {synthesis[:150]}...")
        print(f"    Action items: {len(actions)}\n")

    # Generate report
    report = system.get_communication_report()
    return report


async def main():
    """Main execution: demonstrate communication protocol optimization."""

    # Setup DSPy LLM
    print("\nInitializing DSPy with Gemini...")
    lm = dspy.LM(
        "gemini/gemini-flash-latest",
        api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_tokens=8192
    )

    reflection_lm = dspy.LM(
        "gemini/gemini-flash-latest",
        api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_tokens=8192
    )

    dspy.configure(lm=lm)

    # Option to optimize protocols
    optimize = True  # Set to True to run GEPA optimization

    if optimize:
        optimized_protocols = optimize_communication_protocols(reflection_lm)
        # Update global modules with optimized versions
        global task_requester, data_handoff, router, negotiator
        task_requester = optimized_protocols.get("TaskRequestProtocol", task_requester)
        data_handoff = optimized_protocols.get("DataHandoffProtocol", data_handoff)
        router = optimized_protocols.get("RoutingProtocol", router)
        negotiator = optimized_protocols.get("NegotiationProtocol", negotiator)

    # Run scenarios
    print("\n" + "="*80)
    print("RUNNING MULTI-AGENT SCENARIOS")
    print("="*80)

    system = MultiAgentSystem(lm=lm)

    # Scenario 1: Literature review workflow
    report1 = await run_research_scenario(system, "literature_review")

    # Scenario 2: Model comparison with negotiation
    report2 = await run_research_scenario(system, "model_comparison")

    # Final report
    print("\n" + "="*80)
    print("COMMUNICATION QUALITY REPORT")
    print("="*80 + "\n")

    final_report = system.get_communication_report()
    print(json.dumps(final_report, indent=2))

    # Analyze optimization opportunities
    print("\n" + "="*80)
    print("MAST ANALYSIS: Where to Optimize")
    print("="*80 + "\n")

    mast_summary = final_report["mast_analysis"]
    print(f"Total failures detected: {mast_summary['total_failures']}")
    print(f"Average severity: {mast_summary.get('avg_severity', 'N/A')}")
    print(f"\nFailures by category:")
    for category, count in mast_summary.get("by_category", {}).items():
        print(f"  - {category}: {count}")

    print(f"\nTop failure modes:")
    for item in mast_summary.get("top_failure_modes", []):
        print(f"  - {item['mode']}: {item['count']} occurrences")

    print(f"\nOptimization priorities:")
    for i, priority in enumerate(mast_summary.get("optimization_priority", []), 1):
        print(f"  {i}. {priority}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")

    print("This example demonstrates:")
    print("1. Communication protocols as optimizable DSPy signatures")
    print("2. MAST taxonomy for identifying failure patterns")
    print("3. Model assertions for real-time quality monitoring")
    print("4. GEPA optimization of dialogue structure, not task prompts")
    print("\nUnlike traditional multi-agent systems, we optimize how agents")
    print("TALK to each other, not what they individually produce.")
    print("\nResult: Emergent communication conventions that humans wouldn't design.")


if __name__ == "__main__":
    asyncio.run(main())
