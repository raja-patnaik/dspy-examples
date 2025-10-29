"""
Demonstration: Model Assertions and MAST Monitoring (No API required)

This script demonstrates the MAST monitoring and model assertion concepts
without requiring an API key. Perfect for understanding the system before
running the full example.
"""

import os
os.environ['GEMINI_API_KEY'] = 'dummy'  # Prevent import error

from agent_communication import (
    MASTMonitor,
    MASTFailureMode,
    MASTFailureCategory,
    CommunicationAssertion,
    AGENT_SPECIALTIES,
    AgentRole
)


def demo_assertions():
    """Demonstrate model assertions on sample communications."""
    print("\n" + "="*80)
    print("DEMO: Model Assertions for Communication Quality")
    print("="*80 + "\n")

    # Example 1: Good communication
    print("[Example 1] Clear, well-structured request:")
    good_request = """
    Please search for peer-reviewed papers on transformer architectures published
    between 2023-2024. Deliverables: (1) List of 5-10 papers with titles, authors,
    and publication venues, (2) Brief abstract summary for each. Success criteria:
    Papers should be from reputable ML conferences (NeurIPS, ICML, ICLR, etc.).
    """
    passed, score, msg = CommunicationAssertion.assert_clarity(good_request)
    print(f"Request: {good_request[:100]}...")
    print(f"✓ Clarity assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")

    # Example 2: Poor communication
    print("[Example 2] Vague, unclear request:")
    bad_request = "Please analyze the data"
    passed, score, msg = CommunicationAssertion.assert_clarity(bad_request)
    print(f"Request: {bad_request}")
    print(f"✗ Clarity assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")

    # Example 3: Complete response
    print("[Example 3] Complete response with details:")
    good_response = """
    Analysis complete. Key findings:
    1. Model A shows 85% accuracy on dataset 1
    2. Model B achieves 79% accuracy but processes 2x faster
    3. Statistical significance: p < 0.01 for all comparisons

    Recommendation: Deploy Model A for accuracy-critical applications.
    """
    passed, score, msg = CommunicationAssertion.assert_completeness(
        good_response,
        "analyze model performance"
    )
    print(f"Response: {good_response[:100]}...")
    print(f"✓ Completeness assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")

    # Example 4: Incomplete response
    print("[Example 4] Incomplete response:")
    bad_response = "The models are different."
    passed, score, msg = CommunicationAssertion.assert_completeness(
        bad_response,
        "analyze model performance"
    )
    print(f"Response: {bad_response}")
    print(f"✗ Completeness assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")

    # Example 5: Routing accuracy
    print("[Example 5] Routing decision evaluation:")
    task = "Search for recent papers on neural networks"
    assigned = AgentRole.DATA_COLLECTOR.value
    passed, score, msg = CommunicationAssertion.assert_routing_accuracy(
        task,
        assigned,
        {r.value: AGENT_SPECIALTIES[r] for r in AgentRole}
    )
    print(f"Task: {task}")
    print(f"Assigned to: {assigned}")
    print(f"✓ Routing assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")

    # Example 6: Poor routing
    print("[Example 6] Poor routing decision:")
    task = "Synthesize insights from multiple sources"
    assigned = AgentRole.DATA_COLLECTOR.value  # Wrong agent!
    passed, score, msg = CommunicationAssertion.assert_routing_accuracy(
        task,
        assigned,
        {r.value: AGENT_SPECIALTIES[r] for r in AgentRole}
    )
    print(f"Task: {task}")
    print(f"Assigned to: {assigned}")
    print(f"✗ Routing assertion: {'PASSED' if passed else 'FAILED'}")
    print(f"  Score: {score:.2f}")
    print(f"  Message: {msg}\n")


def demo_mast_monitoring():
    """Demonstrate MAST failure taxonomy and monitoring."""
    print("\n" + "="*80)
    print("DEMO: MAST Failure Taxonomy and Monitoring")
    print("="*80 + "\n")

    monitor = MASTMonitor()

    print("Simulating multi-agent system failures...\n")

    # Simulate various failure modes
    failures = [
        (
            MASTFailureMode.AMBIGUOUS_REQUEST,
            "coordinator",
            "data_collector",
            0.7,
            "Request lacks specific search criteria"
        ),
        (
            MASTFailureMode.INEFFICIENT_ROUTING,
            "coordinator",
            "analyzer",
            0.6,
            "Task routed to wrong specialist"
        ),
        (
            MASTFailureMode.INFORMATION_LOSS,
            "data_collector",
            "analyzer",
            0.8,
            "Critical metadata missing from handoff"
        ),
        (
            MASTFailureMode.INCOMPLETE_RESPONSE,
            "analyzer",
            "synthesizer",
            0.5,
            "Analysis lacks required statistics"
        ),
        (
            MASTFailureMode.AMBIGUOUS_REQUEST,
            "coordinator",
            "synthesizer",
            0.6,
            "Unclear success criteria"
        ),
    ]

    for failure_mode, from_agent, to_agent, severity, message in failures:
        monitor.record_failure(
            failure_mode=failure_mode,
            from_agent=from_agent,
            to_agent=to_agent,
            severity=severity,
            message=message
        )
        print(f"[FAILURE DETECTED] {failure_mode.value}")
        print(f"  From: {from_agent} → To: {to_agent}")
        print(f"  Severity: {severity:.1f}")
        print(f"  Message: {message}\n")

    # Generate summary
    print("="*80)
    print("MAST FAILURE ANALYSIS SUMMARY")
    print("="*80 + "\n")

    summary = monitor.get_failure_summary()

    print(f"Total failures: {summary['total_failures']}")
    print(f"Average severity: {summary['avg_severity']}\n")

    print("Failures by category:")
    for category, count in summary['by_category'].items():
        print(f"  • {category}: {count} failures")

    print("\nTop failure modes:")
    for item in summary['top_failure_modes']:
        print(f"  • {item['mode']}: {item['count']} occurrences")

    print("\nOptimization priorities:")
    for i, protocol in enumerate(summary['optimization_priority'], 1):
        print(f"  {i}. {protocol}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80 + "\n")

    print("This analysis tells us:")
    print("• Communication breakdowns (misalignment) are the primary issue")
    print("• Ambiguous requests occur most frequently")
    print("• Priority: Optimize TaskRequestProtocol first")
    print("• Secondary: Improve routing and data handoff protocols")
    print("\nGEPA would use these insights to focus optimization efforts!")


def demo_failure_categories():
    """Explain the 3 MAST failure categories."""
    print("\n" + "="*80)
    print("MAST TAXONOMY: Understanding the 3 Failure Categories")
    print("="*80 + "\n")

    print("1. SYSTEM DESIGN ISSUES")
    print("   Problems with how the multi-agent system is architected:")
    print("   • Role confusion: Agents unclear about responsibilities")
    print("   • Insufficient context: Missing critical information")
    print("   • Inefficient routing: Tasks sent to wrong specialists")
    print("   → Fix: Optimize RoutingProtocol\n")

    print("2. INTER-AGENT MISALIGNMENT")
    print("   Communication breakdowns between agents:")
    print("   • Information loss: Data lost during handoffs")
    print("   • Ambiguous requests: Unclear what's being asked")
    print("   • Redundant exchanges: Unnecessary back-and-forth")
    print("   • Poor negotiation: Failed conflict resolution")
    print("   → Fix: Optimize TaskRequestProtocol and NegotiationProtocol\n")

    print("3. TASK VERIFICATION")
    print("   Problems with output validation and quality:")
    print("   • Incomplete responses: Missing required information")
    print("   • Format mismatches: Wrong response structure")
    print("   • Unverified handoffs: No confirmation of receipt")
    print("   → Fix: Optimize DataHandoffProtocol\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DSPy + GEPA: Multi-Agent Communication Optimization")
    print("Demonstration Script (No API Key Required)")
    print("="*80)

    demo_assertions()
    demo_mast_monitoring()
    demo_failure_categories()

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80 + "\n")
    print("1. Set your GEMINI_API_KEY environment variable")
    print("2. Run: python agent_communication.py")
    print("3. Watch as GEPA optimizes communication protocols!")
    print("\nThe full example will:")
    print("  • Optimize 4 communication protocols using GEPA")
    print("  • Run 2 multi-agent scenarios")
    print("  • Generate communication quality reports")
    print("  • Identify optimization priorities using MAST")
