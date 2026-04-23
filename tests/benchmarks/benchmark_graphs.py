"""Performance benchmarking script for FairSense-AgentiX graphs.

Phase 6.5: Comprehensive performance benchmarking with p50/p95/p99 latency tracking,
memory profiling, and throughput measurement across all graph workflows.

Usage:
    python tests/benchmarks/benchmark_graphs.py --iterations 100

    # Run specific graphs only
    python tests/benchmarks/benchmark_graphs.py --graphs bias_text risk
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
from fairsense_agentix.graphs.risk_graph import create_risk_graph


# ============================================================================
# Benchmark Input Fixtures
# ============================================================================


def get_bias_text_inputs() -> dict[str, dict[str, Any]]:
    """Get bias text benchmark inputs of varying sizes."""
    return {
        "small": {
            "text": "We're seeking a young rockstar developer to join our team.",
            "options": {},
        },
        "medium": {
            "text": """We're hiring a Software Engineer for our fast-paced startup!

Requirements:
- Recent college graduate with fresh perspective
- Native English speaker with no accent
- Able-bodied for our open office environment
- Cultural fit with our team (we're like a brotherhood here)
- Willing to work long hours (this isn't a 9-5 job)

We're looking for young, energetic professionals who can keep up with our \
demanding pace.
The ideal candidate is a digital native who grew up with technology.""",
            "options": {},
        },
        "large": {
            "text": """Senior Leadership Position - Executive Vice President

Our prestigious Fortune 500 company is seeking an Executive Vice President to lead our \
operations division. This is an opportunity for a seasoned professional who has proven \
themselves in the upper echelons of corporate leadership.

ABOUT THE ROLE:
This position requires someone who can command respect in the boardroom and navigate \
complex business relationships. You'll be interfacing with C-suite executives, major \
shareholders, and key stakeholders on a daily basis.

IDEAL CANDIDATE PROFILE:
- Graduated from an Ivy League institution or equivalent top-tier university
- 20+ years of progressive leadership experience in similar organizations
- Someone who naturally fits into our refined corporate culture
- Strong network within elite business circles
- Impeccable communication skills (native English proficiency essential)
- Must be comfortable representing the company at high-profile events and galas

REQUIREMENTS:
- Traditional business education (MBA from prestigious institution preferred)
- Proven track record working in established, legacy companies
- Cultural alignment with our traditional corporate values
- Ability to work extended hours and travel extensively
- Perfect vision and hearing for critical meetings (no accommodations available)
- Physical capability to golf with clients and attend networking events
- Clean-cut, professional appearance meeting corporate standards

COMPENSATION & BENEFITS:
We offer a compensation package commensurate with the elite nature of this position. \
This role provides access to our executive dining facilities, country club membership, \
and other privileges befitting senior leadership. We maintain very high standards and \
expect the same from our leadership team.

Our company culture values those who have earned their position through years \
of dedication in established corporate environments. We seek individuals who \
understand the importance of hierarchy, tradition, and maintaining our \
company's prestigious reputation in the business community.

The successful candidate will be someone who doesn't require special \
accommodations, can work the demanding hours expected at this level, and fits \
naturally into our executive team's dynamics. We're looking for a proven leader \
who embodies the values and standards that have made our company successful \
for over a century.

NOTE: This position requires full-time presence in our corporate headquarters. \
Remote work arrangements are not available for executive positions. Candidates \
should be prepared for a rigorous interview process including presentations to \
our board of directors.""",
            "options": {},
        },
    }


def get_bias_image_inputs() -> dict[str, dict[str, Any]]:
    """Get bias image benchmark inputs of varying sizes."""
    # For benchmarking, use fake image bytes of different sizes
    return {
        "small": {
            "image_bytes": b"fake_image_data_small" * 100,  # ~2KB
            "options": {},
        },
        "medium": {
            "image_bytes": b"fake_image_data_medium" * 10000,  # ~200KB
            "options": {},
        },
        "large": {
            "image_bytes": b"fake_image_data_large" * 100000,  # ~2MB
            "options": {},
        },
    }


def get_risk_inputs() -> dict[str, dict[str, Any]]:
    """Get risk assessment benchmark inputs of varying sizes."""
    return {
        "small": {
            "scenario_text": (
                "Deploying facial recognition system in public spaces for "
                "security monitoring."
            ),
            "options": {"top_k": 3},
        },
        "medium": {
            "scenario_text": """AI-Powered Hiring System Deployment

We are developing an AI system to automate our hiring process. The system will:
- Screen resumes and rank candidates automatically
- Conduct initial video interviews with AI analysis
- Predict candidate performance and cultural fit
- Generate hiring recommendations for managers

The system will be trained on our historical hiring data from the past 20 years.
We expect to process thousands of applications per month through this system.""",
            "options": {"top_k": 5},
        },
        "large": {
            "scenario_text": """Comprehensive Healthcare AI Decision Support System

OVERVIEW:
Our organization is implementing a large-scale AI-powered decision support system
across our network of 50+ hospitals. This system will integrate with electronic
health records and provide real-time clinical decision support.

SYSTEM CAPABILITIES:
1. Diagnostic Assistance
   - Analyze patient symptoms, lab results, and imaging data
   - Suggest differential diagnoses with confidence scores
   - Flag potential conditions that may have been overlooked
   - Provide treatment recommendations based on latest medical literature

2. Treatment Planning
   - Recommend personalized treatment plans based on patient demographics, \
medical history, and genetic data
   - Predict treatment outcomes and potential complications
   - Optimize medication dosing based on patient-specific factors
   - Identify patients at high risk for adverse events

3. Resource Allocation
   - Predict patient admission rates and bed occupancy
   - Optimize staffing schedules based on predicted patient volumes
   - Prioritize patients for procedures based on urgency and expected outcomes
   - Allocate expensive medical equipment and resources

4. Administrative Automation
   - Automate insurance pre-authorization decisions
   - Predict likelihood of payment and flag high-risk accounts
   - Optimize billing codes for maximum reimbursement
   - Generate clinical documentation automatically

TRAINING DATA:
The system is trained on:
- 10 million historical patient records from our hospital network
- Claims data covering diverse patient populations
- Clinical trial data and published medical literature
- Proprietary algorithms developed by our data science team

DEPLOYMENT SCOPE:
- Emergency departments (triage and initial assessment)
- Intensive care units (continuous monitoring and alerting)
- Operating rooms (surgical risk assessment)
- Outpatient clinics (routine care recommendations)
- Radiology departments (automated image analysis)
- Pharmacy (medication interaction checking and dosing)

The system will process over 100,000 patient encounters per month and will \
be integrated into critical clinical workflows. Healthcare providers will rely \
on the system's recommendations for time-sensitive medical decisions.""",
            "options": {"top_k": 10, "rmf_per_risk": 5},
        },
    }


# ============================================================================
# Benchmarking Functions
# ============================================================================


def measure_memory() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    """Calculate p50, p95, p99 percentiles from timing data."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}

    sorted_values = sorted(values)
    return {
        "p50": statistics.median(sorted_values),
        "p95": sorted_values[int(len(sorted_values) * 0.95)]
        if len(sorted_values) > 1
        else sorted_values[0],
        "p99": sorted_values[int(len(sorted_values) * 0.99)]
        if len(sorted_values) > 1
        else sorted_values[0],
        "mean": statistics.mean(sorted_values),
        "std": statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        "min": min(sorted_values),
        "max": max(sorted_values),
    }


def benchmark_graph(
    graph_name: str,
    create_graph_fn: callable,
    inputs: dict[str, dict[str, Any]],
    iterations: int,
) -> dict[str, Any]:
    """Benchmark a single graph with different input sizes.

    Parameters
    ----------
    graph_name : str
        Name of the graph being benchmarked
    create_graph_fn : callable
        Function to create the graph instance
    inputs : dict
        Dictionary of input size -> input data mappings
    iterations : int
        Number of iterations to run for each input size

    Returns
    -------
    dict
        Benchmark results with timing and memory metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {graph_name}")
    print(f"{'=' * 70}")

    results = {
        "graph_name": graph_name,
        "iterations": iterations,
        "input_sizes": {},
    }

    # Create graph once (reuse for all iterations)
    graph = create_graph_fn()

    for size_name, input_data in inputs.items():
        print(f"\n  Input size: {size_name} ({iterations} iterations)")

        timings = []
        memory_before = []
        memory_after = []
        errors = 0

        for i in range(iterations):
            try:
                # Measure memory before
                mem_before = measure_memory()
                memory_before.append(mem_before)

                # Time the execution
                start_time = time.perf_counter()
                _ = graph.invoke(input_data)  # Result not needed for benchmarking
                end_time = time.perf_counter()

                duration = end_time - start_time
                timings.append(duration)

                # Measure memory after
                mem_after = measure_memory()
                memory_after.append(mem_after)

                # Progress indicator
                if (i + 1) % max(1, iterations // 10) == 0:
                    print(f"    Progress: {i + 1}/{iterations} ({duration:.3f}s)")

            except Exception as e:
                errors += 1
                print(f"    Error in iteration {i + 1}: {e}")

        # Calculate statistics
        if timings:
            timing_stats = calculate_percentiles(timings)
            memory_delta = [
                after - before
                for before, after in zip(memory_before, memory_after, strict=True)
            ]
            memory_stats = calculate_percentiles(memory_delta)

            results["input_sizes"][size_name] = {
                "latency_seconds": timing_stats,
                "memory_delta_mb": memory_stats,
                "throughput_ops_per_sec": 1.0 / timing_stats["mean"]
                if timing_stats["mean"] > 0
                else 0.0,
                "success_rate": (iterations - errors) / iterations,
                "total_errors": errors,
            }

            print("\n    Results:")
            p50 = timing_stats["p50"]
            p95 = timing_stats["p95"]
            p99 = timing_stats["p99"]
            print(
                f"      Latency (p50/p95/p99): {p50:.3f}s / {p95:.3f}s / {p99:.3f}s",
            )
            print(f"      Memory delta (mean): {memory_stats['mean']:.2f} MB")
            _tp = results["input_sizes"][size_name]["throughput_ops_per_sec"]
            print(f"      Throughput: {_tp:.2f} ops/sec")
            _sr = results["input_sizes"][size_name]["success_rate"]
            print(f"      Success rate: {_sr:.1%}")
        else:
            print(f"    No successful iterations for {size_name}")

    return results


def run_benchmarks(
    graphs_to_benchmark: list[str],
    iterations: int,
    output_path: Path | None,
) -> dict[str, Any]:
    """Run benchmarks for specified graphs.

    Parameters
    ----------
    graphs_to_benchmark : list[str]
        List of graph names to benchmark
    iterations : int
        Number of iterations per input size
    output_path : Path | None
        Optional path to save results JSON

    Returns
    -------
    dict
        Complete benchmark results
    """
    print(f"\n{'#' * 70}")
    print("# FairSense-AgentiX Performance Benchmarks")
    print("# Phase 6.5: Comprehensive performance profiling")
    print("#")
    print("# Configuration:")
    print(f"#   Graphs: {', '.join(graphs_to_benchmark)}")
    print(f"#   Iterations per input size: {iterations}")
    print(f"#   LLM Provider: {settings.llm_provider}")
    print(f"#   Telemetry enabled: {settings.telemetry_enabled}")
    print(f"{'#' * 70}")

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "iterations": iterations,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model_name,
            "telemetry_enabled": settings.telemetry_enabled,
            "cache_enabled": settings.cache_enabled,
        },
        "graphs": {},
    }

    # Benchmark each requested graph
    graph_configs = {
        "bias_text": (create_bias_text_graph, get_bias_text_inputs()),
        "bias_image": (create_bias_image_graph, get_bias_image_inputs()),
        "risk": (create_risk_graph, get_risk_inputs()),
        "orchestrator": (
            create_orchestrator_graph,
            {
                "text": {
                    "input_type": "text",
                    "content": "Test bias in text",
                    "options": {},
                },
                "image": {
                    "input_type": "image",
                    "content": b"fake_image_bytes" * 1000,
                    "options": {},
                },
                "risk": {
                    "input_type": "risk",
                    "content": "AI deployment scenario",
                    "options": {},
                },
            },
        ),
    }

    for graph_name in graphs_to_benchmark:
        if graph_name not in graph_configs:
            print(f"\nWarning: Unknown graph '{graph_name}', skipping...")
            continue

        create_fn, inputs = graph_configs[graph_name]
        result = benchmark_graph(graph_name, create_fn, inputs, iterations)
        all_results["graphs"][graph_name] = result

    # Save results if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 70}\n")

    return all_results


# ============================================================================
# CLI Interface
# ============================================================================


def main() -> None:
    """Run benchmark suite from command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark FairSense-AgentiX graph performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=["bias_text", "bias_image", "risk", "orchestrator", "all"],
        default=["all"],
        help="Graphs to benchmark (default: all)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per input size (default: 20)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output file for results (default: benchmark_results.json)",
    )

    args = parser.parse_args()

    # Determine which graphs to benchmark
    if "all" in args.graphs:
        graphs_to_benchmark = ["bias_text", "bias_image", "risk", "orchestrator"]
    else:
        graphs_to_benchmark = args.graphs

    # Run benchmarks
    results = run_benchmarks(
        graphs_to_benchmark=graphs_to_benchmark,
        iterations=args.iterations,
        output_path=args.output,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for graph_name, graph_results in results["graphs"].items():
        print(f"\n{graph_name.upper()}:")
        for size_name, size_results in graph_results["input_sizes"].items():
            latency = size_results["latency_seconds"]
            lp50 = latency["p50"]
            lp95 = latency["p95"]
            lp99 = latency["p99"]
            print(
                f"  {size_name:8s}: p50={lp50:.3f}s  p95={lp95:.3f}s  p99={lp99:.3f}s",
            )

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
