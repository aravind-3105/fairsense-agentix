"""Manual test script for VLM graph using real bias images.

This script loads a real bias image from fixtures and tests the VLM graph.
Useful for quick manual testing during development.
"""

import sys
from pathlib import Path

from fairsense_agentix.graphs.bias_image_vlm_graph import create_bias_image_vlm_graph


# Load a real test image from fixtures
fixture_path = (
    Path(__file__).parent / "tests" / "fixtures" / "bias_images" / "gender_bias.jpg"
)

if not fixture_path.exists():
    print(f"❌ Test image not found at: {fixture_path}")
    print("Available images in bias_images:")
    bias_images_dir = fixture_path.parent
    if bias_images_dir.exists():
        for img in bias_images_dir.glob("*.jpg"):
            print(f"  - {img.name}")
    sys.exit(1)

print(f"Loading test image: {fixture_path.name}")
with open(fixture_path, "rb") as f:
    image_bytes = f.read()

print(f"✓ Loaded {len(image_bytes)} bytes")

# Create VLM graph
print("\nCreating VLM graph...")
graph = create_bias_image_vlm_graph()
print("✓ Graph created successfully")

# Invoke graph with real image
print("\nInvoking graph with real bias image...")
print("(This will make a real API call to OpenAI/Anthropic)")

result = graph.invoke(
    {
        "image_bytes": image_bytes,
        "options": {"temperature": 0.3},
    }
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Display results
vlm_analysis = result["vlm_analysis"]
bias_analysis = vlm_analysis.bias_analysis

print("\n📊 Visual Description:")
print(f"   {vlm_analysis.visual_description}")

print(f"\n🔍 Bias Detected: {bias_analysis.bias_detected}")
print(f"⚠️  Risk Level: {bias_analysis.risk_level.upper()}")

if bias_analysis.bias_detected:
    print(f"\n🚨 Bias Instances ({len(bias_analysis.bias_instances)}):")
    for i, instance in enumerate(bias_analysis.bias_instances, 1):
        print(f"\n   {i}. Type: {instance.type}")
        print(f"      Severity: {instance.severity}")
        print(f"      Visual Element: {instance.visual_element or instance.text_span}")
        print(f"      Explanation: {instance.explanation}")

print("\n📝 Overall Assessment:")
print(f"   {bias_analysis.overall_assessment}")

print("\n💭 Reasoning Trace:")
print(f"   {vlm_analysis.reasoning_trace[:200]}...")

print("\n📄 Summary:")
print(f"   {result['summary']}")

print("\n🖼️  Image Base64 (for UI):")
if result.get("image_base64"):
    preview = (
        result["image_base64"][:80] + "..."
        if len(result["image_base64"]) > 80
        else result["image_base64"]
    )
    print(f"   {preview}")
    print(f"   Total length: {len(result['image_base64'])} characters")
else:
    print("   ❌ Not generated")

print("\n✓ Test completed successfully!")
