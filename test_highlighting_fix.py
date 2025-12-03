"""Test highlighting fix with fallback span generation."""

from fairsense_agentix import FairSense


print("Testing Issue #3a fix: Highlighting with fallback span generation\n")
print("=" * 70)

# Test with job posting text
text = """Senior Developer Position - Elite Team

We are building a rockstar development team and need young,
energetic ninja coders who can meet the long hours required in
our startup environment.

Must-Have Qualifications:
- Recent CS graduate from a top university
- Able-bodied and healthy - no accommodations available
- Willing to work 60+ hour weeks and weekends

Our brotherhood of developers is seeking someone who fits our
culture: smart guys who can handle pressure, stay late, and
socialize with the team after work. This role requires physical
presence in office - remote work is for people who can't commit.

The ideal candidate is a young man straight out of Stanford or
MIT. Someone from a good family who understands professional
expectations. We're not interested in whose mothers but in their
guys or anyone requiring special treatment.

Benefits include free Fridays and a salary commensurate with
elite university pedigrees.
"""

print(f"Analyzing text ({len(text)} chars) for bias...\n")

# Initialize FairSense
fs = FairSense()

# Run analysis
result = fs.analyze_text(text)

print("✓ Analysis completed!")
print(f"  Bias Detected: {result.bias_detected}")
print(f"  Risk Level: {result.risk_level}")
print(
    f"  Instances Found: {len(result.bias_instances) if result.bias_instances else 0}"
)

if result.bias_instances:
    print("\n📝 Bias Instances (first 5):")
    for i, instance in enumerate(result.bias_instances[:5], 1):
        print(f'   {i}. [{instance["type"].upper()}] "{instance["text_span"]}"')
        print(f"      Severity: {instance['severity']}")
        print(
            f"      Positions: start={instance['start_char']}, end={instance['end_char']}"
        )

# Check highlighted HTML
print("\n🎨 Highlighted HTML:")
if result.highlighted_html:
    html_preview = result.highlighted_html[:500]
    print(f"   Length: {len(result.highlighted_html)} chars")
    print(f"   Preview: {html_preview}...")

    # Check if HTML contains span highlights
    if '<span style="background-color:' in result.highlighted_html:
        span_count = result.highlighted_html.count('<span style="background-color:')
        print(f"\n   ✅ SUCCESS! Found {span_count} highlighted spans in HTML")
        print("   🎉 Biased text portions ARE being highlighted with colors!")
    else:
        print("\n   ❌ WARNING: No <span> highlights found in HTML")
        print("   The HTML contains plain text but no colored highlights")
else:
    print("   ❌ No highlighted HTML generated")

print("\n" + "=" * 70)
print("Test complete!")
