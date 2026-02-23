#!/usr/bin/env python3
"""
Test script to verify grade-aware scoring behavior.
Same essay should get different scores for different grades.
"""

from app.sinhala_ml_v2 import score_sinhala_ml_v2

# Test essay (simple Grade 3 level)
test_essay = "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි."

print("=" * 70)
print("GRADE-AWARE SCORING TEST")
print("=" * 70)
print(f"\nTest Essay: {test_essay}")
print(f"Essay Length: {len(test_essay.split())} words")
print("\n" + "-" * 70)

results = {}
for grade in range(3, 9):
    scores = score_sinhala_ml_v2(test_essay, grade)
    results[grade] = scores["total_14"]
    
    print(f"\nGrade {grade}:")
    print(f"  Richness (5):        {scores['richness_5']}")
    print(f"  Organization (6):    {scores['organization_6']}")
    print(f"  Technical (3):       {scores['technical_3']}")
    print(f"  TOTAL (14):          {scores['total_14']}")

print("\n" + "=" * 70)
print("SUMMARY - Total Score by Grade")
print("=" * 70)

for grade, score in results.items():
    print(f"Grade {grade}: {score:6.2f}")

# Check if scores decrease as grade increases (as expected)
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

grade3_score = results[3]
grade8_score = results[8]
difference = grade3_score - grade8_score
percent_decrease = (difference / grade3_score) * 100

print(f"\nGrade 3 Score: {grade3_score:.2f}")
print(f"Grade 8 Score: {grade8_score:.2f}")
print(f"Difference:    {difference:.2f} ({percent_decrease:.1f}% decrease)")

if grade8_score < grade3_score:
    print(f"\n✅ SUCCESS: Grade 8 essay scores LOWER ({grade8_score:.2f}) than Grade 3 ({grade3_score:.2f})")
    print(f"   This is correct because Grade 8 has stricter expectations.")
else:
    print(f"\n❌ FAIL: Grade 8 essay does NOT score lower than Grade 3")

# Check monotonic decrease
is_monotonic = all(results[i] >= results[i+1] for i in range(3, 8))
if is_monotonic:
    print("\n✅ BONUS: Scores decrease monotonically as grade increases (ideal behavior)")
else:
    print("\n⚠️  WARNING: Scores don't decrease monotonically (some inconsistency)")

print("\n" + "=" * 70)
