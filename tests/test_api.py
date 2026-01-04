from fastapi.testclient import TestClient
from app.main import app
from app.grade_detector import detect_grade, infer_grade_from_text

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_score_sinhala_ml_with_explicit_grade():
    """Test scoring with explicitly provided grade."""
    payload = {
        "text": "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි.",
        "grade": 7,
        "topic": "පරිසරය",
        "dyslexic_flag": False,
        "error_tags": []
    }

    r = client.post("/score-sinhala-ml", json=payload)
    assert r.status_code == 200

    js = r.json()

    # Core contract
    assert "score" in js
    assert 0 <= js["score"] <= 100

    # Rubric-level scores
    assert "rubric" in js
    assert "total_14" in js["rubric"]

    # Metadata
    assert "details" in js
    assert js["details"]["model"].startswith("xlm-roberta")
    
    # Check grade detection metadata
    assert "detected_grade" in js["details"]
    assert js["details"]["detected_grade"] == 7  # Should use provided grade
    assert js["details"]["grade_auto_detected"] == False

    # Fairness report exists (system-level, may be None)
    assert "fairness_report" in js


def test_score_sinhala_ml_with_auto_detected_grade():
    """Test scoring with auto-detected grade (no grade provided)."""
    payload = {
        "text": "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි. පරිසරය ගැන ක්‍රියා කිරීම ඉතා වැදගත්ය. අපට එය බේරා ගත හැක.",
        "topic": "පරිසරය",
        "dyslexic_flag": False,
        "error_tags": []
    }

    r = client.post("/score-sinhala-ml", json=payload)
    assert r.status_code == 200

    js = r.json()

    # Core contract
    assert "score" in js
    assert 0 <= js["score"] <= 100

    # Rubric-level scores
    assert "rubric" in js
    assert "total_14" in js["rubric"]

    # Metadata
    assert "details" in js
    assert js["details"]["model"].startswith("xlm-roberta")
    
    # Check grade detection metadata
    assert "detected_grade" in js["details"]
    assert 3 <= js["details"]["detected_grade"] <= 8  # Should be valid grade
    assert js["details"]["grade_auto_detected"] == True

    # Fairness report exists (system-level, may be None)
    assert "fairness_report" in js


def test_grade_detection_functions():
    """Test the grade detection module directly."""
    
    # Test empty text (should default to grade 3)
    assert detect_grade("") == 3
    
    # Test short text (should be lower grade)
    short_text = "මම හොඳ ශිෂ්‍යෙකි."
    grade_short = detect_grade(short_text)
    assert 3 <= grade_short <= 5
    
    # Test longer text (should be higher grade)
    long_text = "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි. පරිසරය ගැන ක්‍රියා කිරීම ඉතා වැදගත්ය. අපට එය බේරා ගත හැකි වෙයි. එම නිසා අපි පසුවිට ඉතා ප්‍රධාන කරුණු නිරීක්ෂණ කළ යුතුය."
    grade_long = detect_grade(long_text)
    assert 5 <= grade_long <= 8
    
    # Test infer_grade_from_text with explicit grade
    explicit_grade = infer_grade_from_text("test text", 6)
    assert explicit_grade == 6
    
    # Test infer_grade_from_text with None (should detect)
    detected_grade = infer_grade_from_text(long_text, None)
    assert 3 <= detected_grade <= 8
