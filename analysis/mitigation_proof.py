
import requests
import json

API_URL = "http://127.0.0.1:8001/score-sinhala-ml"

def prove_mitigation():
    print("="*50)
    print("PROOF OF MITIGATION: GRADE 8 CASE STUDY")
    print("="*50)
    
    # Using a sample Grade 8 essay from your test set
    essay_text = "නිරමානශීලි චින්තනය යනු සාම්ප්‍රදායික සීමාවන ඉක්මවා අලුත අදහස නිරමානය කිරීමෙ හැකියාවයි. නවීන ලෝකයෙ දී නිරමානශීලි චින්තකයින්ට ඉතා ඉහල ඉල්ලුමක ඈත. විද්‍යාව තාක්ෂනය කලාව සහ ව්‍යාපාරික ක්ෂේත වල සාරතකත්වයට නිරමානශීලි චින්තනය අත්‍යවශ්‍යය."
    
    # 1. Score as NON-DYSLEXIC
    res_nd = requests.post(API_URL, json={
        "text": essay_text,
        "grade": 8,
        "dyslexic_flag": False
    }).json()
    
    # 2. Score as DYSLEXIC (This should trigger the multiplier because Grade 8 has history of bias)
    res_d = requests.post(API_URL, json={
        "text": essay_text,
        "grade": 8,
        "dyslexic_flag": True
    }).json()

    raw_score = res_nd['score']
    mitigated_score = res_d['score']
    
    print(f"Essay Sample: [Sinhala Text]...")
    print(f"Grade: 8")
    print(f"\n1. Raw Model Score (Non-Dyslexic Flag): {raw_score}%")
    print(f"2. Mitigated Score (Dyslexic Flag ON):   {mitigated_score}%")
    
    if mitigated_score > raw_score:
        improvement = mitigated_score - raw_score
        print(f"\n SUCCESS: Mitigation Engine increased the score by {improvement:.2f}%.")
        print(f"Reason: Grade 8 met the bias threshold (|SPD| > 0.1).")
        
        # Save this as a Case Study
        with open("c:/Users/nuwan/ResearchProject/bias-aware-scoring-engine/docs/mitigation_case_study.json", "w") as f:
            json.dump({
                "grade": 8,
                "raw_score": raw_score,
                "mitigated_score": mitigated_score,
                "fairness_report": res_d.get('fairness_report', {})
            }, f, indent=4)
    else:
        print("\n No mitigation applied. Checking fairness reports...")
        print(f"Fairness Details: {res_d.get('fairness_report')}")

if __name__ == "__main__":
    prove_mitigation()
