import json
d = json.load(open('output/hackathon_agent_submission.json'))
for s in d:
    if s['scenario_id'] == 'P3':
        for c in s.get('covenants', []):
            print(f"{c.get('covenant_id')}: Num={c.get('calculated_numerator')}, Denom={c.get('calculated_denominator')}, Val={c.get('calculated_value')}, Status={c.get('status')}")
