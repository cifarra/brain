# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the Experiment 0 synthetic data (deterministic, seed 0).

Outputs (in lab/exp0/data/):
  clef_records.jsonl  ~150 systemone requests for the S1 calibration gate (all fit ub=4096)
  s1_workload.jsonl   systemone requests in three size buckets (~500, ~1.8k, ~6k prompt tokens)
  s2_prompts.jsonl    30 short chat prompts for the S2 next-token-distribution quality proxy

Token counts here are estimates (~1.35 tokens/word); the bench records the server's real
usage.input_tokens per request.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
TOK_PER_WORD = 1.35

SUBJECTS = [
    ("beach_drone", "Aerial drone pass over a turquoise beach at sunrise, slow forward glide", "coastline, morning, travel, calm"),
    ("city_night_traffic", "Timelapse of headlights streaking along a downtown avenue at night", "city, pace, modern life, rush"),
    ("chef_onions", "Close-up of a chef rapidly chopping onions on a wooden board", "cooking, skill, kitchen, preparation"),
    ("kids_football", "Children playing football on a dusty village field, handheld", "childhood, joy, community, sport"),
    ("server_racks", "Dolly shot down an aisle of blinking server racks in a cold data center", "technology, cloud, infrastructure, data"),
    ("rain_window", "Raindrops running down a window with blurred city lights behind", "melancholy, waiting, weather, reflection"),
    ("mountain_hike", "Two hikers reaching a rocky summit above the clouds, wide shot", "achievement, nature, effort, adventure"),
    ("robot_arms", "Industrial robot arms welding car frames on an assembly line", "automation, manufacturing, precision, industry"),
    ("farmer_harvest", "Farmer harvesting wheat with a combine at golden hour", "agriculture, food, labor, seasons"),
    ("hospital_corridor", "Nurse walking briskly down a bright hospital corridor, tracking shot", "health care, urgency, care, medicine"),
    ("coffee_pour", "Slow-motion pour of milk into an espresso, latte art forming", "coffee, morning routine, craft, comfort"),
    ("laptop_typing", "Hands typing on a laptop keyboard in a dim room, screen glow on face", "work, coding, focus, remote work"),
    ("concert_crowd", "Crowd jumping at an outdoor concert with confetti and stage lights", "music, celebration, energy, youth"),
    ("train_passing", "High-speed train passing a countryside crossing, locked-off", "travel, speed, transport, connection"),
    ("grandma_knitting", "Elderly woman knitting by a window, soft natural light", "family, patience, tradition, home"),
    ("dog_running", "Golden retriever running through a park toward the camera", "pets, happiness, freedom, play"),
    ("snowstorm", "Car crawling through a heavy snowstorm on a mountain road", "danger, winter, isolation, endurance"),
    ("market_stall", "Vendor arranging fruit at a busy open-air market", "commerce, local economy, color, food"),
    ("solar_panels", "Drone rise over rows of solar panels in a desert", "energy, climate, renewables, future"),
    ("wind_turbines", "Wind turbines turning on a ridge at dusk, telephoto", "energy, climate, scale, wind"),
    ("lab_microscope", "Scientist adjusting a microscope in a research lab, shallow focus", "science, research, discovery, biology"),
    ("classroom", "Teacher writing equations on a chalkboard while students take notes", "education, learning, school, math"),
    ("kayak_river", "Kayaker paddling through white-water rapids, GoPro angle", "adventure, risk, water, sport"),
    ("wedding_dance", "Couple dancing at a wedding reception under string lights", "love, celebration, family, romance"),
    ("printing_press", "Newspapers rolling off a printing press at high speed", "journalism, news, history, media"),
    ("bakery_bread", "Baker pulling loaves of bread from a wood-fired oven", "food, craft, warmth, tradition"),
    ("street_musician", "Street musician playing violin in a subway station, people passing", "art, city, solitude, music"),
    ("departures_board", "Airport departures board flipping, travelers blurred in foreground", "travel, delay, departure, waiting"),
    ("cyclist_city", "Cyclist weaving through morning traffic, POV from handlebar", "commute, city, health, mobility"),
    ("fireworks", "Fireworks bursting over a harbor on New Year's Eve", "celebration, new beginnings, night, crowd"),
    ("desert_road", "Empty desert highway stretching to the horizon, static wide", "freedom, isolation, road trip, emptiness"),
    ("stock_screens", "Trading screens with red and green tickers, rack focus", "finance, markets, volatility, money"),
    ("protest_march", "Peaceful protest march with handmade signs, wide street shot", "activism, politics, community, change"),
    ("ice_melting", "Macro timelapse of an ice block melting on dark rock", "climate change, time, loss, nature"),
    ("factory_workers", "Shift change at a textile factory gate, workers streaming out", "labor, economy, routine, industry"),
    ("child_reading", "Child reading a picture book under a blanket with a flashlight", "imagination, childhood, learning, night"),
    ("surgeon_hands", "Surgeon's gloved hands during an operation, top-down", "medicine, precision, life, expertise"),
    ("vineyard", "Rows of vines on a hillside vineyard, slow pan, autumn colors", "wine, agriculture, terroir, harvest"),
    ("phone_scroll", "Teenager scrolling a phone in bed, face lit by screen", "social media, attention, youth, insomnia"),
    ("ship_port", "Container ship being unloaded by cranes at a busy port", "trade, globalization, logistics, scale"),
]

PARAPHRASE = {
    "beach_drone": ["the sun rose over the quiet shore", "the coast woke up slowly", "waves met the sand at first light"],
    "city_night_traffic": ["the city never stops moving after dark", "rush hour bleeds into the night", "millions of commuters stream home"],
    "chef_onions": ["every great dish starts with preparation", "the kitchen moves at the speed of the knife", "mise en place is everything"],
    "kids_football": ["on the field, the kids forgot everything else", "a ball and an afternoon were enough", "the village gathered to watch the game"],
    "server_racks": ["your photos live in a building you'll never see", "the cloud is really just rooms full of machines", "data centers hum day and night"],
    "rain_window": ["she waited for a call that never came", "the weather matched the mood", "outside, the city blurred into the rain"],
    "mountain_hike": ["after six hours they finally reached the top", "the summit made every step worth it", "above the clouds, the climb was forgotten"],
    "robot_arms": ["machines now build the machines we drive", "precision at a scale no human can match", "automation transformed the factory floor"],
    "farmer_harvest": ["the harvest decides the whole year", "bread starts in a field like this one", "the season ended with the last row cut"],
    "hospital_corridor": ["night shifts are where the real work happens", "the nurses keep the hospital running", "every minute counts on the ward"],
    "coffee_pour": ["the day begins with a small ritual", "a perfect cup takes practice", "the barista's craft hides in the details"],
    "laptop_typing": ["the startup was built at a kitchen table", "she wrote code until the sun came up", "remote work changed where we do our jobs"],
    "concert_crowd": ["fifty thousand people sang the same chorus", "the festival exploded into color", "for one night, nobody felt alone"],
    "train_passing": ["the new line cut travel time in half", "the countryside flashed past at 300 km/h", "rail connected the two cities"],
    "grandma_knitting": ["she learned the pattern from her own mother", "some traditions pass through the hands", "patience was her greatest gift"],
    "dog_running": ["he ran to the door every evening", "pure joy on four legs", "the dog didn't care about the weather"],
    "snowstorm": ["the pass closed an hour after they crossed", "winter made the road a gamble", "visibility dropped to almost nothing"],
    "market_stall": ["the market has sold the same fruit for a century", "local growers set up before dawn", "prices are still haggled by hand"],
    "solar_panels": ["the desert now powers half the region", "sunlight became the cheapest electricity ever", "row after row of panels drinks the sun"],
    "wind_turbines": ["the ridge turns wind into power for a town", "the turbines spin quietly at dusk", "clean energy at the scale of hills"],
    "lab_microscope": ["the breakthrough came from a single slide", "researchers spent years on one protein", "discovery happens at the smallest scale"],
    "classroom": ["the teacher made algebra feel like a puzzle", "learning starts with a blank board", "the exam was only a week away"],
    "kayak_river": ["the rapids tested everything they practiced", "one mistake and the river wins", "white water is a conversation with fear"],
    "wedding_dance": ["they danced as if no one was watching", "two families became one that night", "the first dance lasted three minutes and forever"],
    "printing_press": ["the morning edition went to press at midnight", "news used to smell of ink", "the presses ran for a hundred years"],
    "bakery_bread": ["the bakery has opened at 4am since 1952", "warm bread brings the street together", "the oven never really cools"],
    "street_musician": ["most people never stopped to listen", "a violin echoed through the tunnel", "music for strangers in a hurry"],
    "departures_board": ["every flight that day was delayed", "airports are full of beginnings and endings", "the board flickered with new gate numbers"],
    "cyclist_city": ["more people now commute by bike than by car", "the city redesigned its streets for cyclists", "the ride to work became the best part of the day"],
    "fireworks": ["the new year arrived with a roar over the water", "the whole harbor lit up at midnight", "a fresh start, in color"],
    "desert_road": ["there was nothing for two hundred miles", "the road went on, and so did they", "emptiness has its own kind of beauty"],
    "stock_screens": ["the market lost a trillion dollars in a day", "traders watched the numbers fall", "volatility returned with a vengeance"],
    "protest_march": ["thousands marched for cleaner air", "the movement started with a handful of students", "the streets filled with homemade signs"],
    "ice_melting": ["the glaciers are disappearing faster than predicted", "what took millennia to form is gone in decades", "the ice is running out of time"],
    "factory_workers": ["the mill employed half the town", "shift change was the heartbeat of the city", "the workers' union voted that evening"],
    "child_reading": ["her love of stories started under a blanket", "books opened worlds after bedtime", "one more chapter, she always said"],
    "surgeon_hands": ["the operation lasted eleven hours", "steady hands saved his life", "surgery leaves no room for doubt"],
    "vineyard": ["the vintage depends on the autumn rain", "the hillside has grown grapes for 400 years", "harvest season turned the valley gold"],
    "phone_scroll": ["it was 2am and she was still scrolling", "the feed never ends", "teenagers sleep less than any generation before"],
    "ship_port": ["ninety percent of everything arrives by sea", "the port handles a million containers a month", "global trade, one box at a time"],
}

USAGE_NOTES = [
    "Usable range {a}-{b} s; avoid the jump cut at {c} s.",
    "Good as an establishing shot; too slow for montage.",
    "Licensed for web only; do not use in broadcast cut.",
    "Already used in episode {ep}; prefer not to repeat within 3 episodes.",
    "Strong color cast, needs grading to match the warm palette.",
    "Has a visible logo at {c} s, crop or avoid.",
    "Works under voice-over; no on-screen faces to distract.",
    "Shaky first {a} s, stabilize or trim.",
    "Vertical-safe framing; subject stays centered.",
    "Director's favorite; use for an emotional beat if it fits.",
    "Ambient audio usable (natural sound up allowed).",
    "Slow motion at 120 fps; conform to 24 fps before use.",
]

FILLER = [
    "The production notes ask for a calm, documentary tone with long takes and minimal music under narration.",
    "Editors should keep each section between six and twelve seconds unless the narration runs longer.",
    "The client prefers natural light and warm grading; avoid anything that feels like stock footage.",
    "Transitions are straight cuts only, except the final section which may use a slow fade to black.",
    "Previous episodes reused the same aerial shots too often; reviewers flagged the repetition.",
    "Archive footage is allowed when it is clearly labeled and does not contradict the narration.",
    "Make sure no shot shows identifiable faces of minors without a release on file.",
    "The narrator speaks at roughly one hundred and forty words per minute in this episode.",
    "Music cues change at the start of each act; shot energy should follow the music.",
    "Accessibility review requires that on-screen text remains readable for at least three seconds.",
]

TICKET_TOPICS = {
    "billing": ["I was charged twice for my subscription this month", "my invoice shows a plan I never selected", "the refund you promised hasn't arrived", "my card was declined but the charge still appears"],
    "shipping": ["my package has been stuck in transit for nine days", "the tracking number doesn't work", "the courier left the box at the wrong address", "the order arrived with missing items"],
    "technical": ["the app crashes every time I open the camera", "I can't log in after the latest update", "sync stopped working between my phone and laptop", "the export button produces an empty file"],
    "account": ["I want to change the email on my account", "please delete my account and all my data", "someone else seems to be using my account", "I can't reset my password, the link expired"],
    "sales": ["we'd like a quote for 200 seats", "do you offer discounts for nonprofits", "can we get an annual contract with invoicing", "I'd like a demo for my team"],
    "returns": ["I want to return a jacket that doesn't fit", "the blender stopped working after a week, can I exchange it", "how do I print a return label", "the item is not as described, I want my money back"],
    "legal": ["we received a subpoena regarding a user", "your app may be infringing our trademark", "I want a copy of all personal data you hold on me (GDPR)", "please review our data processing agreement"],
    "feedback": ["love the new dark mode, great work", "the redesign made it harder to find settings", "please add keyboard shortcuts", "the onboarding was confusing for my parents"],
    "security": ["I found a vulnerability in your login page", "I received a phishing email pretending to be you", "my API key may have leaked on GitHub", "there were logins from a country I've never visited"],
    "partnerships": ["we'd like to integrate your API in our product", "interested in a co-marketing campaign", "can we become a reseller in Brazil", "we run a podcast and want a sponsorship"],
    "press": ["journalist here, asking for comment on the outage", "we'd like to interview your CEO", "can you confirm the acquisition rumors", "requesting your press kit and logos"],
    "careers": ["I applied for the backend role and never heard back", "do you sponsor visas", "is the design internship still open", "I'd like to withdraw my application"],
}

REVIEWS = {
    "positive": ["Absolutely love it, works exactly as advertised and arrived early.", "Best purchase this year, my whole family uses it every day.", "The quality surprised me for the price, highly recommend.", "Customer service went above and beyond to help me."],
    "negative": ["Broke after two days and support never replied.", "Total waste of money, nothing like the photos.", "Terrible experience, I will never order from them again.", "It overheats and smells like burning plastic."],
    "neutral": ["It does what it says. Nothing more, nothing less.", "Arrived on time. Haven't used it much yet.", "Average product, similar to others in this range.", "Packaging was fine, the product is okay."],
    "mixed": ["Great sound but the battery barely lasts a morning.", "Beautiful design, though the app is buggy.", "Fast delivery, but one part was missing.", "Comfortable, yet the sizing runs very small."],
}

URGENCY_EVENTS = [
    ("the production database is down and no customer can log in", 5),
    ("a customer asks whether we will support dark mode next year", 0),
    ("payroll did not run and 300 employees were not paid today", 5),
    ("a typo on the pricing page says $9 instead of $19", 3),
    ("a VIP client's demo is in two hours and their account is locked", 4),
    ("an intern asks where the coffee filters are", 0),
    ("the nightly backup failed once but the previous one is fine", 2),
    ("a security researcher reports exposed customer emails", 5),
    ("a partner wants to reschedule next month's meeting", 1),
    ("the office printer is jammed again", 1),
    ("a shipment of perishable goods is stuck at customs and expires tomorrow", 4),
    ("a blog post has a broken image", 1),
]

URGENCY_LEVELS_ALL = [
    "no action needed", "can wait until next month", "handle this week", "handle within 48 hours",
    "handle today", "drop everything now", "wake up the on-call engineer", "escalate to executives",
    "page the whole incident team", "declare a company-wide emergency",
]


def words(n_tokens):
    return max(1, int(n_tokens / TOK_PER_WORD))


def pad_text(rng, base, target_tokens, pool):
    """Append filler sentences from pool until ~target tokens."""
    out = [base]
    n = len(base.split())
    goal = words(target_tokens)
    while n < goal:
        s = rng.choice(pool)
        out.append(s)
        n += len(s.split())
    return " ".join(out)


def shot_line(rng, idx, subj):
    sid, desc, tags = subj
    note = rng.choice(USAGE_NOTES).format(a=rng.randint(1, 3), b=rng.randint(5, 12), c=rng.randint(3, 9), ep=rng.randint(2, 40))
    dur = rng.randint(4, 20)
    return f"[{idx}] {sid}_{rng.randint(1, 9):02d}: {desc}. Duration {dur} s. Tags: {tags}. Notes: {note}"


def shot_state(rng, n_shots, n_sections, target_tokens, extra_notes=True):
    subjects = rng.sample(SUBJECTS, n_shots)
    shot_ids = []
    lines = ["## Available shots"]
    for i, s in enumerate(subjects, 1):
        line = shot_line(rng, i, s)
        shot_ids.append(line.split("]")[1].split(":")[0].strip())
        lines.append(line)
    lines.append("")
    lines.append("## Script")
    sections = []
    for k in range(1, n_sections + 1):
        # mostly lines matching one of the shots, sometimes off-topic (ambiguous)
        if rng.random() < 0.8:
            j = rng.randrange(n_shots)
            text = rng.choice(PARAPHRASE[subjects[j][0]])
            match = j
        else:
            other = rng.choice([s for s in SUBJECTS if s not in subjects])
            text = rng.choice(PARAPHRASE[other[0]])
            match = None
        start = 6 * (k - 1)
        line = f"Section {k} ({start // 60}:{start % 60:02d}-{(start + 6) // 60}:{(start + 6) % 60:02d}, narration): \"{text[0].upper() + text[1:]}.\""
        lines.append(line)
        sections.append((k, match))
    state = "\n".join(lines)
    if extra_notes:
        cur = len(state.split()) * TOK_PER_WORD
        if cur < target_tokens:
            notes = pad_text(rng, "## Production notes\n", target_tokens - cur, FILLER)
            state = state + "\n\n" + notes
    return state, subjects, shot_ids, sections


def shot_questions(rng, subjects, shot_ids, sections, n_q, describe=True):
    qs = {}
    picks = rng.sample(sections, min(n_q, len(sections)))
    for k, _ in picks:
        crit = {sid: (f"{s[1]}" if describe and rng.random() < 0.8 else None) for sid, s in zip(shot_ids, subjects)}
        qs[f"section_{k}_shot"] = {
            "type": "choice",
            "instructions": f"Which shot best illustrates the narration of script section {k}?",
            "criteria": crit,
        }
    if rng.random() < 0.4:
        k = picks[0][0]
        qs[f"section_{k}_fit"] = {
            "type": "score",
            "instructions": f"How well does the best available shot match the meaning of section {k}?",
            "criteria": ["no shot fits", "weak fit", "acceptable", "good fit", "perfect fit"],
        }
    if rng.random() < 0.3:
        qs["needs_new_footage"] = {
            "type": "noul",
            "instructions": "Does at least one script section lack a suitable shot, so that new footage must be sourced?",
        }
    return qs


def rec_shot(rng, target_tokens):
    n_shots = rng.randint(2, max(3, min(15, target_tokens // 45)))
    n_sections = rng.randint(2, max(2, min(30, target_tokens // 60)))
    state, subjects, shot_ids, sections = shot_state(rng, n_shots, n_sections, target_tokens)
    qs = shot_questions(rng, subjects, shot_ids, sections, rng.choice([1, 1, 1, 2, 3]))
    return {"state": state, "questions": qs}


def ticket_text(rng, cat, target_tokens):
    msg = rng.choice(TICKET_TOPICS[cat])
    base = f"From: customer #{rng.randint(1000, 99999)}\nSubject: {msg[:40]}\n\nHi, {msg}. "
    history = []
    hist_pool = [
        f"Order #{rng.randint(10000, 99999)} placed on {rng.randint(1, 28)}/{rng.randint(1, 12)}; status: delivered.",
        "Previous ticket closed as resolved by agent Maria.",
        "Customer has been on the Pro plan for three years.",
        "Automated note: account flagged for unusual activity last month.",
        "Customer previously asked about invoices and received a PDF copy.",
        "Satisfaction survey score last quarter: 4/5.",
        "Agent note: customer prefers email over phone.",
        "System: two failed payment attempts were retried successfully.",
    ]
    text = base + "Thanks in advance."
    cur = len(text.split()) * TOK_PER_WORD
    if cur < target_tokens:
        text += "\n\n--- account history ---\n" + pad_text(rng, "", target_tokens - cur, hist_pool)
    return text, history


def rec_route(rng, target_tokens):
    n = rng.randint(2, len(TICKET_TOPICS))
    cats = rng.sample(sorted(TICKET_TOPICS), n)
    true = rng.choice(cats) if rng.random() < 0.85 else rng.choice([c for c in TICKET_TOPICS if c not in cats] or cats)
    text, _ = ticket_text(rng, true, target_tokens)
    desc = {c: (f"questions about {c}" if rng.random() < 0.6 else None) for c in cats}
    return {"state": text, "questions": {"route": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": desc}}}


def rec_sentiment(rng, target_tokens):
    labels = rng.choice([["positive", "negative"], ["positive", "negative", "neutral"], ["positive", "negative", "neutral", "mixed"]])
    true = rng.choice(list(REVIEWS))
    text = rng.choice(REVIEWS[true])
    if target_tokens > 150:
        extra = []
        for _ in range(rng.randint(1, 6)):
            extra.append(rng.choice(REVIEWS[rng.choice([true, true, "neutral", "mixed"])]))
        text = "Review: " + text + " " + " ".join(extra)
        text = pad_text(rng, text, target_tokens, ["Verified purchase.", "Bought it as a gift.", "Used it for about a month now.", "Compared it with two other brands before buying."])
    crit = {l: None for l in labels} if rng.random() < 0.5 else {l: f"the review is {l}" for l in labels}
    return {"state": text, "questions": {"sentiment": {"type": "choice", "instructions": "What is the overall sentiment of this review?", "criteria": crit}}}


def rec_urgency(rng, target_tokens):
    ev, _ = rng.choice(URGENCY_EVENTS)
    n = rng.randint(3, 10)
    levels = URGENCY_LEVELS_ALL[:n] if rng.random() < 0.7 else sorted(rng.sample(URGENCY_LEVELS_ALL, n), key=URGENCY_LEVELS_ALL.index)
    text = f"Incoming report: {ev}."
    text = pad_text(rng, text, target_tokens, ["Reported via the support chat.", "Several users mention it on social media.", "No workaround is known yet.", "The reporter is a long-time customer.", "Monitoring shows normal error rates otherwise."])
    return {"state": text, "questions": {"urgency": {"type": "score", "instructions": "How urgent is this report?", "criteria": levels}}}


NOUL_ITEMS = [
    ("The customer writes: 'I want my money back for the broken blender.'", "Is a refund requested?"),
    ("The customer writes: 'Just wanted to say thanks for the quick delivery!'", "Is a refund requested?"),
    ("Meeting notes: the team agreed to ship on Friday unless QA finds blockers.", "Was a release date agreed?"),
    ("Meeting notes: we discussed several options but postponed the decision.", "Was a release date agreed?"),
    ("Email: 'Can you send me the slides from yesterday?'", "Does the email contain a request?"),
    ("Email: 'FYI, the office will be closed on Monday.'", "Does the email contain a request?"),
    ("Chat: 'this is the third time your app deleted my notes, I'm furious'", "Is the user angry?"),
    ("Chat: 'no worries, I found the setting myself'", "Is the user angry?"),
    ("Post: 'Selling my bike, DM me, cash only, no questions asked, serial number filed off.'", "Could this listing involve stolen goods?"),
    ("Post: 'Selling my old bike, receipt included, pick up in person.'", "Could this listing involve stolen goods?"),
    ("Log: 'disk usage 97% on db-primary, growing 1%/hour'", "Does this need action within a day?"),
    ("Log: 'disk usage 41% on db-replica, stable'", "Does this need action within a day?"),
]


def rec_noul(rng, target_tokens):
    state, q = rng.choice(NOUL_ITEMS)
    state = pad_text(rng, state, target_tokens, ["(context continues)", "Earlier messages were routine.", "No attachments were included.", "The sender is a known contact."])
    qd = {"type": "noul", "instructions": q}
    if rng.random() < 0.4:
        qd["criteria"] = {"true": "yes", "false": "no"}
    return {"state": state, "questions": {"answer": qd}}


def rec_mixed(rng, target_tokens):
    cat = rng.choice(sorted(TICKET_TOPICS))
    text, _ = ticket_text(rng, cat, target_tokens)
    cats = sorted(rng.sample(sorted(TICKET_TOPICS), rng.randint(3, 8)) + ([cat] if rng.random() < 0.8 else []))
    cats = list(dict.fromkeys(cats))
    qs = {
        "route": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {c: None for c in cats}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["can wait", "this week", "today", "right now"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    }
    if rng.random() < 0.5:
        qs["sentiment"] = {"type": "choice", "instructions": "Tone of the message?", "criteria": {"positive": None, "neutral": None, "negative": None}}
    return {"state": text, "questions": qs}


KINDS = [
    ("shot_assign", rec_shot, 60),
    ("route", rec_route, 20),
    ("sentiment", rec_sentiment, 22),
    ("urgency", rec_urgency, 18),
    ("noul", rec_noul, 15),
    ("mixed", rec_mixed, 15),
]


def est_tokens(req):
    """Rough prompt-size estimate of a whole request (state + questions + clef template overhead)."""
    # fitted on measured usage.input_tokens (clef template): ~1.3 tok/state word, ~3.5 tok/question JSON word
    return 1.3 * len(req["state"].split()) + 3.5 * len(json.dumps(req["questions"]).split()) + 100


def sized(rng, fn, target, cap=None):
    """Generate with fn, then once more with the target corrected for the questions' share."""
    st = rng.getstate()
    req = fn(rng, target)
    over = est_tokens(req) - target
    if abs(over) > 0.05 * target:
        rng.setstate(st)
        req = fn(rng, max(20, int(target - over)))
    if cap and est_tokens(req) > cap:
        rng.setstate(st)
        req = fn(rng, max(20, int(target - over - (est_tokens(req) - cap))))
    return req


def gen_records(rng):
    out = []
    i = 0
    for kind, fn, n in KINDS:
        for _ in range(n):
            # mostly short/medium, some long; cap so every record fits ub=4096
            target = int(rng.choice([rng.uniform(60, 400), rng.uniform(300, 1200), rng.uniform(800, 2400), rng.uniform(2000, 3300)]))
            if kind in ("sentiment", "noul", "urgency") and rng.random() < 0.5:
                target = int(rng.uniform(30, 200))
            req = sized(rng, fn, target, cap=3500)
            out.append({"id": f"r{i:03d}", "kind": kind, "target_tokens": target, "request": req})
            i += 1
    return out


def gen_workload(rng):
    out = []
    for bucket, target in (("500", 500), ("2k", 1650), ("6k", 6000)):
        for j in range(8):
            if j % 4 in (0, 1):
                req = sized(rng, rec_shot, target)
                kind = "shot_assign"
            else:
                kind = rng.choice(["route", "mixed", "urgency", "sentiment", "noul"])
                req = sized(rng, KINDS_BY_NAME[kind], target)
            out.append({"id": f"w{bucket}_{j}", "bucket": bucket, "kind": kind, "target_tokens": target, "request": req})
    return out


KINDS_BY_NAME = {k: f for k, f, _ in KINDS}

S2_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Translate to Spanish: 'The meeting has been moved to Thursday afternoon.'",
    "What are the main differences between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give me a haiku about a rainy Monday morning.",
    "List five tips for improving sleep quality.",
    "What is 17 multiplied by 23? Show your reasoning briefly.",
    "Write a short product description for a stainless steel water bottle.",
    "Explain what a hash map is to a beginner programmer.",
    "Describe the water cycle for a ten-year-old.",
    "Draft a polite email declining a meeting invitation.",
    "What causes inflation? Keep it short.",
    "Write a SQL query that returns the ten most recent orders per customer.",
    "Give three arguments for and against remote work.",
    "Explain the difference between a virus and a bacterium.",
    "Suggest a name for a coffee shop that also sells books.",
    "How does a refrigerator keep food cold?",
    "Write a limerick about a cat who learned to code.",
    "Explain gradient descent in simple terms.",
    "What should I pack for a three-day hiking trip?",
    "Rewrite this sentence to be more formal: 'Hey, can u send me the report asap?'",
    "Describe the city of Lisbon in a short paragraph.",
    "What is the capital of Australia and why is it not Sydney?",
    "Write a bash one-liner that counts lines in all .py files recursively.",
    "Explain what a video editor means by a 'J-cut'.",
    "Give a short motivational message for someone starting a new job.",
    "Compare electric cars and hybrid cars in a few bullet points.",
    "Explain what the Unix philosophy is.",
    "¿Cuáles son los beneficios de aprender un segundo idioma? Responde brevemente.",
]


def main():
    rng = random.Random(0)
    DATA.mkdir(exist_ok=True)
    recs = gen_records(rng)
    with open(DATA / "clef_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    wl = gen_workload(rng)
    with open(DATA / "s1_workload.jsonl", "w") as f:
        for r in wl:
            f.write(json.dumps(r) + "\n")
    with open(DATA / "s2_prompts.jsonl", "w") as f:
        for i, p in enumerate(S2_PROMPTS):
            f.write(json.dumps({"id": f"p{i:02d}", "prompt": p}) + "\n")
    nopt = [max((len(q["criteria"]) if q["type"] != "noul" and "criteria" in q else 2) for q in r["request"]["questions"].values()) for r in recs]
    print(f"records={len(recs)} workload={len(wl)} s2_prompts={len(S2_PROMPTS)}; options range {min(nopt)}-{max(nopt)}")


if __name__ == "__main__":
    main()
