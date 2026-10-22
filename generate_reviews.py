import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent
OPENERS = [
    "At my recent visit,",
    "After comparing a few options,",
    "During the service appointment,",
    "For a first-time buyer,",
    "When I picked up the car,",
    "In my experience,",
    "After speaking with the advisor,",
    "For this particular repair,",
]
ENDINGS = {
    "positive": [
        "I would recommend the team.",
        "The experience met my expectations.",
        "I would use this dealership again.",
        "I appreciated the clear communication.",
    ],
    "negative": [
        "I would not recommend this process.",
        "I still need the issue resolved.",
        "I would ask for everything in writing next time.",
        "The experience needs improvement.",
    ],
}
THEMES = {
    "sales_process": {
        "positive": [
            "The salesperson explained every option and gave me time to decide.",
            "The buying process was clear and the final paperwork matched the quote.",
            "Staff answered my questions without pressuring me.",
        ],
        "negative": [
            "The quoted price changed after I arrived at the dealership.",
            "I felt rushed and nobody explained the extra fees.",
            "The salesperson ignored my questions about the vehicle history.",
        ],
    },
    "vehicle_condition": {
        "positive": [
            "The car was clean, inspected, and ready for the test drive.",
            "The vehicle matched the listing and drove smoothly.",
            "Delivery was well prepared and the car looked excellent.",
        ],
        "negative": [
            "There was an undisclosed scratch on the car at delivery.",
            "The vehicle was not ready when I arrived for pickup.",
            "The car had a warning light that was not mentioned in the listing.",
        ],
    },
    "service_quality": {
        "positive": [
            "The service team diagnosed the issue and finished the repair on time.",
            "The technician explained what was fixed in plain language.",
            "The inspection was thorough and the staff kept me informed.",
        ],
        "negative": [
            "The original problem remained after the repair appointment.",
            "The service department did not explain the extra work on my invoice.",
            "I had to book a second visit because the repair was incomplete.",
        ],
    },
    "wait_time": {
        "positive": [
            "My appointment started on time and the inspection was quick.",
            "The team gave me an accurate estimate and finished early.",
            "The waiting area was comfortable during a short service visit.",
        ],
        "negative": [
            "I waited an hour past my appointment with no update.",
            "The repair took all day and nobody called to explain the delay.",
            "The check-in line was long and the promised pickup time was missed.",
        ],
    },
    "communication": {
        "positive": [
            "The service advisor sent clear updates throughout the repair.",
            "I received a prompt callback and helpful answers.",
            "The team explained the warranty and next steps clearly.",
        ],
        "negative": [
            "Nobody returned my call about the repair estimate.",
            "I had to repeat my issue to several different employees.",
            "The dealership gave conflicting updates about when the car would be ready.",
        ],
    },
    "pricing": {
        "positive": [
            "The invoice matched the written estimate and had no surprise fees.",
            "The advisor explained the cost before starting the repair.",
            "The price was fair and every charge was itemized.",
        ],
        "negative": [
            "The final bill included charges that were never approved.",
            "The repair cost much more than the estimate with no prior call.",
            "Several fees appeared on the contract without explanation.",
        ],
    },
}


def generate_rows(per_group: int, seed: int):
    if per_group < 1:
        raise ValueError("per_group must be at least 1.")
    rng = random.Random(seed)
    rows = []
    for theme, sentiments in THEMES.items():
        for sentiment, examples in sentiments.items():
            for _ in range(per_group):
                text = rng.choice(examples)
                text = f"{rng.choice(OPENERS)} {text[0].lower() + text[1:]}"
                text = f"{text.rstrip('.')} {rng.choice(ENDINGS[sentiment])}"
                rows.append(
                    {
                        "review_id": f"synthetic-{len(rows) + 1:05d}",
                        "review": text,
                        "sentiment": sentiment,
                        "theme": theme,
                        "is_synthetic": "true",
                    }
                )
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate fictional dealership reviews.")
    parser.add_argument("--per-theme-and-sentiment", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "generated_reviews.csv",
    )
    args = parser.parse_args()
    rows = generate_rows(args.per_theme_and_sentiment, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} explicitly synthetic reviews to {args.output}")


if __name__ == "__main__":
    main()
