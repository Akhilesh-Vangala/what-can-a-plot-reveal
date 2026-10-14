"""Reader instructions. Mirrors the human protocol of Li & Cook (JCGS 2024):
select one or more plots that are most different (or none), give a reason, rate how different.
The flaw is never named, so readers cannot answer from the prompt alone.
"""

FAMILY_DESCRIPTION = {
    "residual": "Each panel is a residual plot from a fitted model: residuals on the y-axis, fitted values on the x-axis, "
                "with a red horizontal reference line at zero.",
    "reliability": "Each panel is a reliability diagram for a binary classifier: mean predicted probability on the x-axis, "
                   "observed frequency of the positive class on the y-axis, with a dashed diagonal reference line.",
    "learning_curve": "Each panel shows learning curves for a model: training loss (blue) and validation loss (orange) "
                      "over training epochs.",
}

INSTRUCTIONS = """You are shown a lineup of 20 plots, numbered 1 to 20.
{family}

Look at all 20 plots and select the plot or plots that are most different from the others.
If no plot looks noticeably different, select none (return an empty list).

Then:
- give the main reason for your selection: "shape", "outliers", "clusters", "spread", "other", or "none" if you selected nothing;
- rate how different your selected plots are from the others: "not at all", "slightly", "moderately", "very", or "extremely";
- explain briefly what you saw.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_panels": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string", "enum": ["shape", "outliers", "clusters", "spread", "other", "none"]},
        "difference_rating": {"type": "string",
                              "enum": ["not at all", "slightly", "moderately", "very", "extremely"]},
        "explanation": {"type": "string"},
    },
    "required": ["selected_panels", "reason", "difference_rating", "explanation"],
    "additionalProperties": False,
}


def build_prompt(family: str) -> str:
    return INSTRUCTIONS.format(family=FAMILY_DESCRIPTION[family])
