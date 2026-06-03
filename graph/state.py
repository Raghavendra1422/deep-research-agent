from typing import TypedDict, List, Optional

class ResearchState(TypedDict):
    # Input
    user_query: str                        # "AI semiconductor market in India"

    # Planning
    sub_topics: List[str]                  # ["latest trends", "govt policies", ...]

    # Research
    raw_search_results: List[dict]         # Raw results from Serper
    scraped_content: List[dict]            # Scraped page content

    # Processing
    extracted_facts: List[dict]            # Structured facts per sub-topic

    # Output
    final_report: Optional[str]            # Final markdown report
    current_step: str                      # Track where we are
    error: Optional[str]                   # Any error message