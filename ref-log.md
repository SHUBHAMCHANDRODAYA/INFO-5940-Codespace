
### Multi-Agent Takeaways
Building this travel planner showed me why having two separate agents matters. The Planner only creates the trip plan, and the Reviewer checks that plan before anyone sees it. Because of that split, the plans are clearer and less likely to contain wrong details. The little tool log in the sidebar also helped because I could watch when the Reviewer used Tavily to fact-check something important.

### Challenges and Fixes
Getting the Planner prompt right was the hardest part. I wanted detailed days without getting a lot unnecessary details, so I used a fixed Markdown layout with tables, daily headers, and short assumption lists. For the Reviewer, the challenge was encouraging fact-checks without wasting Tavily calls. I added rules like “at least one search per city but no more than six” and instructions on what to say when a search fails. I also made the Reviewer give a clear Delta List so any fixes are easy to follow. Another challenge was to give a positive feedback for a good plan which I incorporated in the prompt.

### Creative Choices
I told the Planner to suggest backup activities and money-saving tips so the traveler has options. Each plan also ends with a one-line vibe summary that explains the feel of the trip. I’m using persona-style prompts for both agents and a strict Markdown structure so their outputs stay consistent. I skipped chain-of-thought examples, since the procedural instructions already guide the reasoning. For the Reviewer, I added verdict and risk labels so it feels like a proper sign-off, and I asked it to log every search result in one short line.
